"""Task 2 / 5: debugging GUI.

Sources: live webcam, a recorded / loaded video (frame-by-frame scrubbing), or
a synthetic demo. Optional model inference runs on a worker thread. An
annotation mode turns the same window into a labelling tool for training data.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..annotations import MODEL_FILE, AnnotationStore
from ..audio import AudioTrack, SoundState, wav_for_video, write_wav
from ..camera import CameraStream
from ..config import resolve
from ..feedback import BowFeedback, BowState, Feedback
from ..geometry import CameraIntrinsics, FrameGeometry, GeometryEstimator, ViolinModel
from ..keypoints import KEYPOINT_KEYS, KEYPOINT_NAMES, Keypoints2D
from ..overlay import draw_geometry, draw_keypoints, draw_model_fit
from ..positioning import PositionAdvice, PositionAdvisor
from ..synthetic import DemoSource
from ..video import VideoFile, VideoRecorder
from .keypoint_guide import KeypointGuide
from .mic import MicInput
from .view3d import View3DWindow

SNAP_PX = 15  # bow clicks this close to the frame edge snap onto it (= "cut off by the frame")
GRAB_SCREEN_PX = 12  # pressing this close (on screen) to a placed point drags it...
GRAB_TIGHT_PX = 6  # ...or only this close while the selected point still needs placing (close neighbours)
UNSURE_CONF = 0.6
LOOKAHEAD = 90  # recorded takes: the model runs up to this many frames ahead of the playhead  # model points below this confidence are flagged for checking

BANNER_STYLE = {
    BowState.OK: ("#1f9d3a", "#ffffff"),
    BowState.TIP_TOWARD_SCROLL: ("#c62828", "#ffffff"),
    BowState.TIP_TOWARD_BRIDGE: ("#c62828", "#ffffff"),
    BowState.NO_BOW: ("#000000", "#bbbbbb"),
}
OVERLAY_BGR = {BowState.OK: (60, 210, 60), BowState.NO_BOW: (200, 200, 200)}
POSITION_STYLE = {  # (background, text) for the "how to stand" bar
    "ok": ("#16351f", "#7ee08f"),
    "adjust": ("#4a3200", "#ffc857"),
    "limited": ("#3a2f12", "#e0c27e"),
    "unknown": ("#202020", "#9a9a9a"),
}
STRING_CHOICES = ["Any", "G", "D", "A", "E"]


def _needs_update(kps: dict) -> bool:
    """Saved before some keypoints existed: those points were never labelled."""
    return bool(AnnotationStore.missing(kps))


class Mode(Enum):
    IDLE = auto()
    LIVE = auto()
    VIDEO = auto()
    DEMO = auto()


# ---------------------------------------------------------------------------
# Inference worker (model + geometry off the GUI thread)
# ---------------------------------------------------------------------------
class InferenceWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object, object, object, float)  # token, frame, kp, geo, ms
    failed = QtCore.Signal(str)

    def __init__(self, detector, geometry: GeometryEstimator):
        super().__init__()
        self.detector = detector
        self.geometry = geometry

    @QtCore.Slot(object, object)
    def process(self, token, frame: np.ndarray) -> None:
        try:
            t0 = time.perf_counter()
            kp = self.detector(frame)
            geo = self.geometry(kp, (frame.shape[1], frame.shape[0]))
            self.finished.emit(token, frame, kp, geo, (time.perf_counter() - t0) * 1000)
        except Exception as e:  # surface errors in the GUI instead of killing the thread
            self.failed.emit(f"{type(e).__name__}: {e}")


class VideoLabel(QtWidgets.QLabel):
    """Shows frames scaled to fit; reports clicks in *image* pixel coordinates."""

    clicked = QtCore.Signal(float, float, object)  # x, y, Qt.MouseButton (on press)
    dragged = QtCore.Signal(float, float)  # x, y while the left button is held
    released = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setStyleSheet("background:#111;")
        self._scale, self._off = 1.0, QtCore.QPointF(0, 0)

    def show_bgr(self, img: np.ndarray) -> None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.size(), QtCore.Qt.KeepAspectRatio,
                                                   QtCore.Qt.SmoothTransformation)
        self._scale = pix.width() / w
        self._off = QtCore.QPointF((self.width() - pix.width()) / 2, (self.height() - pix.height()) / 2)
        self.setPixmap(pix)

    def to_image(self, p: QtCore.QPointF) -> tuple[float, float]:
        return (p.x() - self._off.x()) / self._scale, (p.y() - self._off.y()) / self._scale

    @property
    def scale(self) -> float:
        return self._scale

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.clicked.emit(*self.to_image(ev.position()), ev.button())

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.buttons() & QtCore.Qt.LeftButton:
            self.dragged.emit(*self.to_image(ev.position()))

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            self.released.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    request_inference = QtCore.Signal(object, object)

    def __init__(self, cfg: dict, checkpoint: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("bowtrack - violin bow angle")
        self.resize(1400, 860)

        cam_cfg = cfg["camera"]
        self.min_conf = float(cfg["feedback"]["min_confidence"])
        self.model = ViolinModel.from_config(cfg)
        calib = resolve(cam_cfg["calibration_file"])
        if calib.exists():
            self.intrinsics = CameraIntrinsics.load(calib)
            self.calib_note = f"calibrated ({calib.name})"
        else:
            self.intrinsics = CameraIntrinsics.from_fov(cam_cfg["width"], cam_cfg["height"],
                                                        cam_cfg["horizontal_fov_deg"])
            self.calib_note = f"approx. intrinsics from {cam_cfg['horizontal_fov_deg']}° FOV"
        self.geo_local = GeometryEstimator(self.model, self.intrinsics, self.min_conf)
        fb = cfg["feedback"]
        self.feedback = BowFeedback(fb["target_angle_deg"], fb["tolerance_deg"], fb["smoothing_window"])
        self.advisor = PositionAdvisor.from_config(cfg, self.model)
        au = cfg.get("audio", {})
        self._audio_params = (float(au.get("threshold_db", -45.0)), float(au.get("margin_db", 12.0)),
                              float(au.get("hold_s", 0.15)))
        self.mic = MicInput(*self._audio_params, parent=self)
        self.mic.failed.connect(self._on_mic_failed)
        self.audio_track: AudioTrack | None = None
        self.current_sound: SoundState | None = None
        self.current_string: tuple[str | None, str] = (None, "")  # (string, where it came from)

        self.mode = Mode.IDLE
        self.camera: CameraStream | None = None
        self._last_cam_id = 0
        self.recorder = VideoRecorder()
        self.video: VideoFile | None = None
        self.frame_idx = 0
        self._last_shown_idx = -10
        self.playing = False
        self.demo = DemoSource(self.model, self.intrinsics.scaled_to(1280, 720))
        self._demo_t0 = 0.0
        self.current_frame: np.ndarray | None = None
        self.current_kp: Keypoints2D | None = None
        self.current_geo: FrameGeometry | None = None
        self.results: dict[int, tuple[Keypoints2D, FrameGeometry]] = {}
        self._fps_times: list[float] = []
        self._infer_ms = 0.0

        # Model / worker (created lazily when "Apply model" is ticked).
        self.checkpoint = resolve(checkpoint or cfg["model"]["checkpoint"])
        self.worker: InferenceWorker | None = None
        self.worker_thread: QtCore.QThread | None = None
        self._busy = False
        self._pending: tuple | None = None

        # Annotation state.
        self.store = AnnotationStore(resolve("data"))  # approved labels (training data)
        self.model_store = AnnotationStore(resolve("data"), MODEL_FILE)  # model labels to check
        self.annot: dict[str, list[float] | None] = {}
        self.annot_source = "new"  # "approved" | "model" | "new": where the frame's labels came from
        self.annot_dirty = False
        self._drag: str | None = None
        self._last_drag_render = 0.0

        self.view3d = View3DWindow(self.model)
        self._build_ui()
        self._build_shortcuts()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(5)
        self._set_banner(Feedback(BowState.NO_BOW, None, None, "Start the camera, open a video, or run the demo"))

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.video_label = VideoLabel()
        self.video_label.clicked.connect(self._on_image_click)
        self.video_label.dragged.connect(self._on_image_drag)
        self.video_label.released.connect(self._on_image_release)
        self.banner = QtWidgets.QLabel()
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setFixedHeight(64)

        # Source / action buttons.
        self.btn_camera = QtWidgets.QPushButton("Start camera")
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_record = QtWidgets.QPushButton("● Record")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_open = QtWidgets.QPushButton("Load video…")
        self.btn_open.clicked.connect(self.open_video_dialog)
        self.btn_demo = QtWidgets.QPushButton("Synthetic demo")
        self.btn_demo.clicked.connect(self.start_demo)
        self.chk_model = QtWidgets.QCheckBox("Apply model")
        self.chk_model.toggled.connect(self.toggle_model)
        self.spin_tol = QtWidgets.QDoubleSpinBox()
        self.spin_tol.setRange(0.5, 20.0)
        self.spin_tol.setSingleStep(0.5)
        self.spin_tol.setDecimals(1)
        self.spin_tol.setSuffix("°")
        self.spin_tol.setValue(self.feedback.tolerance)
        self.spin_tol.setToolTip("Green while the bow angle is within this of the target (default from "
                                 "feedback.tolerance_deg in the config)")
        self.spin_tol.valueChanged.connect(self._on_tolerance)
        self.chk_fit = QtWidgets.QCheckBox("Show fitted violin")
        self.chk_fit.setChecked(True)
        self.chk_fit.setToolTip("Draw the 3D violin model at the estimated pose over the frame. If the pose is "
                                "right it lines up with the real violin.")
        self.chk_fit.toggled.connect(lambda _: self._render())
        self.btn_3d = QtWidgets.QPushButton("Show 3D view")
        self.btn_3d.clicked.connect(self.show_3d)
        self.chk_sound = QtWidgets.QCheckBox("Sound")
        self.chk_sound.setToolTip("Only judge the bow while a note sounds (the bow is then on a string). "
                                  "Live: microphone. Recordings: the take's .wav.")
        self.chk_sound.setChecked(bool(self.cfg.get("audio", {}).get("enabled", True)))
        self.chk_sound.toggled.connect(self._on_sound_toggled)
        self.combo_string = QtWidgets.QComboBox()
        self.combo_string.addItems(STRING_CHOICES)
        self.combo_string.setToolTip("Optional: practising on one string only (open strings, scales on one string). "
                                     "Any = assume the middle of the strings, which is accurate once the position "
                                     "bar is green.")
        self.combo_string.currentIndexChanged.connect(self._on_string_choice)
        top = QtWidgets.QHBoxLayout()
        for w in (self.btn_camera, self.btn_record, self.btn_open, self.btn_demo):
            top.addWidget(w)
        top.addSpacing(20)
        top.addWidget(self.chk_model)
        top.addWidget(self.chk_fit)
        top.addWidget(self.chk_sound)
        top.addWidget(QtWidgets.QLabel("String:"))
        top.addWidget(self.combo_string)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Tolerance ±"))
        top.addWidget(self.spin_tol)
        top.addStretch(1)
        top.addWidget(self.btn_3d)

        # Playback / scrubbing.
        self.btn_first = QtWidgets.QToolButton(text="⏮")
        self.btn_prev = QtWidgets.QToolButton(text="◀")
        self.btn_play = QtWidgets.QToolButton(text="▶")
        self.btn_next = QtWidgets.QToolButton(text="▶|")
        self.btn_first.clicked.connect(lambda: self.seek(0))
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(1))
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        self.frame_text = QtWidgets.QLabel("—")
        self.frame_text.setMinimumWidth(150)
        play = QtWidgets.QHBoxLayout()
        for w in (self.btn_first, self.btn_prev, self.btn_play, self.btn_next):
            play.addWidget(w)
        play.addWidget(self.slider, 1)
        play.addWidget(self.frame_text)
        self._set_playback_enabled(False)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(top)
        left.addWidget(self.video_label, 1)
        left.addLayout(play)
        left.addWidget(self.banner)
        self.pos_banner = QtWidgets.QLabel()
        self.pos_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.pos_banner.setFixedHeight(34)
        self.pos_banner.setToolTip("Is the violin placed so the bow angle can be measured? Separate from the bow "
                                   "banner above: it can be green while this asks you to turn.")
        self._set_position(PositionAdvice("unknown", "Position: waiting for the violin"))
        left.addWidget(self.pos_banner)

        # Right panel: annotation + info.
        self.chk_annot = QtWidgets.QCheckBox("Annotate mode (A)")
        self.chk_annot.toggled.connect(self.toggle_annotate)
        self.kp_list = QtWidgets.QListWidget()
        for n in KEYPOINT_NAMES:
            self.kp_list.addItem(n)
        self.kp_list.setFixedHeight(12 + 19 * len(KEYPOINT_NAMES))
        self.guide = KeypointGuide(self.model)
        self.kp_list.currentRowChanged.connect(self._on_kp_selected)
        self.kp_list.setCurrentRow(0)
        self.btn_approve = QtWidgets.QPushButton("✓ Approve (Enter)")
        self.btn_approve.setToolTip("Points are correct: move this frame to the approved dataset (used for "
                                    "training) and jump to the next frame to check")
        self.btn_approve.setStyleSheet("font-weight:600;")
        self.btn_approve.clicked.connect(self.approve_frame)
        self.btn_save_annot = QtWidgets.QPushButton("Save (S)")
        self.btn_save_annot.setToolTip("Save the points without changing the frame's group (edits are also "
                                       "saved automatically when you leave the frame)")
        self.btn_save_annot.clicked.connect(self.save_annotation)
        self.btn_clear_annot = QtWidgets.QPushButton("Clear")
        self.btn_clear_annot.clicked.connect(self.clear_annotation)
        self.annot_status = QtWidgets.QLabel()
        self.annot_status.setWordWrap(True)
        help_text = QtWidgets.QLabel(
            "<small>Left-click: place selected point (auto-advances)<br>"
            "Drag a point to move it · Right-click: remove nearest point<br>"
            "N: mark selected as not visible (✗)<br>"
            "1-9, 0, F, T: select point · ✓ placed · · not labelled yet<br>"
            "Space: play/pause · ←/→: step · Shift+←/→: ±10<br>"
            "[ / ]: previous / next labelled frame</small>"
        )
        annot_box = QtWidgets.QGroupBox("Labelling")
        al = QtWidgets.QVBoxLayout(annot_box)
        clear_row = QtWidgets.QHBoxLayout()
        clear_row.addWidget(self.btn_save_annot, 1)
        clear_row.addWidget(self.btn_clear_annot, 1)
        for w in (self.chk_annot, self.kp_list, self.guide):
            al.addWidget(w)
        al.addWidget(self.btn_approve)
        al.addLayout(clear_row)
        al.addWidget(self.annot_status)
        al.addWidget(help_text)

        # Labelled-frame navigator: approved frames / model labels still to check.
        self.approved_list = QtWidgets.QListWidget()
        self.check_list = QtWidgets.QListWidget()
        for lst in (self.approved_list, self.check_list):
            lst.setMinimumHeight(150)
            lst.itemActivated.connect(self._on_labelled_clicked)
            lst.itemClicked.connect(self._on_labelled_clicked)
        self.lab_tabs = QtWidgets.QTabWidget()
        self.lab_tabs.addTab(self.approved_list, "Approved")
        self.lab_tabs.addTab(self.check_list, "To check")
        self.btn_auto_label = QtWidgets.QPushButton("Label frames with model")
        self.btn_auto_label.setToolTip("Run the model on this take and add its labels to 'To check' "
                                       "(frames already labelled are skipped)")
        self.btn_auto_label.clicked.connect(self.auto_label)
        self.spin_every = QtWidgets.QSpinBox()
        self.spin_every.setRange(1, 600)
        self.spin_every.setValue(10)
        self.spin_every.setPrefix("every ")
        self.spin_every.setSuffix(" fr")
        self.spin_every.setToolTip("Label every N-th frame (neighbouring frames are nearly identical)")
        auto_row = QtWidgets.QHBoxLayout()
        auto_row.addWidget(self.btn_auto_label, 1)
        auto_row.addWidget(self.spin_every)
        self.btn_prev_lab = QtWidgets.QPushButton("◀ Prev [")
        self.btn_next_lab = QtWidgets.QPushButton("Next ] ▶")
        self.btn_del_lab = QtWidgets.QPushButton("Delete labels")
        self.btn_del_lab.setToolTip("Remove this frame's labels and saved image (approved or to check)")
        self.btn_prev_lab.clicked.connect(lambda: self.goto_labelled(-1))
        self.btn_next_lab.clicked.connect(lambda: self.goto_labelled(1))
        self.btn_del_lab.clicked.connect(self.delete_labels)
        nav = QtWidgets.QHBoxLayout()
        for w in (self.btn_prev_lab, self.btn_next_lab, self.btn_del_lab):
            nav.addWidget(w)
        lab_box = QtWidgets.QGroupBox("Labelled frames")
        ll = QtWidgets.QVBoxLayout(lab_box)
        ll.addWidget(self.lab_tabs)
        ll.addLayout(nav)
        ll.addLayout(auto_row)

        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("font-family: Menlo, monospace; font-size: 12px;")
        self.info.setAlignment(QtCore.Qt.AlignTop)
        self.info.setWordWrap(True)
        info_box = QtWidgets.QGroupBox("Measurements")
        QtWidgets.QVBoxLayout(info_box).addWidget(self.info)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(annot_box)
        right.addWidget(lab_box)
        right.addWidget(info_box)
        right.addStretch(1)
        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(right_w)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(350)

        root = QtWidgets.QHBoxLayout()
        root.addLayout(left, 1)
        root.addWidget(scroll)
        central = QtWidgets.QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._update_annot_status()
        self._refresh_labelled_list()

    def _build_shortcuts(self) -> None:
        def sc(key, fn):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

        sc("Space", self.toggle_play)
        sc("Right", lambda: self.step(1))
        sc("Left", lambda: self.step(-1))
        sc("Shift+Right", lambda: self.step(10))
        sc("Shift+Left", lambda: self.step(-10))
        sc("A", lambda: self.chk_annot.toggle())
        sc("S", self.save_annotation)
        sc("Return", self.approve_frame)
        sc("Enter", self.approve_frame)
        sc("N", self.mark_not_visible)
        sc("R", lambda: self.btn_record.isEnabled() and self.toggle_record())
        sc("[", lambda: self.goto_labelled(-1))
        sc("]", lambda: self.goto_labelled(1))
        for i, key in enumerate(KEYPOINT_KEYS):
            sc(key, lambda i=i: self.kp_list.setCurrentRow(i))

    def _set_playback_enabled(self, on: bool) -> None:
        for w in (self.btn_first, self.btn_prev, self.btn_play, self.btn_next, self.slider):
            w.setEnabled(on)

    # ------------------------------------------------------------- sources
    def _stop_sources(self) -> None:
        if self.recorder.is_recording:
            self.toggle_record()
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self.mic.stop()
        self.audio_track = None
        self.current_sound = None
        self.btn_camera.setText("Start camera")
        self.playing = False
        self.btn_play.setText("▶")
        self.mode = Mode.IDLE
        self.results.clear()
        self.feedback.reset()
        self.advisor.reset()
        self.geo_local.reset()
        if self.worker is not None:
            self.worker.geometry.reset()
        self._update_annot_status()

    def toggle_camera(self) -> None:
        if self.mode == Mode.LIVE:
            self._stop_sources()
            self.btn_record.setEnabled(False)
            return
        self._stop_sources()
        self.chk_annot.setChecked(False)
        c = self.cfg["camera"]
        try:
            self.camera = CameraStream(c["index"], c["width"], c["height"], c["fps"]).start()
        except RuntimeError as e:
            QtWidgets.QMessageBox.warning(self, "Camera", str(e))
            return
        self.mode = Mode.LIVE
        self._last_cam_id = 0
        if self.chk_sound.isChecked():
            self._start_mic()
        self.btn_camera.setText("Stop camera")
        self.btn_record.setEnabled(True)
        self._set_playback_enabled(False)

    def start_demo(self) -> None:
        self._stop_sources()
        self.chk_annot.setChecked(False)
        self.mode = Mode.DEMO
        self._demo_t0 = time.time()
        self.btn_record.setEnabled(True)
        self._set_playback_enabled(False)
        self.show_3d()

    def open_video_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load video", str(resolve("recordings")), "Video (*.mp4 *.mov *.avi *.mkv)")
        if path:
            self.open_video(path)

    def open_video(self, path: str | Path) -> None:
        self._stop_sources()
        self.btn_record.setEnabled(False)
        try:
            self.video = VideoFile(path)
        except RuntimeError as e:
            QtWidgets.QMessageBox.warning(self, "Video", str(e))
            return
        self.mode = Mode.VIDEO
        wav = wav_for_video(path)
        if wav.exists():
            try:
                self.audio_track = AudioTrack(wav, *self._audio_params)
                self.statusBar().showMessage(f"Loaded sound from {wav.name}", 6000)
            except Exception as e:  # unreadable wav: carry on without sound
                self.statusBar().showMessage(f"Could not read {wav.name}: {e}", 8000)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self.video) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._set_playback_enabled(True)
        self.setWindowTitle(f"bowtrack - {Path(path).name}")
        self._refresh_labelled_list()
        self.seek(0)

    def toggle_record(self) -> None:
        if self.recorder.is_recording:
            path = self.recorder.stop()
            n = self.recorder.frames_written
            samples = self.mic.stop_recording()
            if path and samples is not None and len(samples):
                write_wav(wav_for_video(path), samples, self.mic.sample_rate)
            self.btn_record.setText("● Record")
            self.btn_record.setStyleSheet("")
            self.statusBar().showMessage(f"Saved {n} frames to {path}", 8000)
            if path and n and QtWidgets.QMessageBox.question(
                    self, "Recording saved", f"Saved {n} frames to\n{path}\n\nOpen it for frame-by-frame review?"
            ) == QtWidgets.QMessageBox.Yes:
                self.open_video(path)
            return
        if self.current_frame is None:
            return
        h, w = self.current_frame.shape[:2]
        fps = self.camera.actual_fps if self.camera is not None and self.camera.actual_fps > 0 else 30.0
        path = self.recorder.start(resolve("recordings"), (w, h), fps)
        if self.mic.active:
            self.mic.start_recording()  # saved as <take>.wav for scrubbing with sound later
        self.btn_record.setText("■ Stop recording")
        self.btn_record.setStyleSheet("color: white; background: #c62828;")
        self.statusBar().showMessage(f"Recording to {path}")

    # ------------------------------------------------------------ playback
    def toggle_play(self) -> None:
        if self.mode != Mode.VIDEO or self.chk_annot.isChecked():
            return
        self.playing = not self.playing
        self.btn_play.setText("⏸" if self.playing else "▶")
        self._next_play_time = time.time()

    def step(self, delta: int) -> None:
        if self.mode == Mode.VIDEO and self.video is not None:
            self.playing = False
            self.btn_play.setText("▶")
            self.seek(int(np.clip(self.frame_idx + delta, 0, len(self.video) - 1)))

    def _on_slider(self, value: int) -> None:
        if self.mode == Mode.VIDEO and value != self.frame_idx:
            self.seek(value)

    def seek(self, idx: int) -> None:
        if self.video is None:
            return
        self._autosave_annotation()
        frame = self.video.get(idx)
        if frame is None:
            self.playing = False
            self.btn_play.setText("▶")
            return
        if idx != self._last_shown_idx + 1:  # non-sequential jump: drop temporal state
            self.feedback.reset()
            self.geo_local.reset()
        self.frame_idx = idx
        self._update_sound_and_string()
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.frame_text.setText(f"{idx + 1} / {len(self.video)}  ({idx / self.video.fps:.2f}s)")
        self.current_frame = frame
        self._select_labelled_row()
        if self.chk_annot.isChecked():
            self._load_annotation_for_frame()
            self._render()
        elif self.chk_model.isChecked() and self.worker is not None:
            if idx in self.results:
                self.current_kp, self.current_geo = self.results[idx]
                self._render(new_measurement=True)
            else:
                self.current_kp = self.current_geo = None
                self._render()  # scrubbing: show the frame now, its points follow in ~30 ms
                self._submit(("video", idx), frame)
            self._feed_worker()
        else:
            stored = self.store.get(self._annot_file()) or self.model_store.get(self._annot_file())
            if stored:  # show saved labels (approved or model) and their geometry when reviewing
                self.current_kp = Keypoints2D.from_dict(stored)
                self.current_geo = self.geo_local(self.current_kp, (frame.shape[1], frame.shape[0]))
                self._render(new_measurement=True)
            else:
                self.current_kp = self.current_geo = None
                self._render(new_measurement=True)
        self._last_shown_idx = idx
        self._update_annot_status()

    # ---------------------------------------------------------------- tick
    def _tick(self) -> None:
        if self.mode == Mode.LIVE and self.camera is not None:
            fid = self.camera._frame_id
            if fid == self._last_cam_id:
                return
            self._last_cam_id = fid
            ok, frame = self.camera.read()
            if not ok:
                return
            self.recorder.write(frame)
            self._update_sound_and_string()
            if self.chk_model.isChecked() and self.worker is not None:
                self._submit(("live", fid), frame)  # shown when the result arrives
            else:
                self.current_frame, self.current_kp, self.current_geo = frame, None, None
                self._render(new_measurement=True)
        elif self.mode == Mode.DEMO:
            now = time.time()
            if now - getattr(self, "_last_demo", 0) < 1 / 30:
                return
            self._last_demo = now
            frame, scene = self.demo.frame(now - self._demo_t0)
            self.recorder.write(frame)
            self._demo_sound = SoundState(-20.0, True)
            self._update_sound_and_string()
            self.current_frame, self.current_kp = frame, scene.keypoints
            self.current_geo = self.geo_local(scene.keypoints, (frame.shape[1], frame.shape[0]))
            self._render(new_measurement=True)
        elif self.mode == Mode.VIDEO and self.playing and self.video is not None:
            if time.time() < self._next_play_time:
                return
            if self.frame_idx + 1 >= len(self.video):
                self.toggle_play()
                return
            if self._model_on_video() and self.frame_idx + 1 not in self.results:
                self._feed_worker()  # model hasn't reached the next frame yet: wait for it (no blank frames)
                return
            self._next_play_time = time.time() + 1 / self.video.fps
            self.seek(self.frame_idx + 1)

    # --------------------------------------------------------------- model
    def toggle_model(self, on: bool) -> None:
        if on and self.worker is None:
            if not self.checkpoint.exists():
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, f"No checkpoint at {self.checkpoint.name} - pick one", str(resolve("checkpoints")),
                    "PyTorch checkpoint (*.pt *.pth)")
                if not path:
                    QtWidgets.QMessageBox.information(
                        self, "No model yet",
                        "Train one first:\n  1. Record takes and label ~150+ frames (Annotate mode)\n"
                        "  2. python scripts/train.py\n\nUntil then, labelled frames and the synthetic demo "
                        "still exercise the full geometry pipeline.")
                    self.chk_model.setChecked(False)
                    return
                self.checkpoint = Path(path)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                from ..detector import KeypointDetector  # torch import is slow; defer

                detector = KeypointDetector(self.checkpoint)
            except Exception as e:
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.critical(self, "Model", f"Failed to load {self.checkpoint}:\n{e}")
                self.chk_model.setChecked(False)
                return
            QtWidgets.QApplication.restoreOverrideCursor()
            geo = GeometryEstimator(self.model, self.intrinsics, self.min_conf)
            self.worker = InferenceWorker(detector, geo)
            self.worker_thread = QtCore.QThread(self)
            self.worker.moveToThread(self.worker_thread)
            self.request_inference.connect(self.worker.process)
            self.worker.finished.connect(self._on_result)
            self.worker.failed.connect(self._on_worker_error)
            self.worker_thread.start()
            self.statusBar().showMessage(f"Loaded {self.checkpoint.name} on {detector.device}", 6000)
        self.results.clear()
        if self.mode == Mode.VIDEO:
            self.seek(self.frame_idx)

    def _submit(self, token, frame: np.ndarray) -> None:
        if self._busy:
            self._pending = (token, frame)  # keep only the newest request
            return
        self._busy = True
        self.request_inference.emit(token, frame)

    @QtCore.Slot(object, object, object, object, float)
    def _on_result(self, token, frame: np.ndarray, kp: Keypoints2D, geo: FrameGeometry, ms: float) -> None:
        self._busy = False
        self._infer_ms = ms
        kind, idx = token
        if kind == "video":
            self.results[idx] = (kp, geo)
            if self.mode == Mode.VIDEO and idx == self.frame_idx and not self.chk_annot.isChecked():
                self.current_kp, self.current_geo = kp, geo
                self._render(new_measurement=True)
        elif kind == "live" and self.mode == Mode.LIVE:
            self.current_frame, self.current_kp, self.current_geo = frame, kp, geo
            self._render(new_measurement=True)
        if self._pending is not None:
            token, frame = self._pending
            self._pending = None
            self._submit(token, frame)
        else:
            self._feed_worker()

    def _model_on_video(self) -> bool:
        return (self.mode == Mode.VIDEO and self.video is not None and self.worker is not None
                and self.chk_model.isChecked() and not self.chk_annot.isChecked())

    def _feed_worker(self) -> None:
        """Keep the model busy on the frames just ahead of the playhead, so playback runs at the take's
        own frame rate with every frame's points ready (and stepping / replaying is instant)."""
        if self._busy or not self._model_on_video():
            return
        for i in range(self.frame_idx, min(len(self.video), self.frame_idx + LOOKAHEAD)):
            if i not in self.results:
                frame = self.video.get(i)
                if frame is not None:
                    self._submit(("video", i), frame)
                return

    def _on_tolerance(self, value: float) -> None:
        self.feedback.tolerance = float(value)
        self.feedback.reset()
        if self.mode == Mode.VIDEO and not self.playing:
            self.seek(self.frame_idx)

    def _on_worker_error(self, msg: str) -> None:
        self._busy = False
        self.statusBar().showMessage(f"Inference error: {msg}", 10000)

    # ---------------------------------------------------------- annotation
    def _annot_file(self, idx: int | None = None) -> str:
        stem = self.video.path.stem if self.video is not None else "frame"
        return f"frames/{stem}_{self.frame_idx if idx is None else idx:06d}.jpg"

    def _frame_group(self, file: str | None = None) -> str | None:
        file = file or self._annot_file()
        if self.store.get(file) is not None:
            return "approved"
        if self.model_store.get(file) is not None:
            return "model"
        return None

    def toggle_annotate(self, on: bool) -> None:
        if on and self.mode != Mode.VIDEO:
            QtWidgets.QMessageBox.information(self, "Annotate", "Record a take (or load a video) first, then "
                                                                "label individual frames.")
            self.chk_annot.setChecked(False)
            return
        if on:
            self.playing = False
            self.btn_play.setText("▶")
            self.show_3d()  # check the pose as you click
            self._load_annotation_for_frame()
            self._render()
        else:
            self._autosave_annotation()
            self.seek(self.frame_idx)
        self._update_annot_status()

    def _load_annotation_for_frame(self) -> None:
        file = self._annot_file()
        approved, proposed = self.store.get(file), self.model_store.get(file)
        if approved is not None:
            self.annot, self.annot_source = dict(approved), "approved"
        elif proposed is not None:
            self.annot, self.annot_source = dict(proposed), "model"
        elif self.frame_idx in self.results:  # pre-fill from the model to speed up labelling
            self.annot, self.annot_source = self.results[self.frame_idx][0].to_dict(self.min_conf), "model"
        else:
            self.annot, self.annot_source = {}, "new"
        # Start each frame on the first point it still needs (scroll on a fresh frame).
        unplaced = [n for n in KEYPOINT_NAMES if n not in self.annot]
        self.kp_list.setCurrentRow(KEYPOINT_NAMES.index(unplaced[0]) if unplaced else 0)
        self._drag = None
        self.annot_dirty = False
        self._update_annot_status()

    def _on_image_click(self, x: float, y: float, button) -> None:
        if not self.chk_annot.isChecked() or self.current_frame is None:
            return
        h, w = self.current_frame.shape[:2]
        if button == QtCore.Qt.LeftButton:
            if not (0 <= x < w and 0 <= y < h):
                return
            name = KEYPOINT_NAMES[self.kp_list.currentRow()]
            # Pressing on a placed point grabs it. While the selected point still needs placing, only a
            # press right on another point grabs it, so close neighbours (bridge feet) can be placed.
            hit = self._hit_point(x, y, GRAB_SCREEN_PX if self.annot.get(name) is not None else GRAB_TIGHT_PX)
            if hit is not None:
                self._drag = hit  # selection stays on the next point to place
                return
            self.annot[name] = self._snapped(name, x, y)
            self._drag = name  # keep the button down to fine-tune it
            self._advance_selection()
        elif button == QtCore.Qt.RightButton:
            placed = [(n, p) for n, p in self.annot.items() if p is not None]
            if placed:
                n, p = min(placed, key=lambda np_: (np_[1][0] - x) ** 2 + (np_[1][1] - y) ** 2)
                if (p[0] - x) ** 2 + (p[1] - y) ** 2 < (30 * w / 1280) ** 2:
                    self.annot[n] = None
                    self.kp_list.setCurrentRow(KEYPOINT_NAMES.index(n))
        self.annot_dirty = True
        self._update_annot_status()
        self._render()

    def _snapped(self, name: str, x: float, y: float) -> list[float]:
        h, w = self.current_frame.shape[:2]
        x, y = float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))
        if name in ("bow_frog", "bow_tip"):  # bow leaving the frame: put the point exactly on the edge
            x = 0.0 if x < SNAP_PX else (w - 1.0 if x > w - 1 - SNAP_PX else x)
            y = 0.0 if y < SNAP_PX else (h - 1.0 if y > h - 1 - SNAP_PX else y)
        return [x, y]

    def _hit_point(self, x: float, y: float, screen_px: float = GRAB_SCREEN_PX) -> str | None:
        """Placed point under the cursor (within `screen_px` on screen), nearest first."""
        r = screen_px / max(self.video_label.scale, 1e-6)
        placed = [(n, p) for n, p in self.annot.items() if p is not None]
        if not placed:
            return None
        n, p = min(placed, key=lambda np_: (np_[1][0] - x) ** 2 + (np_[1][1] - y) ** 2)
        return n if (p[0] - x) ** 2 + (p[1] - y) ** 2 <= r * r else None

    def _on_image_drag(self, x: float, y: float) -> None:
        if self._drag is None or not self.chk_annot.isChecked() or self.current_frame is None:
            return
        self.annot[self._drag] = self._snapped(self._drag, x, y)
        self.annot_dirty = True
        now = time.time()
        if now - self._last_drag_render > 1 / 60:
            self._last_drag_render = now
            self._render()

    def _on_image_release(self) -> None:
        if self._drag is not None:
            self._drag = None
            self._update_annot_status()
            self._render()

    def _advance_selection(self) -> None:
        row = self.kp_list.currentRow()
        self.kp_list.setCurrentRow(min(row + 1, len(KEYPOINT_NAMES) - 1))

    def mark_not_visible(self) -> None:
        if self.chk_annot.isChecked():
            self.annot[KEYPOINT_NAMES[self.kp_list.currentRow()]] = None
            self.annot_dirty = True
            self._advance_selection()
            self._update_annot_status()
            self._render()

    def clear_annotation(self) -> None:
        self.annot = {}
        self.annot_dirty = True
        self.kp_list.setCurrentRow(0)
        self._update_annot_status()
        self._render()

    def save_annotation(self) -> None:
        """Save into the frame's own group: model labels stay 'to check' until approved."""
        if not self.chk_annot.isChecked() or self.current_frame is None:
            return
        file = self._annot_file()
        if not any(v is not None for v in self.annot.values()):
            self.store.remove(file)
            self.model_store.remove(file)
        elif self.annot_source == "model":
            self.model_store.put(file, self.current_frame, self.annot, **self._model_extra(file))
        else:
            self.store.put(file, self.current_frame, self.annot)
        self.annot_dirty = False
        self.statusBar().showMessage(f"Saved labels for {file}", 3000)
        self._update_annot_status()
        self._refresh_labelled_list()

    def _model_extra(self, file: str) -> dict:
        """Model confidences to keep with a model-labelled frame."""
        e = self.model_store.entry(file)
        if e is not None:
            return {k: v for k, v in e.items() if k in ("conf", "reproj_px")}
        if self.frame_idx in self.results:
            kp = self.results[self.frame_idx][0]
            return {"conf": {n: round(float(c), 3) for n, c in zip(KEYPOINT_NAMES, kp.conf)}}
        return {}

    def approve_frame(self) -> None:
        """Mark this frame's points as checked: move it to the approved (training) set, go to the next to check."""
        if self.mode != Mode.VIDEO or self.current_frame is None:
            return
        file = self._annot_file()
        if not self.chk_annot.isChecked():  # reviewing without annotate mode: approve the stored model labels
            stored = self.model_store.get(file) or self.store.get(file)
            if stored is None:
                self.statusBar().showMessage("No labels on this frame to approve", 3000)
                return
            self.annot = dict(stored)
        if not any(v is not None for v in self.annot.values()):
            self.statusBar().showMessage("No points on this frame to approve", 3000)
            return
        self.store.put(file, self.current_frame, self.annot)
        self.model_store.remove(file, delete_image=False)
        self.annot_source, self.annot_dirty = "approved", False
        left = len(self.model_store)
        self.statusBar().showMessage(f"Approved frame {self.frame_idx + 1} · {left} left to check", 4000)
        self._refresh_labelled_list()
        self._update_annot_status()
        nxt = self._next_frame(self.model_store, +1)
        if nxt is not None:
            self.lab_tabs.setCurrentWidget(self.check_list)
            self._goto_frame(*nxt)

    def auto_label(self) -> None:
        """Run the model over this take (every N frames) and file its labels under 'To check'."""
        if self.mode != Mode.VIDEO or self.video is None:
            QtWidgets.QMessageBox.information(self, "Label frames", "Load a take first ('Load video…').")
            return
        if self.worker is None:
            self.chk_model.setChecked(True)  # loads the checkpoint
            if self.worker is None:
                return
        self._autosave_annotation()
        self.playing = False
        self.btn_play.setText("▶")
        todo = [i for i in range(0, len(self.video), self.spin_every.value())
                if self._frame_group(self._annot_file(i)) is None]
        if not todo:
            self.statusBar().showMessage("Every selected frame of this take is already labelled", 4000)
            return
        dlg = QtWidgets.QProgressDialog(f"Labelling {len(todo)} frames with the model…", "Stop", 0, len(todo), self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)
        while self._busy:  # let the worker finish its current frame; we borrow its detector
            QtWidgets.QApplication.processEvents()
            time.sleep(0.005)
        det = self.worker.detector
        geo = GeometryEstimator(self.model, self.intrinsics, self.min_conf)
        done = 0
        for k, idx in enumerate(todo):
            dlg.setValue(k)
            QtWidgets.QApplication.processEvents()
            if dlg.wasCanceled():
                break
            frame = self.video.get(idx)
            if frame is None:
                continue
            kp = det(frame)
            g = geo(kp, (frame.shape[1], frame.shape[0]))
            geo.reset()  # frames are far apart: no temporal prior
            extra = {"conf": {n: round(float(c), 3) for n, c in zip(KEYPOINT_NAMES, kp.conf)}}
            if g.pose is not None:
                extra["reproj_px"] = round(float(g.pose.reproj_error_px), 2)
            self.model_store.put(self._annot_file(idx), frame, kp.to_dict(self.min_conf), flush=False, **extra)
            done += 1
        self.model_store.save()
        dlg.setValue(len(todo))
        self.statusBar().showMessage(f"Model labelled {done} frames: check them under 'To check', "
                                     "fix any point by dragging, then Approve (Enter)", 8000)
        self.lab_tabs.setCurrentWidget(self.check_list)
        self._refresh_labelled_list()
        first = next(((self.video.path.stem, i) for i in todo if self._frame_group(self._annot_file(i))), None)
        if first is not None:
            if not self.chk_annot.isChecked():
                self.chk_annot.setChecked(True)
            self._goto_frame(*first)

    def _autosave_annotation(self) -> None:
        if self.chk_annot.isChecked() and self.annot_dirty:
            self.save_annotation()

    def _update_annot_status(self) -> None:
        for i, n in enumerate(KEYPOINT_NAMES):
            item = self.kp_list.item(i)
            mark = "✓" if self.annot.get(n) is not None else ("✗" if n in self.annot else "·")
            item.setText(f"{KEYPOINT_KEYS[i]}. {mark} {n}")
        stored = self.store.get(self._annot_file()) if self.mode == Mode.VIDEO else None
        group = self._frame_group() if self.mode == Mode.VIDEO else None
        if self.annot_dirty:
            state = "unsaved changes" + (" (model labels, not approved)" if self.annot_source == "model" else "")
        elif group == "approved":
            state = "<span style='color:#2e7d32'>approved</span>"
        elif group == "model" or (self.chk_annot.isChecked() and self.annot_source == "model"):
            state = "<span style='color:#b26a00'>model labels, not approved yet: fix, then Approve</span>"
        else:
            state = "not labelled"
        if stored is not None and not self.annot_dirty and _needs_update(stored):
            todo = ", ".join(AnnotationStore.missing(stored))
            state += f" <span style='color:#c62828'>(still needs: {todo})</span>"
        if self.chk_annot.isChecked():
            unplaced = [n for n in KEYPOINT_NAMES if n not in self.annot]
            if unplaced and any(v is not None for v in self.annot.values()):
                state += (f"<br><small>{len(unplaced)} point(s) not labelled yet; saving marks them "
                          "not visible</small>")
        n_fix = sum(_needs_update(e["keypoints"]) for e in self.store.entries())
        fix = f"<br><span style='color:#c62828'>{n_fix} frames need the new points</span>" if n_fix else ""
        self.annot_status.setText(f"Approved: {len(self.store)} frames · to check: {len(self.model_store)}{fix}"
                                  f"<br>This frame: {state}")

    def show_3d(self) -> None:
        """Open the 3D window and draw the current frame's pose immediately."""
        self.view3d.show()
        self.view3d.raise_()
        self.view3d.update_geometry(getattr(self, "_last_geo", None), getattr(self, "_last_state", "none"))

    def _on_kp_selected(self, row: int) -> None:
        if 0 <= row < len(KEYPOINT_NAMES):
            self.guide.set_selected(KEYPOINT_NAMES[row])
            if self.chk_annot.isChecked():
                self._render()

    # ------------------------------------------------- labelled-frame navigator
    @staticmethod
    def _frames_in(store: AnnotationStore) -> list[tuple[str, int, dict]]:
        """(take stem, frame index, entry) for every frame in a store, sorted."""
        out = []
        for e in store.entries():
            stem, idx = Path(e["file"]).stem.rsplit("_", 1)
            out.append((stem, int(idx), e))
        return sorted(out, key=lambda t: (t[0], t[1]))

    def _refresh_labelled_list(self) -> None:
        current = self.video.path.stem if self.video is not None else None
        for lst, store in ((self.approved_list, self.store), (self.check_list, self.model_store)):
            lst.blockSignals(True)
            lst.clear()
            for stem, idx, e in self._frames_in(store):
                kps = e["keypoints"]
                placed = sum(kps.get(n) is not None for n in KEYPOINT_NAMES)
                where = f"frame {idx + 1}" if stem == current else f"{stem} · frame {idx + 1}"
                color = None
                if store is self.store:
                    todo = AnnotationStore.missing(kps)
                    mark = f"⚠ needs {len(todo)} pts" if todo else f"✓ {placed}/{len(KEYPOINT_NAMES)}"
                    color = "#c62828" if todo else None
                else:
                    conf = e.get("conf", {})
                    unsure = [n for n in KEYPOINT_NAMES if kps.get(n) is not None and conf.get(n, 1) < UNSURE_CONF]
                    rp = e.get("reproj_px")
                    mark = f"{placed}/{len(KEYPOINT_NAMES)}"
                    if unsure:
                        mark += f" · {len(unsure)} unsure"
                    if rp is None:
                        mark += " · no pose"
                    elif rp > 4:
                        mark += f" · fit {rp:.0f}px"
                    color = "#b26a00" if unsure or rp is None or rp > 4 else None
                item = QtWidgets.QListWidgetItem(f"{mark}  {where}")
                item.setData(QtCore.Qt.UserRole, (stem, idx))
                if color:
                    item.setForeground(QtGui.QColor(color))
                lst.addItem(item)
            lst.blockSignals(False)
        self.lab_tabs.setTabText(0, f"Approved ({len(self.store)})")
        self.lab_tabs.setTabText(1, f"To check ({len(self.model_store)})")
        self._select_labelled_row()

    def _select_labelled_row(self) -> None:
        key = (self.video.path.stem, self.frame_idx) if self.video is not None else None
        for lst in (self.approved_list, self.check_list):
            lst.blockSignals(True)
            lst.clearSelection()
            for i in range(lst.count()):
                if lst.item(i).data(QtCore.Qt.UserRole) == key:
                    lst.setCurrentRow(i)
                    break
            lst.blockSignals(False)

    def _on_labelled_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if item.listWidget() is self.check_list and not self.chk_annot.isChecked():
            self._goto_frame(*item.data(QtCore.Qt.UserRole))
            self.chk_annot.setChecked(True)  # checking means editing
            return
        self._goto_frame(*item.data(QtCore.Qt.UserRole))

    def _goto_frame(self, stem: str, idx: int) -> None:
        if self.video is None or self.video.path.stem != stem:
            path = next((p for p in (resolve("recordings") / f"{stem}.mp4", resolve("data/frames") / f"{stem}.mp4")
                         if p.exists()), None)
            if path is None:
                QtWidgets.QMessageBox.information(
                    self, "Video not found", f"Couldn't find {stem}.mp4 in recordings/. Load it with 'Load video…'.")
                return
            annotating = self.chk_annot.isChecked()
            self._autosave_annotation()
            self.chk_annot.blockSignals(True)
            self.chk_annot.setChecked(False)
            self.chk_annot.blockSignals(False)
            self.open_video(path)
            if annotating:
                self.chk_annot.setChecked(True)
        self.playing = False
        self.btn_play.setText("▶")
        self.seek(idx)

    def _next_frame(self, store: AnnotationStore, direction: int) -> tuple[str, int] | None:
        frames = [(s, i) for s, i, _ in self._frames_in(store)]
        if not frames:
            return None
        here = (self.video.path.stem, self.frame_idx) if self.video is not None else ("", -1)
        later = [f for f in frames if f > here] if direction > 0 else [f for f in frames if f < here]
        return (later[0] if direction > 0 else later[-1]) if later else (frames[0] if direction > 0 else frames[-1])

    def goto_labelled(self, direction: int) -> None:
        """[ / ]: previous / next frame in the group shown (Approved or To check)."""
        store = self.model_store if self.lab_tabs.currentWidget() is self.check_list else self.store
        target = self._next_frame(store, direction)
        if target is not None:
            self._goto_frame(*target)

    def delete_labels(self) -> None:
        group = self._frame_group() if self.mode == Mode.VIDEO else None
        if group is None:
            self.statusBar().showMessage("This frame has no saved labels", 3000)
            return
        where = "the approved dataset" if group == "approved" else "the frames to check"
        if QtWidgets.QMessageBox.question(
                self, "Delete labels", f"Remove frame {self.frame_idx + 1} from {where}?\n"
                                       "(Deletes its labels and saved image; the video is untouched.)"
        ) != QtWidgets.QMessageBox.Yes:
            return
        self.store.remove(self._annot_file())
        self.model_store.remove(self._annot_file())
        self.annot, self.annot_dirty = {}, False
        self.statusBar().showMessage(f"Deleted labels for frame {self.frame_idx + 1}", 4000)
        self._refresh_labelled_list()
        self.seek(self.frame_idx)
        self._update_annot_status()

    # -------------------------------------------------------------- render
    def _render(self, new_measurement: bool = False) -> None:
        if self.current_frame is None:
            return
        img = self.current_frame.copy()
        w, h = img.shape[1], img.shape[0]
        annotating = self.chk_annot.isChecked()
        if annotating:
            self._update_sound_and_string()
            kp = Keypoints2D.from_dict(self.annot)
            geo = self.geo_local(kp, (w, h))
            selected = KEYPOINT_NAMES[self.kp_list.currentRow()]
        else:
            kp, geo, selected = self.current_kp, self.current_geo, None

        cam = self.geo_local.intrinsics.scaled_to(w, h)
        if new_measurement or annotating:
            if annotating:
                self.feedback.reset()
                self.advisor.reset()
            fb = self.feedback.update(geo, self.current_sound)
            self._set_banner(fb)
            self._last_fb = fb
            self._last_adv = self.advisor.update(geo.pose if geo is not None else None, cam,
                                                 string_known=self.current_string[0] is not None)
            self._set_position(self._last_adv)
        fb = getattr(self, "_last_fb", None)
        color = OVERLAY_BGR.get(fb.state, (60, 60, 230)) if fb else (200, 200, 200)

        if geo is not None and kp is not None and self.chk_fit.isChecked():
            draw_model_fit(img, geo, cam, self.model, kp, 0.5 if annotating else self.min_conf)
        if geo is not None:
            draw_geometry(img, geo, cam, self.model, color)
        if kp is not None:
            draw_keypoints(img, kp, self.min_conf if not annotating else 0.5, selected=selected)
        if self.recorder.is_recording:
            cv2.circle(img, (w - 30, 30), 12, (0, 0, 255), -1, cv2.LINE_AA)
        self.video_label.show_bgr(img)

        self._last_geo = geo
        self._last_state = ("none" if fb is None or fb.state == BowState.NO_BOW
                            else ("ok" if fb.state == BowState.OK else "bad"))
        if self.view3d.isVisible():
            self.view3d.update_geometry(geo, self._last_state)
        self._update_info(geo, w, h)

    def _set_position(self, adv: PositionAdvice) -> None:
        bg, fg = POSITION_STYLE[adv.status]
        self.pos_banner.setStyleSheet(f"background:{bg}; color:{fg}; font-size:15px; font-weight:600; "
                                      "border-radius:6px;")
        self.pos_banner.setText(("✓ " if adv.status == "ok" else "↻ " if adv.status == "adjust" else "") + adv.text)

    # ------------------------------------------------------- sound / string
    def _start_mic(self) -> None:
        if not self.mic.active and self.mic.start():
            self.statusBar().showMessage(f"Microphone: {self.mic.device_name}", 5000)

    def _on_mic_failed(self, msg: str) -> None:
        self.statusBar().showMessage(f"Microphone: {msg}", 12000)

    def _on_sound_toggled(self, on: bool) -> None:
        if self.mode == Mode.LIVE:
            self._start_mic() if on else self.mic.stop()
        self._on_string_choice()

    def _on_string_choice(self, *_):
        self.results.clear()  # cached model results were solved with the old string
        self.feedback.reset()
        if self.mode == Mode.VIDEO:
            self.seek(self.frame_idx)

    def _update_sound_and_string(self) -> None:
        """Sound state for the frame being processed, and the string the bow solver should assume."""
        sound = None
        if self.chk_sound.isChecked():
            if self.mode == Mode.LIVE and self.mic.active:
                sound = self.mic.state
            elif self.mode == Mode.VIDEO and self.audio_track is not None and self.video is not None:
                sound = self.audio_track.state_at(self.frame_idx / self.video.fps)
            elif self.mode == Mode.DEMO:
                sound = getattr(self, "_demo_sound", None)
        choice = self.combo_string.currentIndex()
        if choice > 0:
            string, src = STRING_CHOICES[choice], "chosen"
        else:
            string, src = None, "any: middle bowing plane"
        self.current_sound, self.current_string = sound, (string, src)
        self.geo_local.string = string
        if self.worker is not None:
            self.worker.geometry.string = string

    def _set_banner(self, fb: Feedback) -> None:
        bg, fg = BANNER_STYLE[fb.state]
        self.banner.setStyleSheet(f"background:{bg}; color:{fg}; font-size:24px; font-weight:600; "
                                  "border-radius:6px;")
        self.banner.setText(fb.text)

    def _update_info(self, geo: FrameGeometry | None, w: int, h: int) -> None:
        now = time.time()
        self._fps_times = [t for t in self._fps_times if now - t < 1.0] + [now]
        lines = [f"source     {self.mode.name.lower()}  {w}x{h}",
                 f"display    {len(self._fps_times):.0f} fps"]
        if self.chk_model.isChecked():
            lines.append(f"inference  {self._infer_ms:.1f} ms")
        lines.append(f"camera     {self.calib_note}")
        snd = self.current_sound
        if not self.chk_sound.isChecked():
            lines.append("sound      off (every bow counts as playing)")
        elif snd is None:
            lines.append("sound      " + ("no .wav for this take" if self.mode == Mode.VIDEO else "no microphone data"))
        elif snd.sounding:
            lines.append(f"sound      playing  {snd.level_db:.0f} dB")
        else:
            lines.append(f"sound      silent  {snd.level_db:.0f} dB")
        string, src = self.current_string
        lines.append(f"string     {string or '?'}  ({src})")
        adv = getattr(self, "_last_adv", None)
        if adv is not None and adv.string_error_deg is not None:
            fix = ""
            if adv.raise_deg is not None:
                fix = f", raise cam {adv.raise_deg:.0f}°"
            if abs(adv.turn_deg) >= 4:
                fix += f", turn {abs(adv.turn_deg):.0f}° {'L' if adv.turn_deg > 0 else 'R'}"
            lines.append(f"position   ±{adv.string_error_deg:.1f}° across strings{fix}")
        lines.append("")
        if geo is None or geo.pose is None:
            lines.append(geo.message if geo is not None and geo.message else "no violin pose")
        else:
            p = geo.pose
            lines += [f"violin     {np.linalg.norm(p.t) / 10:.1f} cm away",
                      f"pnp        {p.num_points} pts, {p.reproj_error_px:.1f} px rms"]
            if p.dropped:
                lines.append(f"outlier    ignored {p.dropped}")
            va = p.view_angle_to_strings_deg
            hint = "good" if va >= 75 else "turn so scroll & tailpiece are equally far from camera"
            lines.append(f"view       {va:.0f}° to strings ({hint})")
            if p.residuals:
                worst = max(p.residuals, key=p.residuals.get)
                lines.append(f"worst fit  {worst} {p.residuals[worst]:.1f} px")
            if geo.bow is not None:
                b = geo.bow
                lines += ["",
                          f"bow angle  {b.angle_deg:6.1f}°  (3D)",
                          f"image ang  {b.angle_2d_deg:6.1f}°  (2D naive)",
                          f"tilt       {b.tilt_deg:+6.1f}°",
                          f"contact    {b.contact_from_bridge_mm:6.1f} mm from bridge",
                          f"solver     {b.method}",
                          f"G..E range {b.string_spread_deg:6.1f}°  (angle if each string)"]
            else:
                lines.append(geo.message or "no bow")
        tol = self.feedback.tolerance
        lines += ["", f"target     {self.feedback.target:.0f}° ± {tol:.0f}°"]
        self.info.setText("<pre>" + "\n".join(lines) + "</pre>")

    # ------------------------------------------------------------- cleanup
    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        self._autosave_annotation()
        self._stop_sources()
        self.mic.stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        self.view3d.close()
        super().closeEvent(ev)
