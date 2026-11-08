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

from ..annotations import AnnotationStore
from ..camera import CameraStream
from ..config import resolve
from ..feedback import BowFeedback, BowState, Feedback
from ..geometry import CameraIntrinsics, FrameGeometry, GeometryEstimator, ViolinModel
from ..keypoints import KEYPOINT_NAMES, Keypoints2D
from ..overlay import draw_geometry, draw_keypoints
from ..synthetic import DemoSource
from ..video import VideoFile, VideoRecorder
from .view3d import View3DWindow

BANNER_STYLE = {
    BowState.OK: ("#1f9d3a", "#ffffff"),
    BowState.TIP_TOWARD_SCROLL: ("#c62828", "#ffffff"),
    BowState.TIP_TOWARD_BRIDGE: ("#c62828", "#ffffff"),
    BowState.NO_BOW: ("#000000", "#bbbbbb"),
}
OVERLAY_BGR = {BowState.OK: (60, 210, 60), BowState.NO_BOW: (200, 200, 200)}


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

    clicked = QtCore.Signal(float, float, object)  # x, y, Qt.MouseButton

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

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        p = ev.position()
        self.clicked.emit((p.x() - self._off.x()) / self._scale, (p.y() - self._off.y()) / self._scale, ev.button())


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
        self.store = AnnotationStore(resolve("data"))
        self.annot: dict[str, list[float] | None] = {}
        self.annot_dirty = False

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
        self.btn_3d = QtWidgets.QPushButton("Show 3D view")
        self.btn_3d.clicked.connect(lambda: (self.view3d.show(), self.view3d.raise_()))
        top = QtWidgets.QHBoxLayout()
        for w in (self.btn_camera, self.btn_record, self.btn_open, self.btn_demo):
            top.addWidget(w)
        top.addSpacing(20)
        top.addWidget(self.chk_model)
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

        # Right panel: annotation + info.
        self.chk_annot = QtWidgets.QCheckBox("Annotate mode (A)")
        self.chk_annot.toggled.connect(self.toggle_annotate)
        self.kp_list = QtWidgets.QListWidget()
        for n in KEYPOINT_NAMES:
            self.kp_list.addItem(n)
        self.kp_list.setCurrentRow(0)
        self.kp_list.setMaximumHeight(170)
        self.btn_save_annot = QtWidgets.QPushButton("Save frame to dataset (S)")
        self.btn_save_annot.clicked.connect(self.save_annotation)
        self.btn_clear_annot = QtWidgets.QPushButton("Clear points")
        self.btn_clear_annot.clicked.connect(self.clear_annotation)
        self.annot_status = QtWidgets.QLabel()
        self.annot_status.setWordWrap(True)
        help_text = QtWidgets.QLabel(
            "<small>Left-click: place selected point (auto-advances)<br>"
            "Right-click: remove nearest point<br>"
            "N: mark selected as not visible · 1-7: select point<br>"
            "Space: play/pause · ←/→: step · Shift+←/→: ±10</small>"
        )
        annot_box = QtWidgets.QGroupBox("Labelling")
        al = QtWidgets.QVBoxLayout(annot_box)
        for w in (self.chk_annot, self.kp_list, self.btn_save_annot, self.btn_clear_annot, self.annot_status,
                  help_text):
            al.addWidget(w)

        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("font-family: Menlo, monospace; font-size: 12px;")
        self.info.setAlignment(QtCore.Qt.AlignTop)
        self.info.setWordWrap(True)
        info_box = QtWidgets.QGroupBox("Measurements")
        QtWidgets.QVBoxLayout(info_box).addWidget(self.info)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(annot_box)
        right.addWidget(info_box, 1)
        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)
        right_w.setFixedWidth(320)

        root = QtWidgets.QHBoxLayout()
        root.addLayout(left, 1)
        root.addWidget(right_w)
        central = QtWidgets.QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._update_annot_status()

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
        sc("N", self.mark_not_visible)
        sc("R", lambda: self.btn_record.isEnabled() and self.toggle_record())
        for i in range(len(KEYPOINT_NAMES)):
            sc(str(i + 1), lambda i=i: self.kp_list.setCurrentRow(i))

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
        self.btn_camera.setText("Start camera")
        self.playing = False
        self.btn_play.setText("▶")
        self.mode = Mode.IDLE
        self.results.clear()
        self.feedback.reset()
        self.geo_local.reset()
        if self.worker is not None:
            self.worker.geometry.reset()

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
        self.view3d.show()

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
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self.video) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._set_playback_enabled(True)
        self.setWindowTitle(f"bowtrack - {Path(path).name}")
        self.seek(0)

    def toggle_record(self) -> None:
        if self.recorder.is_recording:
            path = self.recorder.stop()
            n = self.recorder.frames_written
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
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.frame_text.setText(f"{idx + 1} / {len(self.video)}  ({idx / self.video.fps:.2f}s)")
        self.current_frame = frame
        if self.chk_annot.isChecked():
            self._load_annotation_for_frame()
            self._render()
        elif self.chk_model.isChecked() and self.worker is not None:
            if idx in self.results:
                self.current_kp, self.current_geo = self.results[idx]
                self._render(new_measurement=True)
            else:
                self.current_kp = self.current_geo = None
                self._render()
                self._submit(("video", idx), frame)
        else:
            stored = self.store.get(self._annot_file())
            if stored:  # show saved labels and their geometry when reviewing
                self.current_kp = Keypoints2D.from_dict(stored)
                self.current_geo = self.geo_local(self.current_kp, (frame.shape[1], frame.shape[0]))
                self._render(new_measurement=True)
            else:
                self.current_kp = self.current_geo = None
                self._render(new_measurement=True)
        self._last_shown_idx = idx

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
            self.current_frame, self.current_kp = frame, scene.keypoints
            self.current_geo = self.geo_local(scene.keypoints, (frame.shape[1], frame.shape[0]))
            self._render(new_measurement=True)
        elif self.mode == Mode.VIDEO and self.playing and self.video is not None:
            if self._busy or time.time() < self._next_play_time:
                return  # with the model on, playback advances at inference speed
            self._next_play_time = time.time() + 1 / self.video.fps
            if self.frame_idx + 1 >= len(self.video):
                self.toggle_play()
                return
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

    def _on_worker_error(self, msg: str) -> None:
        self._busy = False
        self.statusBar().showMessage(f"Inference error: {msg}", 10000)

    # ---------------------------------------------------------- annotation
    def _annot_file(self) -> str:
        stem = self.video.path.stem if self.video is not None else "frame"
        return f"frames/{stem}_{self.frame_idx:06d}.jpg"

    def toggle_annotate(self, on: bool) -> None:
        if on and self.mode != Mode.VIDEO:
            QtWidgets.QMessageBox.information(self, "Annotate", "Record a take (or load a video) first, then "
                                                                "label individual frames.")
            self.chk_annot.setChecked(False)
            return
        if on:
            self.playing = False
            self.btn_play.setText("▶")
            self._load_annotation_for_frame()
            self._render()
        else:
            self._autosave_annotation()
            self.seek(self.frame_idx)
        self._update_annot_status()

    def _load_annotation_for_frame(self) -> None:
        stored = self.store.get(self._annot_file())
        if stored is not None:
            self.annot = dict(stored)
        elif self.frame_idx in self.results:  # pre-fill from the model to speed up labelling
            self.annot = self.results[self.frame_idx][0].to_dict(self.min_conf)
        else:
            self.annot = {}
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
            self.annot[name] = [float(x), float(y)]
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
        if not self.chk_annot.isChecked() or self.current_frame is None:
            return
        if not any(v is not None for v in self.annot.values()):
            self.store.remove(self._annot_file())
        else:
            self.store.put(self._annot_file(), self.current_frame, self.annot)
        self.annot_dirty = False
        self.statusBar().showMessage(f"Saved labels for {self._annot_file()}", 3000)
        self._update_annot_status()

    def _autosave_annotation(self) -> None:
        if self.chk_annot.isChecked() and self.annot_dirty:
            self.save_annotation()

    def _update_annot_status(self) -> None:
        for i, n in enumerate(KEYPOINT_NAMES):
            item = self.kp_list.item(i)
            mark = "✓" if self.annot.get(n) is not None else "·"
            item.setText(f"{i + 1}. {mark} {n}")
        saved = self.mode == Mode.VIDEO and self.store.get(self._annot_file()) is not None
        state = "unsaved changes" if self.annot_dirty else ("saved" if saved else "not labelled")
        self.annot_status.setText(f"Dataset: {len(self.store)} labelled frames<br>This frame: {state}")

    # -------------------------------------------------------------- render
    def _render(self, new_measurement: bool = False) -> None:
        if self.current_frame is None:
            return
        img = self.current_frame.copy()
        w, h = img.shape[1], img.shape[0]
        annotating = self.chk_annot.isChecked()
        if annotating:
            kp = Keypoints2D.from_dict(self.annot)
            geo = self.geo_local(kp, (w, h))
            selected = KEYPOINT_NAMES[self.kp_list.currentRow()]
        else:
            kp, geo, selected = self.current_kp, self.current_geo, None

        if new_measurement or annotating:
            if annotating:
                self.feedback.reset()
            fb = self.feedback.update(geo)
            self._set_banner(fb)
            self._last_fb = fb
        fb = getattr(self, "_last_fb", None)
        color = OVERLAY_BGR.get(fb.state, (60, 60, 230)) if fb else (200, 200, 200)

        cam = self.geo_local.intrinsics.scaled_to(w, h)
        if geo is not None:
            draw_geometry(img, geo, cam, self.model, color)
        if kp is not None:
            draw_keypoints(img, kp, self.min_conf if not annotating else 0.5, selected=selected)
        if self.recorder.is_recording:
            cv2.circle(img, (w - 30, 30), 12, (0, 0, 255), -1, cv2.LINE_AA)
        self.video_label.show_bgr(img)

        if self.view3d.isVisible():
            self.view3d.update_geometry(geo, "none" if fb is None or fb.state == BowState.NO_BOW
                                        else ("ok" if fb.state == BowState.OK else "bad"))
        self._update_info(geo, w, h)

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
        lines.append("")
        if geo is None or geo.pose is None:
            lines.append(geo.message if geo is not None and geo.message else "no violin pose")
        else:
            p = geo.pose
            lines += [f"violin     {np.linalg.norm(p.t) / 10:.1f} cm away",
                      f"pnp        {p.num_points} pts, {p.reproj_error_px:.1f} px rms"]
            if geo.bow is not None:
                b = geo.bow
                lines += ["",
                          f"bow angle  {b.angle_deg:6.1f}°  (3D)",
                          f"image ang  {b.angle_2d_deg:6.1f}°  (2D naive)",
                          f"tilt       {b.tilt_deg:+6.1f}°",
                          f"contact    {b.contact_from_bridge_mm:6.1f} mm from bridge",
                          f"solver     {b.method}  (len resid {b.length_residual_mm:+.0f} mm)"]
            else:
                lines.append(geo.message or "no bow")
        tol = self.feedback.tolerance
        lines += ["", f"target     {self.feedback.target:.0f}° ± {tol:.0f}°"]
        self.info.setText("<pre>" + "\n".join(lines) + "</pre>")

    # ------------------------------------------------------------- cleanup
    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        self._autosave_annotation()
        self._stop_sources()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        self.view3d.close()
        super().closeEvent(ev)
