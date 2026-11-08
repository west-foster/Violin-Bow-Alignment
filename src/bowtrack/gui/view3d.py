"""Task 4: second window showing the 3D keypoints, violin, bow and camera."""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

from ..geometry import FrameGeometry, ViolinModel

STATE_RGBA = {
    "ok": (0.2, 0.9, 0.3, 1.0),
    "bad": (1.0, 0.25, 0.2, 1.0),
    "none": (0.6, 0.6, 0.6, 1.0),
}


def cam_to_display(p: np.ndarray) -> np.ndarray:
    """OpenCV camera frame (x right, y down, z forward) -> z-up display frame."""
    p = np.atleast_2d(p)
    return np.c_[p[:, 0], p[:, 2], -p[:, 1]]


def _segments(polylines: list[np.ndarray]) -> np.ndarray:
    segs = [np.repeat(pl, 2, axis=0)[1:-1] for pl in polylines]
    return np.vstack(segs) if segs else np.zeros((0, 3))


class View3DWindow(QtWidgets.QWidget):
    def __init__(self, model: ViolinModel, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("bowtrack - 3D view")
        self.resize(720, 560)
        self.model = model
        self.wire_violin = model.wireframe()
        self.model_kp = np.array([model.points[n] for n in model.points])

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((22, 22, 26))
        self.reset_view()

        grid = gl.GLGridItem()
        grid.setSize(2000, 2000)
        grid.setSpacing(100, 100)
        grid.translate(0, 900, -400)
        self.view.addItem(grid)

        # Camera frustum at the origin.
        d, w, h = 150, 110, 65
        corners = np.array([[-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]], float)
        fr = [np.array([[0, 0, 0], c]) for c in corners] + [np.vstack([corners, corners[:1]])]
        self.view.addItem(gl.GLLinePlotItem(pos=cam_to_display(_segments(fr)), mode="lines",
                                            color=(0.5, 0.7, 1.0, 1.0), width=1.5, antialias=True))

        self.violin_item = gl.GLLinePlotItem(mode="lines", color=(0.85, 0.6, 0.3, 1.0), width=2, antialias=True)
        self.bow_item = gl.GLLinePlotItem(mode="lines", color=STATE_RGBA["none"], width=4, antialias=True)
        self.string_axis_item = gl.GLLinePlotItem(mode="lines", color=(1, 1, 1, 0.8), width=1.5, antialias=True)
        self.kp_item = gl.GLScatterPlotItem(size=12, color=(1.0, 0.85, 0.2, 1.0), pxMode=True)
        self.contact_item = gl.GLScatterPlotItem(size=16, color=(0.1, 1.0, 1.0, 1.0), pxMode=True)
        # Camera rays through the frog / tip pixels: the bow ends must lie on these.
        self.ray_item = gl.GLLinePlotItem(mode="lines", color=(0.6, 0.7, 1.0, 0.35), width=1, antialias=True)
        for it in (self.violin_item, self.string_axis_item, self.bow_item, self.kp_item, self.contact_item,
                   self.ray_item):
            self.view.addItem(it)
        self.labels: dict[str, gl.GLTextItem] = {}
        for name in list(model.points) + ["bow_frog", "bow_tip", "contact"]:
            item = gl.GLTextItem(text=name, color=(220, 220, 220, 255))
            item.setVisible(False)
            self.view.addItem(item)
            self.labels[name] = item
        self._needs_center = True

        self.info = QtWidgets.QLabel("No pose yet")
        self.info.setStyleSheet("color:#ddd; background:#16161a; padding:6px; font-family: Menlo, monospace;")
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.info, 1)
        bar.addWidget(reset)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)
        self._clear()

    def reset_view(self) -> None:
        self.view.setCameraPosition(pos=QtGui.QVector3D(0, 900, 0), distance=1900, elevation=18, azimuth=-60)
        self._needs_center = True  # re-centre on the violin at the next pose

    def _clear(self) -> None:
        empty = np.zeros((2, 3))
        for it in (self.violin_item, self.bow_item, self.string_axis_item):
            it.setData(pos=empty)
        self.kp_item.setData(pos=np.zeros((0, 3)))
        self.contact_item.setData(pos=np.zeros((0, 3)))
        self.ray_item.setData(pos=np.zeros((2, 3)))
        for item in self.labels.values():
            item.setVisible(False)

    def _label(self, name: str, p_cam: np.ndarray) -> None:
        item = self.labels[name]
        item.setData(pos=cam_to_display(p_cam)[0] + np.array([8.0, 0.0, 8.0]))
        item.setVisible(True)

    def update_geometry(self, geo: FrameGeometry | None, state: str = "none") -> None:
        if geo is None or geo.pose is None:
            self._clear()
            self.info.setText(geo.message if geo is not None and geo.message else "No pose")
            return
        pose = geo.pose
        if self._needs_center:  # orbit around the violin rather than the camera
            c = cam_to_display(pose.t)[0]
            self.view.setCameraPosition(pos=QtGui.QVector3D(*c), distance=1300, elevation=18, azimuth=-60)
            self._needs_center = False
        for item in self.labels.values():
            item.setVisible(False)
        for name, p3 in self.model.points.items():
            self._label(name, pose.to_camera(p3))
        wire = [pose.to_camera(pl) for pl in self.wire_violin]
        self.violin_item.setData(pos=cam_to_display(_segments(wire)))
        axis = pose.to_camera(np.array([[-60, 0, 0], [self.model.string_length + 60, 0, 0]], float))
        self.string_axis_item.setData(pos=cam_to_display(axis))
        kps = [pose.to_camera(self.model_kp)]
        text = f"violin @ {np.linalg.norm(pose.t) / 10:.0f} cm   reproj {pose.reproj_error_px:.1f}px"
        if geo.bow is not None:
            b = geo.bow
            kps.append(np.vstack([b.frog, b.tip]))
            self.bow_item.setData(pos=cam_to_display(np.vstack([b.frog, b.tip])), color=STATE_RGBA[state])
            self.contact_item.setData(pos=cam_to_display(b.contact))
            self.ray_item.setData(pos=cam_to_display(np.array([[0, 0, 0], b.frog * 1.15, [0, 0, 0], b.tip * 1.15])))
            self._label("bow_frog", b.frog)
            self._label("bow_tip", b.tip)
            self._label("contact", b.contact)
            text += (f"   angle {b.angle_deg:.1f}°   tilt {b.tilt_deg:+.1f}°   "
                     f"contact {b.contact_from_bridge_mm:.0f} mm from bridge   [{b.method}]")
        else:
            self.bow_item.setData(pos=np.zeros((2, 3)))
            self.contact_item.setData(pos=np.zeros((0, 3)))
            self.ray_item.setData(pos=np.zeros((2, 3)))
        self.kp_item.setData(pos=cam_to_display(np.vstack(kps)))
        self.info.setText(text)
