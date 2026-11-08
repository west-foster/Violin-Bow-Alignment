"""Labelling aid: a diagram showing where the selected keypoint goes."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..geometry import ViolinModel
from ..keypoints import KEYPOINT_COLORS, KEYPOINT_HELP, KEYPOINT_NAMES

W, H_VIOLIN, H_BOW = 300, 130, 46
X_MIN, X_MAX, Y_HALF = -175.0, 445.0, 112.0  # violin-frame extent shown (mm)


def _top_view(p: np.ndarray) -> np.ndarray:
    """Violin frame -> diagram pixels, looking down at the top plate.

    Scroll points right (as in a typical webcam view); bass (G) side is up.
    """
    s = W / (X_MAX - X_MIN)
    p = np.atleast_2d(p)
    return np.c_[(p[:, 0] - X_MIN) * s, (Y_HALF - p[:, 1]) * s + 4]


def render_guide(model: ViolinModel, selected: str) -> np.ndarray:
    img = np.full((H_VIOLIN + H_BOW, W, 3), 245, np.uint8)
    parts = model.parts()
    style = {"body": (60, 110, 175), "tailpiece": (40, 40, 40), "fingerboard": (40, 40, 40),
             "pegbox": (60, 110, 175)}
    for name, col in style.items():
        cv2.fillPoly(img, [_top_view(parts[name]).astype(np.int32)], col, cv2.LINE_AA)
    for name, line in parts.items():
        if name.startswith("string_"):
            cv2.polylines(img, [_top_view(line).astype(np.int32)], False, (215, 215, 215), 1, cv2.LINE_AA)
    cv2.polylines(img, [_top_view(parts["bridge"][[0, -1]]).astype(np.int32)], False, (170, 215, 240), 3)
    cv2.putText(img, "G side", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "E side", (4, H_VIOLIN - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)

    # Bow sketch below the violin: frog on the left, tip on the right.
    y0 = H_VIOLIN + H_BOW // 2
    frog_x, tip_x = 40, W - 14
    cv2.line(img, (16, y0 - 6), (W - 8, y0 - 8), (40, 60, 90), 3, cv2.LINE_AA)  # stick
    cv2.line(img, (frog_x, y0 + 6), (tip_x, y0 - 4), (235, 235, 235), 2, cv2.LINE_AA)  # hair
    cv2.rectangle(img, (16, y0 - 6), (frog_x, y0 + 10), (30, 30, 30), -1)  # frog
    cv2.putText(img, "bow", (W // 2 - 10, H_VIOLIN + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1,
                cv2.LINE_AA)
    points = {n: _top_view(model.points[n])[0] for n in model.points}
    points["bow_frog"] = np.array([frog_x, y0 + 6], float)
    points["bow_tip"] = np.array([tip_x, y0 - 4], float)

    for name in KEYPOINT_NAMES:
        p = tuple(int(v) for v in points[name])
        bgr = KEYPOINT_COLORS[name]
        if name == selected:
            cv2.circle(img, p, 11, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(img, p, 7, bgr, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 4, bgr, -1, cv2.LINE_AA)
            cv2.circle(img, p, 4, (60, 60, 60), 1, cv2.LINE_AA)
    return img


class KeypointGuide(QtWidgets.QWidget):
    def __init__(self, model: ViolinModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.text = QtWidgets.QLabel()
        self.text.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.image)
        lay.addWidget(self.text)
        self.set_selected(KEYPOINT_NAMES[0])

    def set_selected(self, name: str) -> None:
        rgb = cv2.cvtColor(render_guide(self.model, name), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        self.image.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))
        self.text.setText(f"<b>{name}</b>: {KEYPOINT_HELP[name]}")
