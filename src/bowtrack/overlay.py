"""Draw keypoints and geometry over a video frame (OpenCV, BGR)."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import CameraIntrinsics, FrameGeometry, ViolinModel
from .keypoints import KEYPOINT_COLORS, KEYPOINT_NAMES, SKELETON, Keypoints2D


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_keypoints(img: np.ndarray, kp: Keypoints2D, min_conf: float, labels: bool = True,
                   selected: str | None = None) -> None:
    s = max(1.0, img.shape[1] / 1280)
    for a, b in SKELETON:
        if kp.visible(a, min_conf) and kp.visible(b, min_conf):
            cv2.line(img, _pt(kp.get(a)), _pt(kp.get(b)), (230, 230, 230), max(1, int(2 * s)), cv2.LINE_AA)
    for i, name in enumerate(KEYPOINT_NAMES):
        if kp.conf[i] < min_conf:
            continue
        p = _pt(kp.xy[i])
        r = int(7 * s)
        cv2.circle(img, p, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, p, r, KEYPOINT_COLORS[name], -1, cv2.LINE_AA)
        if name == selected:
            cv2.circle(img, p, r + 7, (255, 255, 255), 2, cv2.LINE_AA)
        if labels:
            text = name if kp.conf[i] >= 0.999 else f"{name} {kp.conf[i]:.2f}"
            org = (p[0] + r + 4, p[1] - r)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 0), int(4 * s), cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, KEYPOINT_COLORS[name], max(1, int(s)),
                        cv2.LINE_AA)


def draw_geometry(img: np.ndarray, geo: FrameGeometry, cam: CameraIntrinsics, model: ViolinModel,
                  color: tuple[int, int, int]) -> None:
    if geo.pose is None:
        return
    s = max(1.0, img.shape[1] / 1280)
    pose = geo.pose
    # Violin coordinate axes at the bridge (x=strings red, y=bass green, z=normal blue).
    axes = pose.to_camera(np.array([[0, 0, 0], [80, 0, 0], [0, 80, 0], [0, 0, 80]], float))
    o, ax, ay, az = cam.project(axes)
    for end, c in ((ax, (0, 0, 255)), (ay, (0, 255, 0)), (az, (255, 0, 0))):
        cv2.arrowedLine(img, _pt(o), _pt(end), c, max(1, int(2 * s)), cv2.LINE_AA, tipLength=0.15)
    # String axis (bridge -> nut) as a dashed line.
    a, b = cam.project(pose.to_camera(np.array([[0, 0, 0], [model.string_length, 0, 0]], float)))
    for k in range(0, 20, 2):
        p0, p1 = a + (b - a) * k / 20, a + (b - a) * (k + 1) / 20
        cv2.line(img, _pt(p0), _pt(p1), (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
    if geo.bow is not None:
        f, t, c = cam.project(np.vstack([geo.bow.frog, geo.bow.tip, geo.bow.contact]))
        cv2.line(img, _pt(f), _pt(t), color, max(2, int(3 * s)), cv2.LINE_AA)
        cv2.drawMarker(img, _pt(c), (0, 255, 255), cv2.MARKER_TILTED_CROSS, int(22 * s), max(2, int(2 * s)))
        label = f"{geo.bow.angle_deg:.1f} deg"
        org = (int(c[0]) + int(14 * s), int(c[1]) + int(30 * s))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 0, 0), int(5 * s), cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, color, max(1, int(2 * s)), cv2.LINE_AA)
