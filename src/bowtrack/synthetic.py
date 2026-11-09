"""Synthetic scenes with known ground truth (for tests and GUI demo mode).

Places the violin model at a pose in front of the camera, puts a bow across
the strings at a chosen angle / tilt / sounding point, and projects all
keypoints to pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import CameraIntrinsics, ViolinModel
from .keypoints import KEYPOINT_NAMES, VIOLIN_KEYPOINTS, Keypoints2D


@dataclass
class SyntheticScene:
    keypoints: Keypoints2D
    R: np.ndarray
    t: np.ndarray
    angle_deg: float
    tilt_deg: float
    contact_mm: float
    frog: np.ndarray
    tip: np.ndarray
    string: str | None = None  # string being played (ground truth; the solver is not told)


def bow_in_violin_frame(angle_deg: float, tilt_deg: float, contact_mm: float, length: float,
                        frog_to_contact: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    a, p = np.radians(angle_deg), np.radians(tilt_deg)
    d = np.array([np.cos(a), np.sin(a) * np.cos(p), np.sin(a) * np.sin(p)])
    c = np.array([contact_mm, 0.0, 0.0])
    return c - frog_to_contact * d, c + (length - frog_to_contact) * d


def make_scene(model: ViolinModel, cam: CameraIntrinsics, rvec: np.ndarray, tvec: np.ndarray,
               angle_deg: float = 90.0, tilt_deg: float = 0.0, contact_mm: float = 30.0,
               frog_to_contact: float = 250.0, noise_px: float = 0.0,
               rng: np.random.Generator | None = None) -> SyntheticScene:
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    t = np.asarray(tvec, np.float64).ravel()
    frog, tip = bow_in_violin_frame(angle_deg, tilt_deg, contact_mm, model.bow_length, frog_to_contact)
    pts_v = {n: model.points[n] for n in VIOLIN_KEYPOINTS} | {"bow_frog": frog, "bow_tip": tip}
    pts_c = np.array([R @ pts_v[n] + t for n in KEYPOINT_NAMES])
    px = cam.project(pts_c)
    if noise_px > 0:
        px = px + (rng or np.random.default_rng()).normal(0, noise_px, px.shape)
    kp = Keypoints2D(xy=px.astype(np.float32), conf=np.ones(len(KEYPOINT_NAMES), np.float32))
    return SyntheticScene(kp, R, t, angle_deg, tilt_deg, contact_mm, R @ frog + t, R @ tip + t)


def default_pose() -> tuple[np.ndarray, np.ndarray]:
    """A plausible webcam view: violin ~70 cm away, face tilted toward the camera."""
    # Start with strings pointing image-left (scroll to the left), top plate facing camera.
    R0 = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])  # columns: x_v, y_v, z_v in camera frame
    R0[:, 1] = np.cross(R0[:, 2], R0[:, 0])  # keep right-handed
    tilt, _ = cv2.Rodrigues(np.array([0.5, 0.35, 0.15]))
    R = tilt @ R0
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), np.array([60.0, -60.0, 1050.0])


def facing_camera_pose(turn_deg: float = 0.0, elevation_deg: float = 55.0,
                       dist: float = 900.0) -> tuple[np.ndarray, np.ndarray]:
    """Player facing the camera: scroll to the player's left (image right), strings across the
    image, top plate tilted toward the camera by `elevation_deg`. `turn_deg` > 0 = the player
    has turned to their RIGHT (so the advisor should say "turn left")."""
    b = np.radians(elevation_deg)
    x = np.array([1.0, 0, 0])
    z = np.array([0, -np.cos(b), -np.sin(b)])
    R = np.c_[x, np.cross(z, x), z]
    c, s_ = np.cos(np.radians(turn_deg)), np.sin(np.radians(turn_deg))
    Ry = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    rvec, _ = cv2.Rodrigues(Ry @ R)
    return rvec.ravel(), np.array([0.0, 40.0, dist])


def clip_to_frame(p0: np.ndarray, p1: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip segment p0-p1 to the image (Liang-Barsky): what a detector would see of a long bow."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-d[0], p0[0]), (d[0], w - 1 - p0[0]), (-d[1], p0[1]), (d[1], h - 1 - p0[1])):
        if abs(pk) < 1e-12:
            if qk < 0:
                return None
            continue
        r = qk / pk
        if pk < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return p0 + t0 * d, p0 + t1 * d


# BGR fill / outline colours for the cartoon violin.
_PART_STYLE = {
    "body": ((40, 85, 150), (70, 130, 200)),
    "tailpiece": ((25, 25, 25), (110, 110, 110)),
    "fingerboard": ((25, 25, 25), (110, 110, 110)),
    "pegbox": ((40, 85, 150), (70, 130, 200)),
    "bridge": ((150, 200, 225), (150, 200, 225)),
}


def render_violin(img: np.ndarray, parts: dict[str, np.ndarray], cam: CameraIntrinsics, R: np.ndarray,
                  t: np.ndarray) -> None:
    """Draw a simple filled violin (back-to-front: body, tailpiece, fingerboard, bridge, strings)."""
    for name in ("body", "tailpiece", "fingerboard", "pegbox", "bridge"):
        px = cam.project((R @ parts[name].T).T + t).astype(np.int32)
        fill, edge = _PART_STYLE[name]
        if name != "bridge":
            cv2.fillPoly(img, [px], fill, cv2.LINE_AA)
        cv2.polylines(img, [px], False, edge, 2, cv2.LINE_AA)
    for name, line in parts.items():
        if name.startswith("string_"):
            px = cam.project((R @ line.T).T + t).astype(np.int32)
            cv2.polylines(img, [px], False, (200, 200, 200), 1, cv2.LINE_AA)


class DemoSource:
    """Animated synthetic take: the bow sweeps through good and bad angles.

    The bow plays G, D, A, E in turn (each at its own tilt), and
    the player slowly turns away from the camera and back, so the positioning
    indicator can be seen reacting. The bow is long enough to leave the frame:
    only the visible part is "detected", as in a real close-up take.
    Lets the GUI, 3D view and banners be exercised without a trained model.
    """

    STRING_SECONDS = 4.0

    def __init__(self, model: ViolinModel, cam: CameraIntrinsics):
        self.model, self.cam = model, cam
        self.parts = model.parts()

    def frame(self, t: float, noise_px: float = 1.0) -> tuple[np.ndarray, SyntheticScene]:
        turn = 35 * np.sin(2 * np.pi * t / 40.0)  # good -> turned right -> good -> turned left
        rvec, tvec = facing_camera_pose(turn_deg=turn)
        rvec = rvec + 0.03 * np.array([np.sin(0.7 * t), np.sin(0.5 * t), 0.0])
        tvec = tvec + np.array([20 * np.sin(0.4 * t), 10 * np.sin(0.6 * t), 0.0])
        string = ["G", "D", "A", "E"][int(t / self.STRING_SECONDS) % 4]
        angle = 90 + 12 * np.sin(0.45 * t)  # wanders in and out of tolerance
        tilt = -self.model.string_tilt_deg[string] + 2 * np.sin(0.9 * t)  # resting on that string
        stroke = 250 + 200 * np.sin(1.6 * t)  # up / down bow
        scene = make_scene(self.model, self.cam, rvec, tvec, angle, tilt, 30.0, stroke, noise_px)
        scene.string = string
        img = np.full((self.cam.height, self.cam.width, 3), (58, 52, 48), np.uint8)
        render_violin(img, self.parts, self.cam, scene.R, scene.t)
        f, tp = self.cam.project(np.vstack([scene.frog, scene.tip]))
        cv2.line(img, tuple(f.astype(int)), tuple(tp.astype(int)), (200, 200, 200), 3, cv2.LINE_AA)
        from .keypoints import KP_INDEX

        seen = clip_to_frame(f, tp, self.cam.width, self.cam.height)
        if seen is None:
            scene.keypoints.conf[[KP_INDEX["bow_frog"], KP_INDEX["bow_tip"]]] = 0
        else:
            scene.keypoints.xy[KP_INDEX["bow_frog"]], scene.keypoints.xy[KP_INDEX["bow_tip"]] = seen
        cv2.putText(img, "SYNTHETIC DEMO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2, cv2.LINE_AA)
        return img, scene
