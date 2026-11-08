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


class DemoSource:
    """Animated synthetic take: the bow sweeps through good and bad angles.

    Lets the GUI, 3D view and banner be exercised without a trained model.
    """

    def __init__(self, model: ViolinModel, cam: CameraIntrinsics):
        from .geometry import violin_wireframe

        self.model, self.cam = model, cam
        self.wire = violin_wireframe(model.string_length)
        self.base_r, self.base_t = default_pose()

    def frame(self, t: float, noise_px: float = 1.0) -> tuple[np.ndarray, SyntheticScene]:
        rvec = self.base_r + 0.05 * np.array([np.sin(0.7 * t), np.sin(0.5 * t), 0.0])
        tvec = self.base_t + np.array([20 * np.sin(0.4 * t), 10 * np.sin(0.6 * t), 0.0])
        angle = 90 + 12 * np.sin(0.45 * t)  # wanders in and out of tolerance
        tilt = 12 * np.sin(0.9 * t)
        stroke = 250 + 200 * np.sin(1.6 * t)  # up / down bow
        scene = make_scene(self.model, self.cam, rvec, tvec, angle, tilt, 30.0, stroke, noise_px)
        img = np.full((self.cam.height, self.cam.width, 3), (58, 52, 48), np.uint8)
        R, tv = scene.R, scene.t
        for line in self.wire:
            px = self.cam.project((R @ line.T).T + tv).astype(np.int32)
            cv2.polylines(img, [px], False, (90, 150, 210), 2, cv2.LINE_AA)
        f, tp = self.cam.project(np.vstack([scene.frog, scene.tip])).astype(int)
        cv2.line(img, tuple(f), tuple(tp), (200, 200, 200), 3, cv2.LINE_AA)
        cv2.putText(img, "SYNTHETIC DEMO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2, cv2.LINE_AA)
        return img, scene
