"""Task 4: lift 2D keypoints into 3D and measure the bow/string angle.

Pipeline (single monocular camera, no depth sensor):

1. Violin pose. The violin is (nearly) rigid with known dimensions, so the
   visible violin keypoints are matched to a 3D violin model with solvePnP,
   giving rotation R and translation t (violin frame -> camera frame).

2. Contact point. The 2D bow line plus the camera centre span a plane in 3D
   (the bow's "interpretation plane"). Intersecting it with the 3D string axis
   gives the point where the bow crosses the strings, and its distance from the
   bridge (the "sounding point").

3. Bow direction. Inside that plane the bow is a line through the contact
   point; its two ends lie on the camera rays through the frog and tip pixels.
   The known frog-tip distance pins down the remaining degree of freedom (two
   mirror solutions are possible; we keep the one closest to lying on the
   strings, i.e. the smallest tilt). If the length equation has no solution
   (noisy keypoints), we fall back to assuming the bow lies in the string plane.

4. Angle. angle = acos(d_bow . x_strings). 90 deg is a perfectly straight
   bow. < 90 means the tip points toward the scroll, > 90 toward the bridge.
   The bow's tilt (string crossings) is a rotation about the string axis and
   therefore does not change this angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from .keypoints import VIOLIN_KEYPOINTS, Keypoints2D


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
@dataclass
class CameraIntrinsics:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)
        return cls(K, np.zeros(5), width, height)

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            np.array(data["K"], np.float64),
            np.array(data["dist"], np.float64).ravel(),
            int(data["width"]),
            int(data["height"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(
                {"K": self.K.tolist(), "dist": self.dist.ravel().tolist(), "width": self.width, "height": self.height}
            )
        )

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Rescale intrinsics if frames arrive at a different resolution."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        return CameraIntrinsics(K, self.dist, width, height)

    def rays(self, px: np.ndarray) -> np.ndarray:
        """Pixels (N,2) -> unit viewing rays (N,3) in camera coordinates."""
        pts = cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), self.K, self.dist).reshape(-1, 2)
        r = np.hstack([pts, np.ones((len(pts), 1))])
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """Camera-frame 3D points (N,3) -> pixels (N,2)."""
        px, _ = cv2.projectPoints(pts_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return px.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Violin model
# ---------------------------------------------------------------------------
@dataclass
class ViolinModel:
    points: dict[str, np.ndarray]
    string_length: float
    bow_length: float

    @classmethod
    def from_config(cls, cfg: dict) -> "ViolinModel":
        vm = cfg["violin_model"]
        points = {k: np.asarray(vm[k], np.float64) for k in VIOLIN_KEYPOINTS}
        return cls(points, float(vm["string_length"]), float(cfg["bow"]["hair_length"]))


def violin_wireframe(string_length: float = 325.0) -> list[np.ndarray]:
    """Rough 4/4 violin outline (violin frame, mm) as a list of polylines, for drawing."""
    # Body half-width vs. x (bridge at x=0, body from -161 to +195).
    cx = np.array([-161, -155, -140, -110, -75, -45, -20, 0, 25, 55, 90, 130, 165, 188, 195], float)
    cw = np.array([0, 55, 85, 102, 100, 85, 58, 55, 58, 75, 82, 78, 60, 35, 0], float)
    xs = np.linspace(cx[0], cx[-1], 120)
    hw = np.clip(CubicSpline(cx, cw)(xs), 0, None)
    z_body = -38.0
    body = np.vstack(
        [np.c_[xs, hw, np.full_like(xs, z_body)], np.c_[xs[::-1], -hw[::-1], np.full_like(xs, z_body)]]
    )
    lines = [body]
    # Fingerboard outline.
    fb_end, nut = 55.0, string_length
    fb = np.array(
        [[fb_end, 21, -6], [nut, 12, -6], [nut, -12, -6], [fb_end, -21, -6], [fb_end, 21, -6]], float
    )
    lines.append(fb)
    # Pegbox + scroll as a simple line.
    lines.append(np.array([[nut, 0, -6], [nut + 70, 0, -20], [nut + 105, 0, -15]], float))
    # Strings G, D, A, E.
    for y_b, y_n, z_b in [(17, 8, -3), (5.7, 2.7, 0), (-5.7, -2.7, 0), (-17, -8, -3)]:
        lines.append(np.array([[0, y_b, z_b], [nut, y_n, 0]], float))
    # Bridge.
    lines.append(np.array([[0, 20, -35], [0, 17, -3], [0, 0, 0.5], [0, -17, -3], [0, -20, -35]], float))
    return lines


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ViolinPose:
    R: np.ndarray  # 3x3 violin -> camera
    t: np.ndarray  # (3,) camera-frame position of the bridge centre, mm
    reproj_error_px: float
    num_points: int

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return (self.R @ np.atleast_2d(pts).T).T + self.t

    @property
    def string_axis(self) -> np.ndarray:  # toward scroll
        return self.R[:, 0]

    @property
    def bass_axis(self) -> np.ndarray:
        return self.R[:, 1]

    @property
    def normal(self) -> np.ndarray:  # out of the top plate
        return self.R[:, 2]


@dataclass
class BowEstimate:
    frog: np.ndarray  # camera-frame mm
    tip: np.ndarray
    contact: np.ndarray
    contact_from_bridge_mm: float  # along the strings, + toward the fingerboard
    angle_deg: float  # bow vs. strings, 90 = straight
    tilt_deg: float  # rotation out of the string plane (string crossing)
    angle_2d_deg: float  # same angle measured naively in the image
    method: str  # "length" or "plane"
    length_residual_mm: float


@dataclass
class FrameGeometry:
    pose: ViolinPose | None = None
    bow: BowEstimate | None = None
    bow_visible: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
class GeometryEstimator:
    def __init__(self, model: ViolinModel, intrinsics: CameraIntrinsics, min_conf: float = 0.35,
                 max_reproj_px: float = 25.0):
        self.model = model
        self.base_intrinsics = intrinsics
        self.intrinsics = intrinsics
        self.min_conf = min_conf
        self.max_reproj_px = max_reproj_px
        self._prev_pose: ViolinPose | None = None

    def reset(self) -> None:
        self._prev_pose = None

    # -- violin --------------------------------------------------------------
    def estimate_pose(self, kp: Keypoints2D) -> ViolinPose | None:
        names = [n for n in VIOLIN_KEYPOINTS if kp.visible(n, self.min_conf)]
        if len(names) < 4:
            return None
        obj = np.array([self.model.points[n] for n in names], np.float64)
        img = np.array([kp.get(n) for n in names], np.float64)
        K, dist = self.intrinsics.K, self.intrinsics.dist

        candidates = []
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            candidates.append((rvec, tvec))
        if self._prev_pose is not None:  # temporal prior resolves near-planar flips
            rv0, _ = cv2.Rodrigues(self._prev_pose.R)
            ok2, rv2, tv2 = cv2.solvePnP(obj, img, K, dist, rv0.copy(), self._prev_pose.t.reshape(3, 1).copy(),
                                         useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2:
                candidates.append((rv2, tv2))

        best = None
        for rvec, tvec in candidates:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
            if tvec[2, 0] <= 0:
                continue
            if best is None or err < best[0] - 0.5:
                best = (err, rvec, tvec)
        if best is None or best[0] > self.max_reproj_px:
            return None
        R, _ = cv2.Rodrigues(best[1])
        pose = ViolinPose(R, best[2].ravel(), best[0], len(names))
        self._prev_pose = pose
        return pose

    # -- bow -----------------------------------------------------------------
    def estimate_bow(self, kp: Keypoints2D, pose: ViolinPose) -> BowEstimate | None:
        frog_px, tip_px = kp.get("bow_frog").astype(np.float64), kp.get("bow_tip").astype(np.float64)
        if np.linalg.norm(tip_px - frog_px) < 5:
            return None
        r_f, r_t = self.intrinsics.rays(np.vstack([frog_px, tip_px]))
        n = np.cross(r_f, r_t)
        n /= np.linalg.norm(n)

        # Contact point: interpretation plane (n . X = 0) intersect string axis.
        x_s = pose.string_axis
        denom = float(n @ x_s)
        if abs(denom) < 1e-6:
            return None  # bow parallel to strings in the image
        lam = -float(n @ pose.t) / denom
        C = pose.t + lam * x_s

        # 2D coordinates inside the interpretation plane.
        e1 = C / np.linalg.norm(C)
        e2 = np.cross(n, e1)
        c2 = np.array([np.linalg.norm(C), 0.0])
        rf2 = np.array([r_f @ e1, r_f @ e2])
        rt2 = np.array([r_t @ e1, r_t @ e2])

        def endpoints(theta: float):
            u = np.array([np.cos(theta), np.sin(theta)])
            out = []
            for r in (rf2, rt2):
                # Solve C + mu*u = s*r  ->  [r, -u] [s, mu]^T = C
                A = np.column_stack([r, -u])
                if abs(np.linalg.det(A)) < 1e-9:
                    return None
                s, _ = np.linalg.solve(A, c2)
                if s <= 0:
                    return None
                out.append(s * r)
            return out

        def length_err(theta: float) -> float:
            ep = endpoints(theta)
            return np.nan if ep is None else float(np.linalg.norm(ep[1] - ep[0]) - self.model.bow_length)

        # Vectorised scan over line directions (Cramer's rule for both rays).
        thetas = np.linspace(0, np.pi, 721)
        ux, uy = np.cos(thetas), np.sin(thetas)
        pts, valid = [], np.ones_like(thetas, bool)
        for r in (rf2, rt2):
            det = -r[0] * uy + ux * r[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (-c2[0] * uy + ux * c2[1]) / det
            valid &= (np.abs(det) > 1e-9) & (s > 0)
            pts.append(s[:, None] * r[None, :])
        errs = np.where(valid, np.linalg.norm(pts[1] - pts[0], axis=1) - self.model.bow_length, np.nan)
        solutions = []
        for i in range(len(thetas) - 1):
            a, b = errs[i], errs[i + 1]
            if np.isfinite(a) and np.isfinite(b) and np.sign(a) != np.sign(b):
                theta = brentq(length_err, thetas[i], thetas[i + 1], xtol=1e-7)
                solutions.append(theta)

        def to_3d(theta: float):
            ep = endpoints(theta)
            if ep is None:
                return None
            F = ep[0][0] * e1 + ep[0][1] * e2
            T = ep[1][0] * e1 + ep[1][1] * e2
            return F, T

        candidates = []
        for theta in solutions:
            pts = to_3d(theta)
            if pts is None:
                continue
            F, T = pts
            d = (T - F) / np.linalg.norm(T - F)
            candidates.append((abs(float(d @ pose.normal)), F, T, "length"))

        if candidates:
            _, F, T, method = min(candidates, key=lambda c: c[0])
        else:
            # Fallback: bow lies in the string plane -> direction = n x normal.
            d = np.cross(n, pose.normal)
            if np.linalg.norm(d) < 1e-6:
                return None
            d /= np.linalg.norm(d)
            theta = float(np.arctan2(d @ e2, d @ e1)) % np.pi
            pts = to_3d(theta)
            if pts is None:
                return None
            F, T = pts
            method = "plane"

        d = (T - F) / np.linalg.norm(T - F)
        angle = float(np.degrees(np.arccos(np.clip(d @ x_s, -1, 1))))
        tilt = float(np.degrees(np.arcsin(np.clip(d @ pose.normal, -1, 1))))

        # Naive image-space angle, for comparison/debugging.
        s0, s1 = self.intrinsics.project(pose.to_camera(np.array([[0, 0, 0], [self.model.string_length, 0, 0]])))
        sv, bv = s1 - s0, tip_px - frog_px
        a2d = float(np.degrees(np.arccos(np.clip(sv @ bv / (np.linalg.norm(sv) * np.linalg.norm(bv)), -1, 1))))

        return BowEstimate(
            frog=F, tip=T, contact=C, contact_from_bridge_mm=float(lam), angle_deg=angle, tilt_deg=tilt,
            angle_2d_deg=a2d, method=method,
            length_residual_mm=float(np.linalg.norm(T - F) - self.model.bow_length),
        )

    # -- full frame ----------------------------------------------------------
    def __call__(self, kp: Keypoints2D, frame_wh: tuple[int, int]) -> FrameGeometry:
        self.intrinsics = self.base_intrinsics.scaled_to(*frame_wh)
        out = FrameGeometry()
        out.bow_visible = kp.visible("bow_frog", self.min_conf) and kp.visible("bow_tip", self.min_conf)
        out.pose = self.estimate_pose(kp)
        if out.pose is None:
            self._prev_pose = None
            out.message = "violin not found (need 4+ violin keypoints)"
            return out
        if not out.bow_visible:
            out.message = "no bow detected"
            return out
        out.bow = self.estimate_bow(kp, out.pose)
        if out.bow is None:
            out.message = "bow geometry degenerate"
        return out
