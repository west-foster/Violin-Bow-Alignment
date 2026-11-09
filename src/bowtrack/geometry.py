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
   point: one unknown left (how far it leans toward / away from the camera).
   * "contact" (default when the played string is known, i.e. chosen in the
     GUI): a bow sounding a string rests on it, so it lies in
     that string's bowing plane (the plane tangent to the arched bridge at
     that string). Plane ∩ interpretation plane = the bow direction, in closed
     form. Needs only the bow's 2D line -- any visible stretch of stick/hair,
     no bow length, both ends may be out of frame.
   * "joint" (string unknown, both bow ends visible): MAP estimate using the
     known frog-tip length plus a prior that the tilt is near the string plane.
   * string unknown and an end cut off: assume the middle (D/A) plane; the
     error this can cause is reported as `string_spread_deg`. The position
     indicator (positioning.py) keeps the player at a viewpoint where this
     error is small, so the string doesn't need to be known.

4. Angle. angle = acos(d_bow . x_strings). 90 deg is a perfectly straight
   bow. < 90 means the tip points toward the scroll, > 90 toward the bridge.
   The bow's tilt (string crossings) is a rotation about the string axis and
   therefore does not change this angle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

from .keypoints import VIOLIN_KEYPOINTS, Keypoints2D

STRINGS = ["G", "D", "A", "E"]
# Bowing-plane tilt (deg, + toward the bass side). Adjacent double-stop planes
# on a standard bridge arch (radius ~42 mm, 11.3 mm spacing) are ~0 and ~+-15
# deg; a single string is bowed between its two neighbouring double-stop planes.
DEFAULT_STRING_TILT = {"G": 20.0, "D": 7.0, "A": -7.0, "E": -20.0}


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
    # Bowing-plane tilt per string: rotation of the plane's normal about the
    # string axis, + toward the bass side (deg). The bridge is arched, so each
    # string is bowed in a differently tilted plane.
    string_tilt_deg: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STRING_TILT))
    playing_region_mm: tuple[float, float] = (-15.0, 150.0)  # contact must be this far from the bridge

    @classmethod
    def from_config(cls, cfg: dict) -> "ViolinModel":
        vm = cfg["violin_model"]
        points = {k: np.asarray(vm[k], np.float64) for k in VIOLIN_KEYPOINTS}
        st = cfg.get("strings", {})
        tilt = {k: float(v) for k, v in (st.get("tilt_deg") or DEFAULT_STRING_TILT).items()}
        region = tuple(float(v) for v in st.get("playing_region_mm", (-15.0, 150.0)))
        return cls(points, float(vm["string_length"]), float(cfg["bow"]["hair_length"]), tilt, region)

    def string_plane_normal(self, string: str | None) -> np.ndarray:
        """Normal of the bowing plane for `string` (None = halfway between D and A), violin frame."""
        a = np.radians(self.string_tilt_deg[string] if string else 0.0)
        return np.array([0.0, np.sin(a), np.cos(a)])

    def parts(self) -> dict[str, np.ndarray]:
        """Outline polylines that follow this model's (possibly fitted) keypoints."""
        return violin_parts(self.string_length, self.points)

    def wireframe(self) -> list[np.ndarray]:
        return list(self.parts().values())


def violin_wireframe(string_length: float = 325.0, points: dict[str, np.ndarray] | None = None) -> list[np.ndarray]:
    """Violin outline (violin frame, mm) as a list of polylines, for drawing."""
    return list(violin_parts(string_length, points).values())


def _smooth(pts: np.ndarray, n: int) -> np.ndarray:
    """Interpolating cubic spline through 2D points (chord-length parameter)."""
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return CubicSpline(t, pts, axis=0)(np.linspace(0, t[-1], n))


# Standard full-size body (bass half, neck -> tail), bridge at x = 0, in mm.
# Bout widths follow published 4/4 measurements: upper ~166, C ~110,
# lower ~207, body length ~356, stop (body top edge -> bridge) ~195. Corners
# protrude from a deep C-bout. Only the corners move to the model's (fitted)
# corner keypoints; the bouts keep their standard size.
_UPPER = [(195, 0), (190, 30), (176, 56), (156, 74), (132, 82), (112, 83), (94, 81), (80, 78)]
_C_BOUT = [(66, 68), (56, 61), (42, 57), (28, 55), (14, 56), (3, 60), (-6, 67), (-11, 75)]
_LOWER = [(-22, 93), (-38, 100), (-62, 103.5), (-88, 103.5), (-115, 99), (-138, 86), (-152, 62), (-159, 32),
          (-161, 0)]
_STD_UPPER_CORNER = np.array([70.0, 75.0, -41.0])
_STD_LOWER_CORNER = np.array([-15.0, 84.0, -49.0])


def _body_outline(points: dict[str, np.ndarray] | None) -> np.ndarray:
    """Closed body outline on the top-plate edge, passing exactly through the corner keypoints."""
    up, lo = _STD_UPPER_CORNER.copy(), _STD_LOWER_CORNER.copy()
    if points is not None and "corner_upper_bass" in points and "corner_lower_bass" in points:
        up, lo = points["corner_upper_bass"].copy(), points["corner_lower_bass"].copy()
        up[1], lo[1] = abs(up[1]), abs(lo[1])
    upper = _smooth(np.array(_UPPER + [tuple(up[:2])], float), 40)
    c_bout = _smooth(np.array([tuple(up[:2])] + _C_BOUT + [tuple(lo[:2])], float), 30)
    lower = _smooth(np.array([tuple(lo[:2])] + _LOWER, float), 40)
    half = np.vstack([upper[:-1], c_bout[:-1], lower])
    # Plate-edge height: the string line is not parallel to the top, so the
    # edge drops further below the strings toward the tail.
    slope = (up[2] - lo[2]) / max(up[0] - lo[0], 1.0)
    z = lo[2] + slope * (half[:, 0] - lo[0])
    side = np.c_[half, z]
    return np.vstack([side, (side * [1, -1, 1])[::-1][1:]])


def violin_parts(string_length: float = 325.0, points: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Named violin outline polylines (violin frame, mm). Closed shapes repeat their first point.

    Pass the model's keypoints to make the body, tailpiece and scroll follow them.
    """
    parts: dict[str, np.ndarray] = {"body": _body_outline(points)}
    nut = string_length
    tp_top = points["tailpiece_top"] if points is not None and "tailpiece_top" in points else np.array([-55, 0, -15.0])
    tp_end = points["tailpiece_end"] if points is not None and "tailpiece_end" in points else np.array([-155, 0, -36.0])
    scroll = points["scroll"] if points is not None and "scroll" in points else np.array([nut + 105, 0, -15.0])
    # Fingerboard outline.
    fb_end = 55.0
    parts["fingerboard"] = np.array(
        [[fb_end, 21, -6], [nut, 12, -6], [nut, -12, -6], [fb_end, -21, -6], [fb_end, 21, -6]], float)
    # Pegbox + scroll.
    sx, sz = scroll[0], scroll[2]
    mid = nut + 0.7 * (sx - nut)
    parts["pegbox"] = np.array([[nut, 9, -8], [mid, 7, sz], [sx, 0, sz], [mid, -7, sz], [nut, -9, -8], [nut, 9, -8]],
                               float)
    # Tailpiece: wide top edge, narrow end.
    parts["tailpiece"] = np.array([[tp_top[0], 21, tp_top[2]], [tp_end[0], 10, tp_end[2]],
                                   [tp_end[0], -10, tp_end[2]], [tp_top[0], -21, tp_top[2]],
                                   [tp_top[0], 21, tp_top[2]]], float)
    # Bridge.
    bz = points["bridge_bass"][2] if points is not None and "bridge_bass" in points else -3.0
    parts["bridge"] = np.array([[0, 20, -35], [0, 17, bz], [0, 0, 0.5], [0, -17, bz], [0, -20, -35]], float)
    # Strings G, D, A, E run from the tailpiece over the bridge to the nut.
    for name, y_b, y_n, z_b in [("G", 17, 8, bz), ("D", 5.7, 2.7, 0), ("A", -5.7, -2.7, 0), ("E", -17, -8, bz)]:
        parts[f"string_{name}"] = np.array([[tp_top[0], y_b * 0.7, tp_top[2] + 1], [0, y_b, z_b], [nut, y_n, 0]],
                                           float)
    return parts


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ViolinPose:
    R: np.ndarray  # 3x3 violin -> camera
    t: np.ndarray  # (3,) camera-frame position of the bridge centre, mm
    reproj_error_px: float
    num_points: int
    dropped: str | None = None  # keypoint rejected as an outlier, if any
    residuals: dict[str, float] | None = None  # per-keypoint reprojection error, px

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

    @property
    def view_angle_to_strings_deg(self) -> float:
        """Angle between the camera's line of sight and the strings.

        90 deg = strings lie flat across the image (best: the bow angle is then
        directly visible). Small values = the camera looks along the strings,
        where the single-camera depth ambiguity corrupts the bow angle.
        """
        v = -self.t / np.linalg.norm(self.t)
        return float(np.degrees(np.arccos(min(1.0, abs(float(v @ self.string_axis))))))


@dataclass
class BowEstimate:
    frog: np.ndarray  # camera-frame mm
    tip: np.ndarray
    contact: np.ndarray
    contact_from_bridge_mm: float  # along the strings, + toward the fingerboard
    angle_deg: float  # bow vs. strings, 90 = straight
    tilt_deg: float  # rotation out of the string plane (string crossing)
    angle_2d_deg: float  # same angle measured naively in the image
    method: str  # "contact G" etc., "joint", "joint (clipped)", "contact (string unknown)", "length", "plane"
    length_residual_mm: float
    string: str | None = None  # string assumed by the solver (None = unknown)
    # Range of angles this bow line gives across the four strings' bowing
    # planes: how much not knowing the string could matter in this view.
    string_spread_deg: float = 0.0


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
                 max_reproj_px: float = 25.0, outlier_px: float = 4.0, bow_solver: str = "auto",
                 sigma_length_mm: float = 25.0, sigma_tilt_deg: float = 12.0):
        self.model = model
        self.base_intrinsics = intrinsics
        self.intrinsics = intrinsics
        self.min_conf = min_conf
        self.max_reproj_px = max_reproj_px
        self.outlier_px = outlier_px
        self.bow_solver = bow_solver
        self.sigma_length_mm = sigma_length_mm
        self.sigma_tilt_deg = sigma_tilt_deg
        self._prev_pose: ViolinPose | None = None
        # String being played, if known (set by the GUI from the user's
        # choice). Read once per frame, so setting it from another
        # thread is safe.
        self.string: str | None = None

    def reset(self) -> None:
        self._prev_pose = None

    # -- violin --------------------------------------------------------------
    def estimate_pose(self, kp: Keypoints2D) -> ViolinPose | None:
        names = [n for n in VIOLIN_KEYPOINTS if kp.visible(n, self.min_conf)]
        if len(names) < 4:
            return None
        best, dropped_list = self._solve_pnp(kp, names), []
        # A bad keypoint (mislabel, model error, occlusion) drags the whole pose.
        # Greedy leave-one-out: drop the point whose removal helps most, if it
        # clearly helps, keeping at least 5 points (up to 2 drops).
        used = list(names)
        while (best is not None and best[0] > self.outlier_px and len(used) > 5 and len(dropped_list) < 2):
            trials = [(self._solve_pnp(kp, [n for n in used if n != d]), d) for d in used]
            trials = [(t, d) for t, d in trials if t is not None]
            if not trials:
                break
            alt, drop = min(trials, key=lambda td: td[0][0])
            if alt[0] > 0.6 * best[0]:
                break
            best, used = alt, [n for n in used if n != drop]
            dropped_list.append(drop)
        dropped = ", ".join(dropped_list) or None
        if best is None or best[0] > self.max_reproj_px:
            return None
        err, rvec, tvec, n_used = best
        R, _ = cv2.Rodrigues(rvec)
        # Residual of every labelled point (including a dropped one) under the final pose.
        obj = np.array([self.model.points[n] for n in names], np.float64)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.intrinsics.K, self.intrinsics.dist)
        res = np.linalg.norm(proj.reshape(-1, 2) - np.array([kp.get(n) for n in names]), axis=1)
        pose = ViolinPose(R, tvec.ravel(), err, n_used, dropped, dict(zip(names, res.tolist())))
        self._prev_pose = pose
        return pose

    def _solve_pnp(self, kp: Keypoints2D, names: list[str]):
        """Returns (rms_px, rvec, tvec, n_points) or None."""
        obj = np.array([self.model.points[n] for n in names], np.float64)
        img = np.array([kp.get(n) for n in names], np.float64)
        K, dist = self.intrinsics.K, self.intrinsics.dist

        candidates = []
        try:  # degenerate input (e.g. points almost on one line) makes OpenCV assert
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            ok = False
        if ok:
            candidates.append((rvec, tvec))
        if self._prev_pose is not None:  # temporal prior resolves near-planar flips
            rv0, _ = cv2.Rodrigues(self._prev_pose.R)
            try:
                ok2, rv2, tv2 = cv2.solvePnP(obj, img, K, dist, rv0.copy(), self._prev_pose.t.reshape(3, 1).copy(),
                                             useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                ok2 = False
            if ok2:
                candidates.append((rv2, tv2))

        best = None
        for rvec, tvec in candidates:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
            except cv2.error:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
            if tvec[2, 0] <= 0:
                continue
            if best is None or err < best[0] - 0.5:
                best = (err, rvec, tvec, len(names))
        return best

    # -- bow -----------------------------------------------------------------
    def bow_direction_on_string(self, n: np.ndarray, pose: ViolinPose, string: str | None) -> np.ndarray:
        """Bow direction resting on `string`: interpretation plane (normal n) ∩ that string's bowing plane."""
        d = np.cross(n, pose.R @ self.model.string_plane_normal(string))
        return d / np.linalg.norm(d)

    def estimate_bow(self, kp: Keypoints2D, pose: ViolinPose, string: str | None = None) -> BowEstimate | None:
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

        # A bow end touching the frame edge is probably cut off, so the visible
        # frog-tip span is shorter than the real bow: the length constraint is
        # invalid there and only the tilt prior is used.
        m = 3.0
        w, h = self.intrinsics.width, self.intrinsics.height
        clipped = any(p[0] < m or p[1] < m or p[0] > w - 1 - m or p[1] > h - 1 - m for p in (frog_px, tip_px))

        normal = pose.normal
        L0 = self.model.bow_length

        def geometry_at(theta: np.ndarray):
            """Bow through C at in-plane direction theta -> (F, T, length, tilt_deg, valid)."""
            theta = np.atleast_1d(theta)
            ux, uy = np.cos(theta), np.sin(theta)
            ends, valid = [], np.ones_like(theta, bool)
            for r in (rf2, rt2):
                # Solve C + mu*u = s*r with Cramer's rule.
                det = -r[0] * uy + ux * r[1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    sc = (-c2[0] * uy + ux * c2[1]) / det
                valid &= (np.abs(det) > 1e-9) & (sc > 0)
                p2 = sc[:, None] * r[None, :]
                ends.append(p2[:, :1] * e1 + p2[:, 1:] * e2)
            F, T = ends
            length = np.linalg.norm(T - F, axis=1)
            with np.errstate(invalid="ignore"):
                d = (T - F) / length[:, None]
                tilt = np.degrees(np.arcsin(np.clip(d @ normal, -1, 1)))
            return F, T, length, tilt, valid & np.isfinite(length)

        thetas = np.linspace(0, np.pi, 1441)[:-1]
        F_all, T_all, L_all, tilt_all, ok = geometry_at(thetas)
        if not ok.any():
            return None

        def theta_of(d: np.ndarray) -> float:  # in-plane angle (a line: theta and theta+pi are the same)
            return float(np.arctan2(d @ e2, d @ e1) % np.pi)

        # Angle each string's bowing plane would give (diagnostic + spread).
        per_string = []
        for s in STRINGS:
            d_s = self.bow_direction_on_string(n, pose, s)
            per_string.append(float(np.degrees(np.arccos(np.clip(abs(d_s @ x_s), 0, 1)))))
        spread = max(per_string) - min(per_string)

        solver = self.bow_solver
        if solver == "auto":
            solver = "contact" if (string is not None or clipped) else "joint"

        if solver == "contact":
            theta = theta_of(self.bow_direction_on_string(n, pose, string))
            method = f"contact {string}" if string else "contact (string unknown)"
        elif solver == "switch":  # legacy: length roots, else assume tilt = 0
            err = np.where(ok, L_all - L0, np.nan)
            roots = [i for i in range(len(thetas) - 1)
                     if np.isfinite(err[i]) and np.isfinite(err[i + 1]) and np.sign(err[i]) != np.sign(err[i + 1])]
            if roots and not clipped:
                i = min(roots, key=lambda i: abs(tilt_all[i]))
                theta = brentq(lambda t: geometry_at(t)[2][0] - L0, thetas[i], thetas[i + 1], xtol=1e-9)
                method = "length"
            else:
                i = int(np.nanargmin(np.where(ok, np.abs(tilt_all), np.nan)))
                step = thetas[1] - thetas[0]
                theta = float(minimize_scalar(lambda t: abs(geometry_at(t)[3][0]), method="bounded",
                                              bounds=(thetas[i] - step, thetas[i] + step),
                                              options={"xatol": 1e-9}).x)
                method = "plane"
        else:
            # MAP estimate: bow length ~ L0 (frog/tip clicks are imprecise, the
            # frog is often under the hand) AND tilt near the string plane
            # (string crossings tilt the bow by up to ~+-20 deg). Both soft, so
            # the answer varies smoothly instead of switching between methods.
            def cost(L, tilt):
                c = (tilt / self.sigma_tilt_deg) ** 2
                if not clipped:
                    c = c + ((L - L0) / self.sigma_length_mm) ** 2
                return c

            costs = np.where(ok, cost(L_all, tilt_all), np.inf)
            i = int(np.argmin(costs))
            step = thetas[1] - thetas[0]
            res = minimize_scalar(lambda t: (lambda g: cost(g[2][0], g[3][0]) if g[4][0] else 1e9)(geometry_at(t)),
                                  bounds=(thetas[i] - step, thetas[i] + step), method="bounded",
                                  options={"xatol": 1e-6})
            theta = float(res.x)
            method = "joint (clipped)" if clipped else "joint"

        F_, T_, _, _, valid = geometry_at(theta)
        if not valid[0]:
            # The frog/tip rays meet this line behind the camera on one side
            # (bow line passes near the camera): fall back to the contact point
            # plus direction, oriented frog -> tip in the image.
            if solver != "contact":
                return None
            d = np.cos(theta) * e1 + np.sin(theta) * e2
            p0, p1 = self.intrinsics.project(np.vstack([C, C + 10 * d]))
            if (p1 - p0) @ (tip_px - frog_px) < 0:
                d = -d
            F, T = C - 250 * d, C + (L0 - 250) * d
        else:
            F, T = F_[0], T_[0]

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
            string=string if solver == "contact" else None, string_spread_deg=spread,
        )

    # -- full frame ----------------------------------------------------------
    def __call__(self, kp: Keypoints2D, frame_wh: tuple[int, int], string: str | None = None) -> FrameGeometry:
        """`string` overrides `self.string` (the string being played, if known)."""
        string = string or self.string
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
        out.bow = self.estimate_bow(kp, out.pose, string)
        if out.bow is None:
            out.message = "bow geometry degenerate"
            return out
        lo, hi = self.model.playing_region_mm
        if not lo <= out.bow.contact_from_bridge_mm <= hi:
            out.message = f"bow not on the strings ({out.bow.contact_from_bridge_mm:.0f} mm from the bridge)"
            out.bow = None
        return out
