"""Fit the 3D violin model (and the camera focal length) to labelled frames.

Every labelled frame is a view of the same rigid violin, so the unknown 3D
keypoint positions, the focal length and each frame's pose can be solved
jointly by minimising reprojection error (bundle adjustment, as in
structure-from-motion).

Gauge / constraints:
  * origin = bridge centre, +x along the strings, nut fixed at (L, 0, 0)
    where L is the (measured) string length: this sets the scale;
  * the violin is left/right symmetric: centre-line points have y = 0 and
    bass/treble pairs mirror each other (this also fixes the roll gauge);
  * weak Gaussian priors pull each coordinate toward standard full-size
    dimensions, so a handful of frames cannot drift into nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import CameraIntrinsics
from .keypoints import VIOLIN_KEYPOINTS

# name -> (indices into the shared parameter vector for x, y, z), with
# y given as +param / -param / 0 for symmetric pairs.
# Parameter layout (14 values):
#   0 scroll_x 1 scroll_z | 2 bridge_y 3 bridge_z | 4 tp_top_x 5 tp_top_z
#   6 tp_end_x 7 tp_end_z | 8 up_x 9 up_y 10 up_z | 11 lo_x 12 lo_y 13 lo_z
PARAM_NAMES = ["scroll_x", "scroll_z", "bridge_half_width", "bridge_z", "tailpiece_top_x", "tailpiece_top_z",
               "tailpiece_end_x", "tailpiece_end_z", "upper_corner_x", "upper_corner_half_width",
               "upper_corner_z", "lower_corner_x", "lower_corner_half_width", "lower_corner_z"]

# Prior standard deviations (mm): how far the fit may move from standard dims
# without paying a price. Corners are least certain (no published standard).
PRIOR_SIGMA = np.array([20, 15, 4, 5, 10, 8, 15, 10, 25, 10, 10, 25, 10, 10], float)


def params_from_model(points: dict[str, np.ndarray]) -> np.ndarray:
    p = points
    return np.array([
        p["scroll"][0], p["scroll"][2],
        abs(p["bridge_bass"][1]), p["bridge_bass"][2],
        p["tailpiece_top"][0], p["tailpiece_top"][2],
        p["tailpiece_end"][0], p["tailpiece_end"][2],
        p["corner_upper_bass"][0], abs(p["corner_upper_bass"][1]), p["corner_upper_bass"][2],
        p["corner_lower_bass"][0], abs(p["corner_lower_bass"][1]), p["corner_lower_bass"][2],
    ], float)


def model_from_params(m: np.ndarray, string_length: float) -> dict[str, np.ndarray]:
    return {
        "scroll": np.array([m[0], 0, m[1]]),
        "nut": np.array([string_length, 0, 0.0]),
        "bridge_bass": np.array([0, m[2], m[3]]),
        "bridge_treble": np.array([0, -m[2], m[3]]),
        "tailpiece_top": np.array([m[4], 0, m[5]]),
        "tailpiece_end": np.array([m[6], 0, m[7]]),
        "corner_upper_bass": np.array([m[8], m[9], m[10]]),
        "corner_upper_treble": np.array([m[8], -m[9], m[10]]),
        "corner_lower_bass": np.array([m[11], m[12], m[13]]),
        "corner_lower_treble": np.array([m[11], -m[12], m[13]]),
    }


@dataclass
class Observation:
    file: str
    names: list[str]
    px: np.ndarray  # (N, 2)


@dataclass
class FitResult:
    points: dict[str, np.ndarray]
    focal: float
    hfov_deg: float
    rms_before: float
    rms_after: float
    per_keypoint_rms: dict[str, float]
    per_frame_rms: dict[str, float]
    param_shift: dict[str, float]
    param_std: dict[str, float]  # 1-sigma uncertainty (mm) from the Jacobian


def _rodrigues_batch(rvecs: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(rvecs, axis=1, keepdims=True)
    k = rvecs / np.maximum(th, 1e-12)
    K = np.zeros((len(rvecs), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def fit_violin_model(obs: list[Observation], init_points: dict[str, np.ndarray], string_length: float,
                     image_wh: tuple[int, int], init_hfov_deg: float, fit_focal: bool = True,
                     focal_prior_rel: float = 0.15, robust_px: float = 3.0,
                     measured: dict[str, float] | None = None, measured_sigma: float = 1.5) -> FitResult:
    """`measured` maps PARAM_NAMES to ruler measurements (mm), which get a tight prior."""
    w, h = image_wh
    cx, cy = w / 2, h / 2
    f0 = (w / 2) / np.tan(np.radians(init_hfov_deg) / 2)
    m0 = params_from_model(init_points)
    sigma = PRIOR_SIGMA.copy()
    for name, value in (measured or {}).items():
        i = PARAM_NAMES.index(name)
        m0[i], sigma[i] = float(value), measured_sigma
    cam0 = CameraIntrinsics.from_fov(w, h, init_hfov_deg)

    # Initial per-frame poses from PnP with the starting model.
    poses = []
    for o in obs:
        objp = np.array([init_points[n] for n in o.names])
        ok, rv, tv = cv2.solvePnP(objp, o.px, cam0.K, cam0.dist, flags=cv2.SOLVEPNP_SQPNP)
        rv, tv = cv2.solvePnPRefineLM(objp, o.px, cam0.K, cam0.dist, rv, tv)
        poses.append(np.r_[rv.ravel(), tv.ravel()])
    poses = np.array(poses)

    name_idx = {n: i for i, n in enumerate(VIOLIN_KEYPOINTS)}
    frame_of = np.concatenate([[i] * len(o.names) for i, o in enumerate(obs)])
    kp_of = np.concatenate([[name_idx[n] for n in o.names] for o in obs])
    px_all = np.vstack([o.px for o in obs])

    def unpack(x):
        m = x[:14]
        logf = x[14]
        P = x[15:].reshape(-1, 6)
        return m, np.exp(logf), P

    def project(x):
        m, f, P = unpack(x)
        pts = model_from_params(m, string_length)
        X = np.array([pts[n] for n in VIOLIN_KEYPOINTS])[kp_of]  # (M, 3)
        R = _rodrigues_batch(P[:, :3])[frame_of]
        Xc = np.einsum("mij,mj->mi", R, X) + P[frame_of, 3:]
        return np.c_[f * Xc[:, 0] / Xc[:, 2] + cx, f * Xc[:, 1] / Xc[:, 2] + cy]

    def residuals(x):
        r = (project(x) - px_all).ravel()
        prior = (x[:14] - m0) / sigma
        fprior = [(x[14] - np.log(f0)) / focal_prior_rel] if fit_focal else [(x[14] - np.log(f0)) * 1e3]
        return np.r_[r, prior, fprior]

    x0 = np.r_[m0, np.log(f0), poses.ravel()]
    rms0 = float(np.sqrt(np.mean(np.sum((project(x0) - px_all) ** 2, axis=1))))
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=robust_px, x_scale="jac", max_nfev=2000)
    m, f, _ = unpack(sol.x)
    err = np.linalg.norm(project(sol.x) - px_all, axis=1)

    per_kp = {n: float(np.sqrt(np.mean(err[kp_of == i] ** 2))) for n, i in name_idx.items() if np.any(kp_of == i)}
    per_frame = {o.file: float(np.sqrt(np.mean(err[frame_of == i] ** 2))) for i, o in enumerate(obs)}
    # Gauss-Newton covariance, scaled by the residual variance of the image terms.
    J = sol.jac
    n_img = len(px_all) * 2
    dof = max(1, n_img - len(sol.x))
    s2 = float(np.sum(sol.fun[:n_img] ** 2) / dof)
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        std = np.sqrt(np.clip(np.diag(cov)[:14], 0, None))
    except np.linalg.LinAlgError:
        std = np.full(14, np.nan)
    return FitResult(
        points=model_from_params(m, string_length),
        focal=float(f),
        hfov_deg=float(np.degrees(2 * np.arctan((w / 2) / f))),
        rms_before=rms0,
        rms_after=float(np.sqrt(np.mean(err**2))),
        per_keypoint_rms=per_kp,
        per_frame_rms=per_frame,
        param_shift={n: float(a - b) for n, a, b in zip(PARAM_NAMES, m, m0)},
        param_std=dict(zip(PARAM_NAMES, std.tolist())),
    )
