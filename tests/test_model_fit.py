import cv2
import numpy as np

from bowtrack.config import load_config
from bowtrack.geometry import CameraIntrinsics, ViolinModel
from bowtrack.keypoints import VIOLIN_KEYPOINTS
from bowtrack.model_fit import Observation, fit_violin_model, model_from_params, params_from_model
from bowtrack.synthetic import default_pose


def test_recovers_true_violin_and_focal():
    default = ViolinModel.from_config(load_config(use_fit=False))
    m_true = params_from_model(default.points) + np.array([20, -10, 1, 0, 5, 0, -8, 4, 30, 6, 5, -25, -5, 3.0])
    true_pts = model_from_params(m_true, default.string_length)
    cam = CameraIntrinsics.from_fov(1280, 720, 60.0)
    rng = np.random.default_rng(0)
    obs = []
    base_r, base_t = default_pose()
    for i in range(20):
        rv = base_r + rng.normal(0, 0.25, 3)
        tv = base_t + rng.normal(0, [80, 50, 120])
        R, _ = cv2.Rodrigues(rv)
        pts = np.array([R @ true_pts[n] + tv for n in VIOLIN_KEYPOINTS])
        px = cam.project(pts) + rng.normal(0, 1.0, (len(pts), 2))
        obs.append(Observation(f"f{i}", list(VIOLIN_KEYPOINTS), px))

    res = fit_violin_model(obs, default.points, default.string_length, (1280, 720), init_hfov_deg=70.0)
    assert res.rms_before > 5 and res.rms_after < 1.5
    assert abs(res.hfov_deg - 60.0) < 3.0
    for n in VIOLIN_KEYPOINTS:
        assert np.linalg.norm(res.points[n] - true_pts[n]) < 8.0, (n, res.points[n], true_pts[n])
