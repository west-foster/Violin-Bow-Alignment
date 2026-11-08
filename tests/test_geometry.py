import numpy as np
import pytest

from bowtrack.config import load_config
from bowtrack.geometry import CameraIntrinsics, GeometryEstimator, ViolinModel
from bowtrack.synthetic import default_pose, make_scene

CFG = load_config()
MODEL = ViolinModel.from_config(CFG)
CAM = CameraIntrinsics.from_fov(1280, 720, 70.0)


def _random_pose(rng):
    base_r, base_t = default_pose()
    rvec = base_r + rng.normal(0, 0.15, 3)
    tvec = base_t + rng.normal(0, [60, 40, 100])
    return rvec, tvec


@pytest.mark.parametrize("angle", [75.0, 85.0, 90.0, 95.0, 105.0])
@pytest.mark.parametrize("tilt", [-20.0, 0.0, 15.0])
def test_exact_recovery(angle, tilt):
    rvec, tvec = default_pose()
    scene = make_scene(MODEL, CAM, rvec, tvec, angle_deg=angle, tilt_deg=tilt, contact_mm=35.0)
    geo = GeometryEstimator(MODEL, CAM)(scene.keypoints, (1280, 720))
    assert geo.pose is not None and geo.bow is not None
    assert geo.pose.reproj_error_px < 1e-3
    assert geo.bow.angle_deg == pytest.approx(angle, abs=0.05)
    assert geo.bow.contact_from_bridge_mm == pytest.approx(35.0, abs=0.5)
    assert np.linalg.norm(geo.bow.frog - scene.frog) < 1.0


def test_noise_robustness():
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(60):
        rvec, tvec = _random_pose(rng)
        angle = rng.uniform(78, 102)
        scene = make_scene(MODEL, CAM, rvec, tvec, angle_deg=angle, tilt_deg=rng.uniform(-15, 15),
                           contact_mm=rng.uniform(15, 60), noise_px=1.5, rng=rng)
        geo = GeometryEstimator(MODEL, CAM)(scene.keypoints, (1280, 720))
        if geo.bow is not None:
            errors.append(abs(geo.bow.angle_deg - angle))
    assert len(errors) >= 50
    assert np.median(errors) < 2.0, np.median(errors)


def test_no_bow_reports_not_visible():
    rvec, tvec = default_pose()
    scene = make_scene(MODEL, CAM, rvec, tvec)
    scene.keypoints.conf[-2:] = 0.0
    geo = GeometryEstimator(MODEL, CAM)(scene.keypoints, (1280, 720))
    assert geo.pose is not None and not geo.bow_visible and geo.bow is None


def test_intrinsics_rescale():
    small = CAM.scaled_to(640, 360)
    assert small.K[0, 0] == pytest.approx(CAM.K[0, 0] / 2)
