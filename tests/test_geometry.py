import numpy as np
import pytest

from bowtrack.config import load_config
from bowtrack.geometry import CameraIntrinsics, GeometryEstimator, ViolinModel
from bowtrack.keypoints import KP_INDEX
from bowtrack.synthetic import default_pose, make_scene

CFG = load_config(use_fit=False)
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
    # The legacy solver is exact on perfect data (checks the geometry itself).
    geo = GeometryEstimator(MODEL, CAM, bow_solver="switch")(scene.keypoints, (1280, 720))
    assert geo.pose is not None and geo.bow is not None
    assert geo.pose.reproj_error_px < 1e-3
    assert geo.bow.angle_deg == pytest.approx(angle, abs=0.05)
    assert geo.bow.contact_from_bridge_mm == pytest.approx(35.0, abs=0.5)
    assert np.linalg.norm(geo.bow.frog - scene.frog) < 1.0


@pytest.mark.parametrize("angle", [75.0, 90.0, 105.0])
@pytest.mark.parametrize("tilt", [-20.0, 0.0, 15.0])
def test_joint_solver_bias_is_small(angle, tilt):
    """The default (joint) solver trades a little bias for no method switching."""
    rvec, tvec = default_pose()
    scene = make_scene(MODEL, CAM, rvec, tvec, angle_deg=angle, tilt_deg=tilt, contact_mm=35.0)
    geo = GeometryEstimator(MODEL, CAM)(scene.keypoints, (1280, 720))
    assert geo.bow.method == "joint"
    assert geo.bow.angle_deg == pytest.approx(angle, abs=1.0)


def test_joint_solver_is_continuous():
    """Tiny label changes must not flip the answer (the old solver jumped ~12 deg)."""
    rvec, tvec = default_pose()
    rng = np.random.default_rng(3)
    for _ in range(30):
        scene = make_scene(MODEL, CAM, rvec, tvec, angle_deg=rng.uniform(80, 100), tilt_deg=rng.uniform(-20, 20))
        a = GeometryEstimator(MODEL, CAM)(scene.keypoints, (1280, 720)).bow.angle_deg
        scene.keypoints.xy[KP_INDEX["bow_frog"]] += np.array([1.0, -1.0], np.float32)
        b = GeometryEstimator(MODEL, CAM)(scene.keypoints, (1280, 720)).bow.angle_deg
        assert abs(a - b) < 1.0


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


def test_outlier_keypoint_rejected():
    rvec, tvec = default_pose()
    scene = make_scene(MODEL, CAM, rvec, tvec, angle_deg=90.0)
    scene.keypoints.xy[KP_INDEX["tailpiece_top"]] += np.array([60.0, -40.0], np.float32)  # badly off
    geo = GeometryEstimator(MODEL, CAM)(scene.keypoints, (1280, 720))
    assert geo.pose is not None and geo.pose.dropped == "tailpiece_top"
    assert geo.bow.angle_deg == pytest.approx(90.0, abs=0.5)


def test_clipped_bow_end_uses_plane_fallback():
    rvec, tvec = default_pose()
    scene = make_scene(MODEL, CAM, rvec, tvec, angle_deg=90.0, tilt_deg=0.0)
    fi, ti = KP_INDEX["bow_frog"], KP_INDEX["bow_tip"]
    f, t = scene.keypoints.xy[fi].copy(), scene.keypoints.xy[ti].copy()
    # Pretend the frame ends 70% of the way along the bow: the visible "tip" sits on the border.
    cut = f + 0.7 * (t - f)
    height = int(np.ceil(cut[1])) + 2
    cam = CameraIntrinsics(CAM.K, CAM.dist, 1280, height)
    scene.keypoints.xy[ti] = cut
    geo = GeometryEstimator(MODEL, cam)(scene.keypoints, (1280, height))
    assert geo.bow.method == "joint (clipped)"
    assert geo.bow.angle_deg == pytest.approx(90.0, abs=1.0)
    # Without clipping handling, the shortened span would distort the length solve.
    geo_full = GeometryEstimator(MODEL, CAM)(scene.keypoints, (1280, 720))
    assert geo_full.bow.method == "joint"
