import numpy as np
import pytest

from bowtrack.config import load_config
from bowtrack.geometry import CameraIntrinsics, ViolinModel, ViolinPose
from bowtrack.positioning import PositionAdvisor, _rot_y, string_error_deg

MODEL = ViolinModel.from_config(load_config(use_fit=False))
CAM = CameraIntrinsics.from_fov(1280, 720, 70.0)


def good_pose(dist=900.0, elevation_deg=55.0):
    """Player faces the camera: scroll to the player's left (image right), strings across the
    image, top plate tilted toward the camera by `elevation_deg`."""
    b = np.radians(elevation_deg)
    x = np.array([1.0, 0, 0])
    z = np.array([0, -np.cos(b), -np.sin(b)])  # up (-y) and toward the camera (-z)
    R = np.c_[x, np.cross(z, x), z]
    return R, np.array([0.0, 40.0, dist])


def advise(R, t, string_known=False):
    adv = PositionAdvisor(MODEL)
    return adv.update(ViolinPose(R, t, 0.0, 10), CAM, string_known)


def test_good_position_is_ok():
    R, t = good_pose()
    a = advise(R, t)
    assert a.status == "ok", a.text
    assert a.string_error_deg < 5


@pytest.mark.parametrize("turned_right", [25.0, 40.0])
def test_turned_right_is_told_to_turn_left(turned_right):
    R, t = good_pose()
    # Player turned to their right by `turned_right` deg (inverse of a left turn).
    R_bad = _rot_y(np.radians([turned_right]))[0] @ R
    a = advise(R_bad, t)
    assert a.status == "adjust" and "to your left" in a.text, a.text
    assert 0 < a.turn_deg <= turned_right + 2
    assert a.best_error_deg < a.string_error_deg


def test_turned_left_is_told_to_turn_right():
    R, t = good_pose()
    a = advise(_rot_y(np.radians([-35.0]))[0] @ R, t)
    assert "to your right" in a.text, a.text


def test_error_grows_with_turn():
    R, t = good_pose()
    errs = string_error_deg(np.einsum("nij,jk->nik", _rot_y(np.radians([0.0, 20.0, 40.0])), R), t, MODEL)
    assert errs[0] < errs[1] < errs[2]


def test_too_close_step_back():
    R, t = good_pose(dist=350.0)
    assert advise(R, t).text.startswith("Step back")


def test_too_far_step_closer():
    R, t = good_pose(dist=2500.0)
    assert advise(R, t).text.startswith("Step closer")


def test_cut_off_on_image_left_moves_player_left():
    R, t = good_pose()
    t = t + np.array([-500.0, 0, 0])  # violin shifted toward the image's left edge
    assert advise(R, t).text.startswith("Move to your left")


def test_no_violin():
    assert PositionAdvisor(MODEL).update(None, CAM).status == "unknown"


def test_top_seen_edge_on_raise_camera():
    R, t = good_pose(elevation_deg=15.0)
    a = advise(R, t)
    assert a.text.startswith("Raise the camera"), a.text
    assert a.raise_error_deg < a.string_error_deg


def test_chosen_string_relaxes_the_requirement():
    R, t = good_pose(elevation_deg=33.0)  # ~7-8 deg if the string is unknown
    assert advise(R, t).status != "ok"
    assert advise(R, t, string_known=True).status == "ok"
