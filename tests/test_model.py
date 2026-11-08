import numpy as np
import torch

from bowtrack.keypoints import NUM_KEYPOINTS, Keypoints2D
from bowtrack.model import HeatmapPoseNet, Letterbox, decode_heatmaps, gaussian_heatmaps


def test_forward_shape():
    m = HeatmapPoseNet("resnet18", pretrained=False).eval()
    out = m(torch.zeros(2, 3, 256, 448))
    assert out.shape == (2, NUM_KEYPOINTS, 64, 112)


def test_heatmap_roundtrip():
    xy = np.array([[100.3, 50.7], [300.0, 200.0]] + [[0, 0]] * (NUM_KEYPOINTS - 2), np.float32)
    vis = np.array([True, True] + [False] * (NUM_KEYPOINTS - 2))
    hm = gaussian_heatmaps(xy, vis, (256, 448), sigma=2.0)
    pred, conf = decode_heatmaps(torch.from_numpy(hm)[None])
    assert np.all(np.linalg.norm(pred[0, :2] - xy[:2], axis=1) < 2.0)
    assert conf[0, 0] > 0.9 and conf[0, 2] == 0.0


def test_letterbox_inverse():
    lb = Letterbox((720, 1280), (256, 448))
    p = np.array([[640.0, 360.0], [10.0, 700.0]], np.float32)
    assert np.allclose(lb.to_source(lb.to_input(p)), p, atol=1e-3)


def test_keypoints_dict_roundtrip():
    kp = Keypoints2D.from_dict({"scroll": [1.0, 2.0], "bow_tip": None})
    d = kp.to_dict()
    assert d["scroll"] == [1.0, 2.0] and d["bow_tip"] is None
