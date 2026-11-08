"""Task 3: run the keypoint model on a full-resolution BGR frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .keypoints import Keypoints2D
from .model import HeatmapPoseNet, Letterbox, decode_heatmaps, normalize


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: str | Path, model: HeatmapPoseNet, input_hw: tuple[int, int], **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": model.backbone_name,
            "input_hw": list(input_hw),
            "state_dict": model.state_dict(),
            **extra,
        },
        path,
    )


def load_model(path: str | Path, device: torch.device | None = None) -> tuple[HeatmapPoseNet, tuple[int, int]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = HeatmapPoseNet(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device or pick_device())
    return model, tuple(ckpt["input_hw"])


class KeypointDetector:
    """Wraps a trained HeatmapPoseNet: frame (BGR) -> Keypoints2D in frame pixels."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = pick_device(device)
        self.model, self.input_hw = load_model(checkpoint, self.device)
        self._letterbox: Letterbox | None = None
        self._src_hw: tuple[int, int] | None = None

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> Keypoints2D:
        src_hw = frame_bgr.shape[:2]
        if self._src_hw != src_hw:
            self._letterbox = Letterbox(src_hw, self.input_hw)
            self._src_hw = src_hw
        assert self._letterbox is not None
        x = normalize(self._letterbox.apply(frame_bgr)).unsqueeze(0).to(self.device)
        heatmaps = self.model(x)
        xy, conf = decode_heatmaps(heatmaps)
        return Keypoints2D(
            xy=self._letterbox.to_source(xy[0]).astype(np.float32),
            conf=np.clip(conf[0], 0.0, 1.0).astype(np.float32),
        )
