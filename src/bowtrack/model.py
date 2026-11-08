"""Heatmap keypoint network (SimpleBaseline-style, Xiao et al. 2018).

ImageNet-pretrained torchvision backbone -> 3 upsampling stages -> one heatmap
per keypoint at 1/4 input resolution. The peak location is the keypoint and the
peak value is its confidence (a hidden keypoint is trained to an all-zero map).

Design choices with the later distillation / INT8 goal in mind:
  * The same head works on any backbone, so a large FP32 teacher (resnet50)
    and a small student (resnet18 / mobilenet_v3) produce identical outputs,
    which makes heatmap knowledge-distillation trivial.
  * The head uses nearest-upsample + conv + BN + ReLU instead of transposed
    convs, so it fuses and quantizes cleanly with torch.ao FX quantization.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from .keypoints import NUM_KEYPOINTS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
HEATMAP_STRIDE = 4


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name.startswith("resnet"):
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        body = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc
        channels = net.fc.in_features
        return body, channels
    if name == "mobilenet_v3_large":
        weights = "DEFAULT" if pretrained else None
        net = torchvision.models.mobilenet_v3_large(weights=weights)
        return net.features, 960
    raise ValueError(f"Unknown backbone {name!r}")


class HeatmapPoseNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        num_keypoints: int = NUM_KEYPOINTS,
        pretrained: bool = True,
        head_channels: int | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, in_ch = _build_backbone(backbone, pretrained)
        head_channels = head_channels or (256 if in_ch >= 1024 else 128)
        layers: list[nn.Module] = []
        for _ in range(3):  # stride 32 -> 4
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, head_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
            ]
            in_ch = head_channels
        self.head = nn.Sequential(*layers)
        self.final = nn.Conv2d(head_channels, num_keypoints, 1)
        nn.init.normal_(self.final.weight, std=0.001)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.head(self.backbone(x)))


# ---------------------------------------------------------------------------
# Pre / post-processing
# ---------------------------------------------------------------------------
class Letterbox:
    """Resize with preserved aspect ratio + padding; remembers the mapping."""

    def __init__(self, src_hw: tuple[int, int], dst_hw: tuple[int, int]):
        sh, sw = src_hw
        dh, dw = dst_hw
        self.scale = min(dw / sw, dh / sh)
        self.new_w, self.new_h = round(sw * self.scale), round(sh * self.scale)
        self.pad_x = (dw - self.new_w) // 2
        self.pad_y = (dh - self.new_h) // 2
        self.dst_hw = dst_hw

    def apply(self, img: np.ndarray) -> np.ndarray:
        dh, dw = self.dst_hw
        out = np.zeros((dh, dw, 3), img.dtype)
        resized = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        out[self.pad_y : self.pad_y + self.new_h, self.pad_x : self.pad_x + self.new_w] = resized
        return out

    def to_input(self, xy: np.ndarray) -> np.ndarray:
        return xy * self.scale + np.array([self.pad_x, self.pad_y], np.float32)

    def to_source(self, xy: np.ndarray) -> np.ndarray:
        return (xy - np.array([self.pad_x, self.pad_y], np.float32)) / self.scale


def normalize(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3 -> normalized float tensor 3xHxW."""
    rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def decode_heatmaps(heatmaps: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(B,K,H,W) heatmaps -> (B,K,2) xy in *input-image* pixels, (B,K) confidence.

    Argmax plus the classic quarter-pixel shift toward the higher neighbour.
    """
    hm = heatmaps.detach().float().cpu().numpy()
    b, k, h, w = hm.shape
    flat = hm.reshape(b, k, -1)
    idx = flat.argmax(-1)
    conf = flat.max(-1)
    xs = (idx % w).astype(np.float32)
    ys = (idx // w).astype(np.float32)
    for bi in range(b):
        for ki in range(k):
            x, y = int(xs[bi, ki]), int(ys[bi, ki])
            m = hm[bi, ki]
            if 0 < x < w - 1:
                xs[bi, ki] += 0.25 * np.sign(m[y, x + 1] - m[y, x - 1])
            if 0 < y < h - 1:
                ys[bi, ki] += 0.25 * np.sign(m[y + 1, x] - m[y - 1, x])
    xy = np.stack([xs, ys], -1) * HEATMAP_STRIDE + (HEATMAP_STRIDE - 1) / 2
    return xy, conf


def gaussian_heatmaps(
    xy: np.ndarray, visible: np.ndarray, out_hw: tuple[int, int], sigma: float
) -> np.ndarray:
    """Training targets: (K,2) input-pixel coords -> (K,H/4,W/4) Gaussians."""
    h, w = out_hw[0] // HEATMAP_STRIDE, out_hw[1] // HEATMAP_STRIDE
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    maps = np.zeros((len(xy), h, w), np.float32)
    for i, ((x, y), v) in enumerate(zip(xy, visible)):
        if not v:
            continue
        cx = (x - (HEATMAP_STRIDE - 1) / 2) / HEATMAP_STRIDE
        cy = (y - (HEATMAP_STRIDE - 1) / 2) / HEATMAP_STRIDE
        if -3 * sigma < cx < w + 3 * sigma and -3 * sigma < cy < h + 3 * sigma:
            maps[i] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return maps
