"""PyTorch dataset over the labelled frames in data/annotations.json."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .annotations import AnnotationStore
from .keypoints import KEYPOINT_NAMES
from .model import gaussian_heatmaps, normalize


class KeypointDataset(Dataset):
    def __init__(self, store: AnnotationStore, entries: list[dict], input_hw: tuple[int, int],
                 sigma: float = 2.0, augment: bool = False):
        self.root = Path(store.root)
        self.entries = entries
        self.input_hw = input_hw
        self.sigma = sigma
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def _affine(self, w: int, h: int) -> np.ndarray:
        """Letterbox (+ random scale/rotate/shift) as one 2x3 matrix: source px -> input px."""
        dh, dw = self.input_hw
        s = min(dw / w, dh / h)
        rot, scale, tx, ty = 0.0, 1.0, 0.0, 0.0
        if self.augment:
            rot = np.random.uniform(-15, 15)
            scale = np.random.uniform(0.75, 1.3)
            tx, ty = np.random.uniform(-0.12, 0.12) * dw, np.random.uniform(-0.12, 0.12) * dh
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, s * scale)
        M[0, 2] += dw / 2 - w / 2 + tx
        M[1, 2] += dh / 2 - h / 2 + ty
        return M

    def __getitem__(self, i: int):
        e = self.entries[i]
        img = cv2.imread(str(self.root / e["file"]))
        h, w = img.shape[:2]
        xy = np.zeros((len(KEYPOINT_NAMES), 2), np.float32)
        vis = np.zeros(len(KEYPOINT_NAMES), bool)
        for k, n in enumerate(KEYPOINT_NAMES):
            p = e["keypoints"].get(n)
            if p is not None:
                xy[k], vis[k] = p, True

        M = self._affine(w, h)
        dh, dw = self.input_hw
        inp = cv2.warpAffine(img, M, (dw, dh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        xy_in = (np.c_[xy, np.ones(len(xy))] @ M.T).astype(np.float32)
        vis &= (xy_in[:, 0] >= 0) & (xy_in[:, 0] < dw) & (xy_in[:, 1] >= 0) & (xy_in[:, 1] < dh)

        if self.augment:
            alpha, beta = np.random.uniform(0.7, 1.3), np.random.uniform(-25, 25)
            inp = cv2.convertScaleAbs(inp, alpha=alpha, beta=beta)
            if np.random.rand() < 0.3:
                inp = cv2.GaussianBlur(inp, (5, 5), np.random.uniform(0.5, 1.5))

        target = gaussian_heatmaps(xy_in, vis, self.input_hw, self.sigma)
        return normalize(inp), torch.from_numpy(target), torch.from_numpy(xy_in), torch.from_numpy(vis)


def split_entries(entries: list[dict], val_frac: float = 0.15, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Split by recording (take) so near-duplicate consecutive frames don't leak into validation."""
    takes = sorted({e["file"].rsplit("_", 1)[0] for e in entries})
    rng = np.random.default_rng(seed)
    if len(takes) >= 3:
        rng.shuffle(takes)
        n_val = max(1, round(len(takes) * val_frac))
        val_takes = set(takes[:n_val])
        return ([e for e in entries if e["file"].rsplit("_", 1)[0] not in val_takes],
                [e for e in entries if e["file"].rsplit("_", 1)[0] in val_takes])
    idx = rng.permutation(len(entries))
    n_val = max(1, round(len(entries) * val_frac))
    return [entries[i] for i in idx[n_val:]], [entries[i] for i in idx[:n_val]]
