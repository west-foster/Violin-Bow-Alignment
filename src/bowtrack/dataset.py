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


def split_entries(entries: list[dict], val_frac: float = 0.15, seed: int = 0,
                  min_takes: int = 6) -> tuple[list[dict], list[dict]]:
    """Train / validation split that keeps near-duplicate neighbouring frames apart.

    With many takes, whole takes are held out. With few (a take can hold most
    of the data), the last `val_frac` of each take's labelled frames (by frame
    number) is held out instead, so every take is in both sets.
    """
    def take(e):
        return e["file"].rsplit("_", 1)[0]

    takes = sorted({take(e) for e in entries})
    rng = np.random.default_rng(seed)
    if len(takes) >= min_takes:
        rng.shuffle(takes)
        n_val = max(1, round(len(takes) * val_frac))
        val_takes = set(takes[:n_val])
        return ([e for e in entries if take(e) not in val_takes], [e for e in entries if take(e) in val_takes])
    train, val = [], []
    for t in takes:
        es = sorted((e for e in entries if take(e) == t), key=lambda e: e["file"])
        n_val = int(round(len(es) * val_frac)) if len(es) >= 4 else 0
        train += es[:len(es) - n_val]
        val += es[len(es) - n_val:]
    return train, val
