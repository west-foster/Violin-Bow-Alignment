"""Keypoint annotation store (simple JSON, one entry per labelled frame).

data/annotations.json
{
  "keypoint_names": [...],
  "images": [
    {"file": "frames/take_..._000123.jpg", "width": 1280, "height": 720,
     "keypoints": {"scroll": [x, y], "bow_tip": null, ...}}
  ]
}
A null keypoint means "not visible in this frame" (trained as absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .keypoints import KEYPOINT_NAMES, LEGACY_RENAMES


class AnnotationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "annotations.json"
        self.images: dict[str, dict] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for entry in data.get("images", []):
                kps = entry["keypoints"]
                for old, new in LEGACY_RENAMES.items():
                    if old in kps and new not in kps:
                        kps[new] = kps.pop(old)
                self.images[entry["file"]] = entry

    def __len__(self) -> int:
        return len(self.images)

    def get(self, file: str) -> dict[str, list[float] | None] | None:
        entry = self.images.get(file)
        return None if entry is None else entry["keypoints"]

    def put(self, file: str, image: np.ndarray, keypoints: dict[str, list[float] | None]) -> Path:
        """Save the frame image (if new) and its labels, then flush to disk."""
        img_path = self.root / file
        img_path.parent.mkdir(parents=True, exist_ok=True)
        if not img_path.exists():
            cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = image.shape[:2]
        self.images[file] = {
            "file": file,
            "width": w,
            "height": h,
            "keypoints": {n: keypoints.get(n) for n in KEYPOINT_NAMES},
        }
        self.save()
        return img_path

    def remove(self, file: str) -> None:
        """Delete a frame's labels and its saved image."""
        if self.images.pop(file, None) is not None:
            (self.root / file).unlink(missing_ok=True)
            self.save()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {"keypoint_names": KEYPOINT_NAMES, "images": list(self.images.values())}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.path)

    @staticmethod
    def missing(keypoints: dict) -> list[str]:
        """Keypoints never labelled on this frame (added after it was saved).

        A key with value None means "marked not visible", which is a real label.
        """
        return [n for n in KEYPOINT_NAMES if n not in keypoints]

    def entries(self) -> list[dict]:
        return list(self.images.values())
