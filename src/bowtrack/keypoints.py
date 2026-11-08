"""Keypoint definitions shared by the model, annotation tool, and geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Order matters: it is the channel order of the model's heatmaps.
KEYPOINT_NAMES: list[str] = [
    "scroll",
    "nut",
    "bridge_bass",
    "bridge_treble",
    "chinrest",
    "bow_frog",
    "bow_tip",
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
KP_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

VIOLIN_KEYPOINTS = ["scroll", "nut", "bridge_bass", "bridge_treble", "chinrest"]
BOW_KEYPOINTS = ["bow_frog", "bow_tip"]

# Lines drawn between keypoints in the overlay.
SKELETON: list[tuple[str, str]] = [
    ("scroll", "nut"),
    ("bridge_bass", "bridge_treble"),
    ("bridge_bass", "chinrest"),
    ("bow_frog", "bow_tip"),
]

# BGR colours for OpenCV drawing.
KEYPOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "scroll": (255, 180, 0),
    "nut": (255, 255, 0),
    "bridge_bass": (0, 200, 255),
    "bridge_treble": (0, 140, 255),
    "chinrest": (200, 0, 200),
    "bow_frog": (60, 60, 255),
    "bow_tip": (180, 180, 255),
}


@dataclass
class Keypoints2D:
    """Per-frame 2D detections in full-image pixel coordinates."""

    xy: np.ndarray = field(default_factory=lambda: np.zeros((NUM_KEYPOINTS, 2), np.float32))
    conf: np.ndarray = field(default_factory=lambda: np.zeros(NUM_KEYPOINTS, np.float32))

    def visible(self, name: str, min_conf: float) -> bool:
        return bool(self.conf[KP_INDEX[name]] >= min_conf)

    def get(self, name: str) -> np.ndarray:
        return self.xy[KP_INDEX[name]]

    @classmethod
    def from_dict(cls, points: dict[str, list[float] | None]) -> "Keypoints2D":
        """Build from {name: [x, y]} (missing / None = not visible, conf 0)."""
        kp = cls()
        for name, value in points.items():
            if value is None or name not in KP_INDEX:
                continue
            i = KP_INDEX[name]
            kp.xy[i] = value[:2]
            kp.conf[i] = 1.0
        return kp

    def to_dict(self, min_conf: float = 0.5) -> dict[str, list[float] | None]:
        return {
            name: (self.xy[i].tolist() if self.conf[i] >= min_conf else None)
            for i, name in enumerate(KEYPOINT_NAMES)
        }
