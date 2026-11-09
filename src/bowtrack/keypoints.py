"""Keypoint definitions shared by the model, annotation tool, and geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Order matters: it is the channel order of the model's heatmaps.
# "upper" = toward the neck/scroll, "lower" = toward the tailpiece/chin.
# "bass" = G-string side, "treble" = E-string side.
KEYPOINT_NAMES: list[str] = [
    "scroll",
    "nut",
    "bridge_bass",
    "bridge_treble",
    "tailpiece_top",
    "tailpiece_end",
    "corner_upper_bass",
    "corner_upper_treble",
    "corner_lower_bass",
    "corner_lower_treble",
    "bow_frog",
    "bow_tip",
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
KP_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

BOW_KEYPOINTS = ["bow_frog", "bow_tip"]
VIOLIN_KEYPOINTS = [n for n in KEYPOINT_NAMES if n not in BOW_KEYPOINTS]

# Keyboard shortcut per keypoint in the labelling tool (same order as above).
KEYPOINT_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "F", "T"]

# Older label files used other names; they are renamed on load.
LEGACY_RENAMES = {"tailpiece": "tailpiece_top"}

# Lines drawn between keypoints in the overlay.
SKELETON: list[tuple[str, str]] = [
    ("scroll", "nut"),
    ("bridge_bass", "bridge_treble"),
    ("tailpiece_top", "tailpiece_end"),
    ("corner_upper_bass", "corner_lower_bass"),
    ("corner_upper_treble", "corner_lower_treble"),
    ("bow_frog", "bow_tip"),
]

_CORNER = ("Tip of the {where} corner of the body, where the purfling (the inlaid line along the edge) "
           "comes to a point beside the C-bout. Press N if it's hidden.")

# Shown in the labelling panel so every frame is labelled consistently.
KEYPOINT_HELP: dict[str, str] = {
    "scroll": "Outermost point of the scroll curl, at the very end of the pegbox.",
    "nut": "Top of the fingerboard where the strings leave the nut (centre, between D and A).",
    "bridge_bass": "Top of the bridge where the G string (thickest) crosses it.",
    "bridge_treble": "Top of the bridge where the E string (thinnest) crosses it.",
    "tailpiece_top": "Centre of the tailpiece's TOP edge: the end nearest the bridge, between the D and A "
                     "strings. Not the chin rest.",
    "tailpiece_end": "Centre of the tailpiece's BOTTOM end, where it loops toward the end button. "
                     "Press N if your jaw or the chin rest hides it.",
    "corner_upper_bass": _CORNER.format(where="G-side corner nearest the neck (upper bass)"),
    "corner_upper_treble": _CORNER.format(where="E-side corner nearest the neck (upper treble)"),
    "corner_lower_bass": _CORNER.format(where="G-side corner nearest the tailpiece (lower bass)"),
    "corner_lower_treble": _CORNER.format(where="E-side corner nearest the tailpiece (lower treble)"),
    "bow_frog": "Where the hair meets the frog (front edge of the metal ferrule), just past your thumb. "
                "If it's out of frame, click where the bow leaves the frame (snaps to the edge); if your "
                "hand hides it, click where the hair disappears into the hand. The angle only needs the "
                "bow's LINE, so any visible frog-side point works.",
    "bow_tip": "Where the hair meets the tip of the bow. If the tip is out of frame, click where the stick "
               "leaves the frame (clicks near the edge snap to it).",
}

# BGR colours for OpenCV drawing.
KEYPOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "scroll": (255, 180, 0),
    "nut": (255, 255, 0),
    "bridge_bass": (0, 200, 255),
    "bridge_treble": (0, 140, 255),
    "tailpiece_top": (200, 0, 200),
    "tailpiece_end": (140, 0, 140),
    "corner_upper_bass": (120, 255, 120),
    "corner_upper_treble": (40, 190, 40),
    "corner_lower_bass": (255, 200, 120),
    "corner_lower_treble": (200, 130, 40),
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
            name = LEGACY_RENAMES.get(name, name)
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
