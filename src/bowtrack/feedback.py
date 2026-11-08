"""Task 5: turn per-frame angle estimates into a stable green / red / black state.

`FeedbackSink` is the hook for the future haptic servo: anything implementing
`send(state, error_deg)` (e.g. a pyserial writer to a microcontroller) can be
attached without touching the vision code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .geometry import FrameGeometry


class BowState(Enum):
    NO_BOW = "no_bow"  # black: nothing to measure
    OK = "ok"  # green
    TIP_TOWARD_SCROLL = "tip_toward_scroll"  # red: angle < target - tol
    TIP_TOWARD_BRIDGE = "tip_toward_bridge"  # red: angle > target + tol


@dataclass
class Feedback:
    state: BowState
    angle_deg: float | None  # smoothed
    error_deg: float | None  # smoothed angle - target (signed)
    text: str


class FeedbackSink(Protocol):
    def send(self, feedback: Feedback) -> None: ...


class BowFeedback:
    def __init__(self, target_deg: float = 90.0, tolerance_deg: float = 5.0, window: int = 5,
                 max_missing: int = 3):
        self.target = target_deg
        self.tolerance = tolerance_deg
        self._angles: deque[float] = deque(maxlen=max(1, window))
        self._missing = 0
        self._max_missing = max_missing  # tolerate brief dropouts before going black
        self.sinks: list[FeedbackSink] = []

    def reset(self) -> None:
        self._angles.clear()
        self._missing = 0

    def update(self, geo: FrameGeometry | None) -> Feedback:
        if geo is not None and geo.bow is not None:
            self._angles.append(geo.bow.angle_deg)
            self._missing = 0
        else:
            self._missing += 1
            if self._missing > self._max_missing:
                self._angles.clear()

        if not self._angles:
            reason = geo.message if geo is not None and geo.message else "no bow detected"
            fb = Feedback(BowState.NO_BOW, None, None, reason.capitalize())
        else:
            angle = float(np.median(self._angles))
            err = angle - self.target
            if abs(err) <= self.tolerance:
                fb = Feedback(BowState.OK, angle, err, f"Straight bow  {angle:.1f}°")
            elif err < 0:
                fb = Feedback(BowState.TIP_TOWARD_SCROLL, angle, err,
                              f"{angle:.1f}°  tip angled toward the scroll ({err:+.1f}°)")
            else:
                fb = Feedback(BowState.TIP_TOWARD_BRIDGE, angle, err,
                              f"{angle:.1f}°  tip angled toward the bridge ({err:+.1f}°)")
        for sink in self.sinks:
            sink.send(fb)
        return fb
