"""Recording webcam frames and random-access playback of video files."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    """Write raw BGR frames to an .mp4 file."""

    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.frames_written = 0

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(self, out_dir: str | Path, frame_size: tuple[int, int], fps: float = 30.0) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"take_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        w, h = frame_size
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"Could not open video writer for {self.path}")
        self.frames_written = 0
        return self.path

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            self.frames_written += 1

    def stop(self) -> Path | None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self.path


class VideoFile:
    """Random-access frame reader with a small LRU cache for scrubbing."""

    def __init__(self, path: str | Path, cache_size: int = 120):
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video {self.path}")
        self.num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._next_index = 0  # index the decoder will return on the next read()
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return self.num_frames

    def get(self, index: int) -> np.ndarray | None:
        if not 0 <= index < self.num_frames:
            return None
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        if index != self._next_index:
            # Short forward jumps: decode through (exact). Otherwise seek.
            if 0 < index - self._next_index <= 30:
                while self._next_index < index:
                    self._cap.grab()
                    self._next_index += 1
            else:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                self._next_index = index
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._next_index = index + 1
        self._cache[index] = frame
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def release(self) -> None:
        self._cap.release()
