"""Task 1: fetch frames from the webcam.

`CameraStream` grabs frames on a background thread and always hands back the
most recent one, so a slow consumer (the model, the GUI) never builds up lag.

    with CameraStream(index=0) as cam:
        ok, frame = cam.read()
"""

from __future__ import annotations

import platform
import threading
import time

import cv2
import numpy as np


def _backend() -> int:
    return cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def open_camera(index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    """Open a webcam and request a resolution / frame rate.

    Raises RuntimeError if the camera cannot be opened (on macOS, also check
    System Settings > Privacy & Security > Camera for your terminal / IDE).
    """
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. On macOS grant camera access to your "
            "terminal/IDE in System Settings > Privacy & Security > Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def list_cameras(max_index: int = 4) -> list[int]:
    """Return indices of cameras that open and deliver a frame."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _backend())
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found


class CameraStream:
    """Threaded webcam reader that always returns the latest frame."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._timestamp = 0.0
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._running = False
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "CameraStream":
        if self._running:
            return self
        self._cap = open_camera(self.index, self.width, self.height, self.fps)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="CameraStream", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def actual_fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap is not None else 0.0

    # -- reading -------------------------------------------------------------
    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._new_frame:
                self._frame = frame
                self._frame_id += 1
                self._timestamp = time.time()
                self._new_frame.notify_all()

    def read(self, wait_new: bool = False, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        """Return (ok, frame). With wait_new=True, block until a newer frame arrives."""
        with self._new_frame:
            if wait_new:
                last = self._frame_id
                self._new_frame.wait_for(lambda: self._frame_id != last, timeout=timeout)
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def read_with_id(self, timeout: float = 1.0) -> tuple[int, float, np.ndarray | None]:
        """Block until a new frame; return (frame_id, timestamp, frame)."""
        with self._new_frame:
            last = self._frame_id
            self._new_frame.wait_for(lambda: self._frame_id != last, timeout=timeout)
            if self._frame is None:
                return self._frame_id, self._timestamp, None
            return self._frame_id, self._timestamp, self._frame.copy()


def fetch_camera(index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> CameraStream:
    """Convenience: open and start a threaded webcam stream."""
    return CameraStream(index, width, height, fps).start()


if __name__ == "__main__":
    # Quick smoke test: python -m bowtrack.camera
    print("Cameras found:", list_cameras())
    with fetch_camera() as cam:
        t0, n = time.time(), 0
        while time.time() - t0 < 3:
            ok, frame = cam.read(wait_new=True)
            n += ok
        print(f"Got {n} frames in 3 s ({n / 3:.1f} fps), shape={None if frame is None else frame.shape}")
