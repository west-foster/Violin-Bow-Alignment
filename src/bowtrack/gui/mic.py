"""Live microphone input (Qt Multimedia, no extra dependency) -> SoundState.

Also buffers the raw samples while a take is being recorded so the GUI can
save them as a .wav next to the .mp4 (OpenCV's video writer has no audio).
"""

from __future__ import annotations

import time

import numpy as np
from PySide6 import QtCore
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

from ..audio import BLOCK, SoundAnalyzer, SoundState


class MicInput(QtCore.QObject):
    state_changed = QtCore.Signal(object)  # SoundState
    failed = QtCore.Signal(str)

    def __init__(self, threshold_db: float = -45.0, margin_db: float = 12.0, hold_s: float = 0.15,
                 parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._params = (threshold_db, margin_db, hold_s)
        self.source: QAudioSource | None = None
        self.io = None
        self.sample_rate = 44100
        self._buf = np.zeros(0, np.float32)
        self._since_analysis = 0
        self.analyzer: SoundAnalyzer | None = None
        self.state: SoundState | None = None
        self.device_name = ""
        self._recording: list[np.ndarray] | None = None
        self._last_data = 0.0
        self._watchdog = QtCore.QTimer(self)
        self._watchdog.timeout.connect(self._check_alive)

    @property
    def active(self) -> bool:
        return self.source is not None

    def start(self) -> bool:
        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            self.failed.emit("No microphone found")
            return False
        fmt = QAudioFormat()
        fmt.setSampleRate(44100)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Float)
        if not device.isFormatSupported(fmt):
            fmt = device.preferredFormat()
        self._fmt = fmt
        self.sample_rate = fmt.sampleRate()
        self.device_name = device.description()
        self.analyzer = SoundAnalyzer(self.sample_rate, *self._params)
        self.source = QAudioSource(device, fmt, self)
        self.source.setBufferSize(int(self.sample_rate * 0.1) * fmt.bytesPerFrame())
        self.io = self.source.start()
        if self.io is None:
            self.failed.emit("Could not open the microphone")
            self.stop()
            return False
        self.io.readyRead.connect(self._read)
        self._last_data = time.time()
        self._watchdog.start(1000)
        return True

    def stop(self) -> None:
        self._watchdog.stop()
        if self.source is not None:
            self.source.stop()
            self.source.deleteLater()
        self.source, self.io, self.state = None, None, None
        self._buf = np.zeros(0, np.float32)

    def _check_alive(self) -> None:
        if self.active and time.time() - self._last_data > 2.0:
            self.failed.emit("No audio from the microphone. On macOS allow microphone access for the app "
                             "running this (System Settings > Privacy & Security > Microphone), then restart it.")
            self._last_data = time.time()

    def _decode(self, raw: bytes) -> np.ndarray:
        f = self._fmt.sampleFormat()
        if f == QAudioFormat.Float:
            x = np.frombuffer(raw, np.float32)
        elif f == QAudioFormat.Int16:
            x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        elif f == QAudioFormat.Int32:
            x = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
        else:  # UInt8
            x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
        ch = max(1, self._fmt.channelCount())
        n = len(x) // ch * ch
        return x[:n].reshape(-1, ch).mean(1)

    def _read(self) -> None:
        if self.io is None:
            return
        raw = bytes(self.io.readAll())
        if not raw:
            return
        self._last_data = time.time()
        x = self._decode(raw)
        if self._recording is not None:
            self._recording.append(x.copy())
        self._buf = np.r_[self._buf, x][-BLOCK * 2:]
        self._since_analysis += len(x)
        # ~20 analyses per second over the newest BLOCK samples.
        if len(self._buf) >= BLOCK and self._since_analysis >= self.sample_rate // 20:
            self._since_analysis = 0
            self.state = self.analyzer.update(self._buf[-BLOCK:], time.time())
            self.state_changed.emit(self.state)

    # -- recording alongside a video take ------------------------------------
    def start_recording(self) -> None:
        self._recording = []

    def stop_recording(self) -> np.ndarray | None:
        chunks, self._recording = self._recording, None
        return np.concatenate(chunks) if chunks else None
