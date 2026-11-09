"""Is the violin sounding? (microphone or a recorded .wav)

A sounding note means the bow is resting on a string, so the bow is in its
playing plane and the angle can be judged; silence means nothing to measure
(the banner goes black instead of guessing from a bow held in the air).

Only the level is used, not the pitch: which string is played does not
matter once the position indicator has the player standing where the four
strings' bowing planes give nearly the same bow angle.

Plain numpy, so it is testable without an audio device; the Qt microphone
wrapper lives in gui/mic.py.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BLOCK = 2048  # analysis window (46 ms at 44.1 kHz)


@dataclass
class SoundState:
    level_db: float  # RMS level, dBFS
    sounding: bool


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-10))


class SoundAnalyzer:
    """Turns successive audio blocks into a debounced SoundState.

    Sounding = level above `threshold_db` AND `margin_db` above the tracked
    noise floor (room noise varies a lot between mics). The state is held for
    `hold_s` so the gap at a bow change doesn't blank the banner.
    """

    def __init__(self, sample_rate: float, threshold_db: float = -45.0, margin_db: float = 12.0,
                 hold_s: float = 0.15):
        self.sr = sample_rate
        self.threshold_db = threshold_db
        self.margin_db = margin_db
        self.hold_s = hold_s
        self.noise_floor_db = -70.0
        self._last_on = -1e9

    def update(self, block: np.ndarray, t: float) -> SoundState:
        level = rms_db(block)
        # Noise floor: drops quickly to quiet levels, rises only slowly.
        rate = 0.3 if level < self.noise_floor_db else 0.002
        self.noise_floor_db += rate * (level - self.noise_floor_db)
        loud = level > max(self.threshold_db, self.noise_floor_db + self.margin_db)
        if loud:
            self._last_on = t
        return SoundState(level, t - self._last_on <= self.hold_s)


# ---------------------------------------------------------------------------
# .wav next to a recorded take
# ---------------------------------------------------------------------------
def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return path


class AudioTrack:
    """Sound state at any time of a recorded take (for scrubbing / labelled frames)."""

    def __init__(self, path: str | Path, threshold_db: float = -45.0, margin_db: float = 12.0,
                 hold_s: float = 0.15):
        with wave.open(str(path), "rb") as w:
            self.sr = w.getframerate()
            ch, width = w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}[width]
        x = np.frombuffer(raw, dtype).astype(np.float64)
        x = (x - 128) / 128 if width == 1 else x / float(2 ** (8 * width - 1))
        self.samples = x.reshape(-1, ch).mean(1)
        self.path = Path(path)
        self.threshold_db, self.margin_db, self.hold_s = threshold_db, margin_db, hold_s
        # Per-block levels for the whole take: noise floor = a low percentile.
        hop = BLOCK // 2
        n = max(0, (len(self.samples) - BLOCK) // hop + 1)
        self._hop = hop
        self._levels = np.array([rms_db(self.samples[i * hop:i * hop + BLOCK]) for i in range(n)])
        self.noise_floor_db = float(np.percentile(self._levels, 10)) if n else -70.0

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr

    def state_at(self, t: float) -> SoundState:
        if not len(self._levels):
            return SoundState(-100.0, False)
        thresh = max(self.threshold_db, self.noise_floor_db + self.margin_db)
        i_end = int(np.clip((t * self.sr - BLOCK) / self._hop, 0, len(self._levels) - 1))
        i_start = max(0, i_end - int(self.hold_s * self.sr / self._hop))
        recent = self._levels[i_start:i_end + 1]
        loud = np.flatnonzero(recent > thresh)
        if not len(loud):
            return SoundState(float(self._levels[i_end]), False)
        return SoundState(float(self._levels[i_end]), True)


def wav_for_video(video_path: str | Path) -> Path:
    return Path(video_path).with_suffix(".wav")
