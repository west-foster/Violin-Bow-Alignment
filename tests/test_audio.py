import numpy as np

from bowtrack.audio import AudioTrack, SoundAnalyzer, write_wav

SR = 44100


def violin_tone(f0, dur=0.2, amp=0.3, sr=SR, seed=0):
    """Bowed-string-like: sawtooth (all harmonics, 1/k) plus a little noise."""
    t = np.arange(int(dur * sr)) / sr
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 15) if k * f0 < sr / 2)
    rng = np.random.default_rng(seed)
    return amp * x / np.max(np.abs(x)) + rng.normal(0, 0.003, len(t))


def test_analyzer_gates_on_sound_and_holds_briefly():
    an = SoundAnalyzer(SR)
    quiet = np.random.default_rng(2).normal(0, 0.0005, 2048)
    for i in range(20):  # learn the noise floor
        assert not an.update(quiet, i * 0.046).sounding
    s = an.update(violin_tone(440.0)[:2048], 1.0)
    assert s.sounding
    assert an.update(quiet, 1.1).sounding  # within the hold time (bow change)
    assert not an.update(quiet, 1.5).sounding


def test_audio_track_roundtrip(tmp_path):
    x = np.r_[np.zeros(SR // 2), violin_tone(293.7, dur=0.5), np.zeros(SR // 2)]
    track = AudioTrack(write_wav(tmp_path / "take.wav", x, SR))
    assert not track.state_at(0.3).sounding
    s = track.state_at(0.8)
    assert s.sounding
    assert not track.state_at(1.4).sounding
