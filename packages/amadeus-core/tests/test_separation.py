"""Tests for optional harmonic source separation before chroma extraction."""

import numpy as np
import pytest
import soundfile as sf

from amadeus_core.errors import ProcessingError
from amadeus_core.features import extract_chroma24, separate_harmonic


def _triad(sr: int = 22050, sec: float = 1.0) -> np.ndarray:
    t = np.linspace(0, sec, int(sr * sec), endpoint=False)
    freqs = (261.63, 329.63, 392.0)  # C major
    return (sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)).astype(
        np.float32
    )


def test_none_is_passthrough():
    y = _triad()
    assert separate_harmonic(y, 22050, method="none") is y


def test_empty_signal_passthrough():
    y = np.array([], dtype=np.float32)
    assert separate_harmonic(y, 22050, method="hpss").size == 0


def test_hpss_preserves_length_and_removes_percussive_click():
    sr = 22050
    y = _triad(sr=sr).copy()
    y[1000:1050] += 3.0  # a broadband percussive transient
    out = separate_harmonic(y, sr, method="hpss")
    assert out.shape == y.shape
    assert out.dtype == np.float32
    # The harmonic component should attenuate the percussive click.
    assert np.max(np.abs(out[1000:1050])) < np.max(np.abs(y[1000:1050]))


def test_unknown_method_raises():
    with pytest.raises(ProcessingError):
        separate_harmonic(_triad(), 22050, method="bogus")


def test_demucs_missing_dependency_is_reported_clearly():
    try:
        import demucs  # noqa: F401
    except Exception:
        with pytest.raises(ProcessingError, match="demucs"):
            separate_harmonic(_triad(), 22050, method="demucs")
    else:  # pragma: no cover - only when the optional dep happens to be installed
        pytest.skip("demucs is installed")


def test_extract_chroma24_with_hpss(tmp_path):
    sr = 22050
    path = tmp_path / "clip.wav"
    sf.write(str(path), _triad(sr=sr, sec=2.0), sr, subtype="PCM_16")
    feats, _timestamps, duration = extract_chroma24(str(path), separation="hpss")
    assert feats.shape[1] == 24
    assert duration == pytest.approx(2.0, abs=0.1)
