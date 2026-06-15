"""Tests for 24-dim chroma feature extraction."""

import numpy as np
import pytest

from amadeus_core.errors import ProcessingError
from amadeus_core.features import FEATURE_DIM, chroma24_from_audio


def test_chroma24_shape_and_dtype():
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    # A major triad (A4, C#5, E5).
    y = sum(np.sin(2 * np.pi * f * t) for f in (440.0, 554.37, 659.25)).astype(
        np.float32
    )

    features, timestamps = chroma24_from_audio(y, sr)

    assert features.ndim == 2
    assert features.shape[1] == FEATURE_DIM
    assert features.shape[0] == timestamps.shape[0]
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_chroma24_empty_signal_raises():
    with pytest.raises(ProcessingError):
        chroma24_from_audio(np.array([], dtype=np.float32), 22050)


def test_smoothing_preserves_shape():
    sr = 22050
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(
        np.float32
    )
    raw, _ = chroma24_from_audio(y, sr, smooth_window=0)
    smoothed, _ = chroma24_from_audio(y, sr, smooth_window=9)
    assert raw.shape == smoothed.shape
