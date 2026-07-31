"""
Audio -> 24-dim chroma feature extraction.

The shipped model + scaler were trained on McGill-Billboard NNLS "bothchroma"
features: a 24-dim vector per frame == bass-register chroma (12) concatenated with
treble-register chroma (12). librosa cannot reproduce NNLS chroma exactly, so we
*approximate* that layout with two band-limited CQT chromagrams. This keeps the
trained weights usable; fidelity is approximate-but-functional (documented in the
backend README / CLAUDE.md).
"""
from typing import Tuple

import librosa
import numpy as np

from app.errors import ProcessingError

# 24 == 12 (bass) + 12 (treble). Order must match how scaler.pkl was fit.
N_CHROMA = 12
FEATURE_DIM = 24
DEFAULT_HOP_LENGTH = 2048
BASS_FIRST = True  # concat order: True -> [bass(12) | treble(12)]

# Register boundaries (3 octaves each: C1-C4 bass, C4-C7 treble).
_BASS_FMIN = librosa.note_to_hz("C1")
_TREBLE_FMIN = librosa.note_to_hz("C4")
_N_OCTAVES = 3
_STAGE = "chromagram_extraction"


def _moving_average(matrix: np.ndarray, window: int) -> np.ndarray:
    """Moving average along the time axis (axis=1) of a (bins, frames) matrix."""
    if window <= 1:
        return matrix
    pad = window // 2
    padded = np.pad(matrix, ((0, 0), (pad, pad)), mode="edge")
    kernel = np.ones(window) / window
    out = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        out[i] = np.convolve(padded[i], kernel, mode="valid")[: matrix.shape[1]]
    return out


def chroma24_from_audio(
    y: np.ndarray,
    sr: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
    smooth_window: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute approximate 24-dim Billboard-style chroma from an in-memory signal.

    This is the single source of truth for the feature representation, shared by
    inference (:func:`extract_chroma24`) and training, so train-time and
    inference-time features are guaranteed identical.

    Args:
        y: Mono audio samples.
        sr: Sample rate.
        hop_length: STFT/CQT hop in samples (controls frame rate).

    Returns:
        ``(features, timestamps)`` where ``features`` is ``(n_frames, 24)`` float32
        and ``timestamps`` is ``(n_frames,)`` seconds at each frame's start.

    Raises:
        ProcessingError: if the signal is empty or features cannot be computed.
    """
    if y is None or np.size(y) == 0:
        raise ProcessingError("Audio file contains no samples.", stage="audio_load")

    try:
        bass = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop_length,
            fmin=_BASS_FMIN, n_chroma=N_CHROMA, n_octaves=_N_OCTAVES,
        )
        treble = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop_length,
            fmin=_TREBLE_FMIN, n_chroma=N_CHROMA, n_octaves=_N_OCTAVES,
        )
    except Exception as exc:  # noqa: BLE001
        raise ProcessingError(
            "Could not extract chroma features from the audio.",
            stage=_STAGE,
            original_error=exc,
        ) from exc

    # Temporal smoothing: chords change slowly, but a single frame during
    # strumming/arpeggiation rarely contains the full chord. Averaging chroma over a
    # short window makes each frame represent its harmonic neighborhood.
    bass = _moving_average(bass, smooth_window)
    treble = _moving_average(treble, smooth_window)

    # Align frame counts (both use the same hop + signal, but guard anyway).
    n_frames = min(bass.shape[1], treble.shape[1])
    bass = bass[:, :n_frames].T  # (n_frames, 12)
    treble = treble[:, :n_frames].T  # (n_frames, 12)

    if BASS_FIRST:
        features = np.concatenate([bass, treble], axis=1)
    else:
        features = np.concatenate([treble, bass], axis=1)
    features = features.astype(np.float32, copy=False)

    if features.shape[1] != FEATURE_DIM:  # defensive; should never happen
        raise ProcessingError(
            f"Expected {FEATURE_DIM}-dim features, got {features.shape[1]}.",
            stage=_STAGE,
        )

    timestamps = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length
    )
    return features, timestamps


def extract_chroma24(
    audio_path: str,
    hop_length: int = DEFAULT_HOP_LENGTH,
    smooth_window: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract approximate 24-dim Billboard-style chroma from an audio file.

    Args:
        audio_path: Path to a decodable audio file (wav/mp3/flac).
        hop_length: STFT/CQT hop in samples (controls frame rate).

    Returns:
        ``(features, timestamps, duration)`` where ``features`` is
        ``(n_frames, 24)`` float32, ``timestamps`` is ``(n_frames,)`` seconds at
        each frame's start, and ``duration`` is the clip length in seconds.

    Raises:
        ProcessingError: if the file cannot be loaded or features cannot be computed.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as exc:  # noqa: BLE001 - surface as a clean pipeline error
        raise ProcessingError(
            "Audio file could not be loaded; it may be corrupted or unsupported.",
            stage="audio_load",
            original_error=exc,
        ) from exc

    features, timestamps = chroma24_from_audio(
        y, sr, hop_length=hop_length, smooth_window=smooth_window
    )
    duration = float(librosa.get_duration(y=y, sr=sr))
    return features, timestamps, duration
