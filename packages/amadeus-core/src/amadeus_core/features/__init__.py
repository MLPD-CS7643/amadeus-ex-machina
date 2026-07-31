"""Audio feature extraction (chroma)."""

from amadeus_core.features.chroma import (
    BASS_FIRST,
    DEFAULT_HOP_LENGTH,
    FEATURE_BACKENDS,
    FEATURE_DIM,
    N_CHROMA,
    chroma24_from_audio,
    extract_chroma24,
)
from amadeus_core.features.separation import SEPARATION_METHODS, separate_harmonic

__all__ = [
    "BASS_FIRST",
    "DEFAULT_HOP_LENGTH",
    "FEATURE_BACKENDS",
    "FEATURE_DIM",
    "N_CHROMA",
    "SEPARATION_METHODS",
    "chroma24_from_audio",
    "extract_chroma24",
    "separate_harmonic",
]
