"""Chord vocabulary, label remapping, maj/min folding, and mir_eval scoring."""

from amadeus_core.chords.majmin import to_majmin, transpose_label
from amadeus_core.chords.remap import CullMode, remap_chord_label
from amadeus_core.chords.vocab import (
    CHORDS,
    FLAT_TO_SHARP,
    INVERSIONS,
    NOTE_INDEX,
    NOTES,
)

__all__ = [
    "CHORDS",
    "FLAT_TO_SHARP",
    "INVERSIONS",
    "NOTE_INDEX",
    "NOTES",
    "CullMode",
    "remap_chord_label",
    "to_majmin",
    "transpose_label",
]
