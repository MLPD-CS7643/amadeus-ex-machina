"""Unit tests for the chord vocabulary, maj/min folding, transposition, and remapping."""

import pytest

from amadeus_core.chords import (
    CullMode,
    remap_chord_label,
    to_majmin,
    transpose_label,
)


@pytest.mark.parametrize(
    "label,expected",
    [
        ("C:maj", "C:maj"),
        ("A:min", "A:min"),
        ("D:min7", "D:min"),
        ("G:7", "G:maj"),
        ("E:dim", "E:min"),
        ("B:hdim7", "B:min"),
        ("N", "N"),
        ("X", "N"),
    ],
)
def test_to_majmin(label, expected):
    assert to_majmin(label) == expected


def test_transpose_label_wraps_octave():
    assert transpose_label("C:maj", 0) == "C:maj"
    assert transpose_label("C:maj", 2) == "D:maj"
    assert transpose_label("A:min", 3) == "C:min"  # A -> C wraps past B
    assert transpose_label("N", 5) == "N"


def test_transpose_label_normalizes_flats():
    # Bb -> A#, then up 1 semitone -> B
    assert transpose_label("Bb:maj", 1) == "B:maj"


def test_remap_bypass_normalizes_flats():
    assert remap_chord_label("Db:maj7", CullMode.BYPASS) == ("C#", "maj7")


def test_remap_no_chord():
    assert remap_chord_label("N", CullMode.CULL) == ("N", "N")
    assert remap_chord_label("X", CullMode.CULL) == ("N", "N")
