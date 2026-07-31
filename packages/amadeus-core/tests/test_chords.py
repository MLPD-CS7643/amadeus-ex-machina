"""Unit tests for the chord vocabulary, maj/min folding, transposition, and remapping."""

import pytest

from amadeus_core.chords import (
    CullMode,
    remap_chord_label,
    to_majmin,
    to_sevenths,
    transpose_label,
)
from amadeus_core.chords.majmin import SEVENTH_QUALITIES


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


@pytest.mark.parametrize(
    "label,expected",
    [
        ("C:maj", "C:maj"),
        ("A:min", "A:min"),
        ("G:7", "G:7"),
        ("F:maj7", "F:maj7"),
        ("D:min7", "D:min7"),
        ("C:9", "C:7"),  # dominant extensions -> 7
        ("C:maj9", "C:maj7"),  # major extensions -> maj7
        ("C:min11", "C:min7"),  # minor extensions -> min7
        ("C:minmaj7", "C:min7"),
        ("B:hdim7", "B:hdim7"),  # half-diminished kept distinct
        ("C:dim", "C:dim"),
        ("C:dim7", "C:dim"),  # fully diminished folds to dim
        ("F:aug", "F:aug"),
        ("D:sus2", "D:sus2"),
        ("E:sus4", "E:sus4"),
        ("E:sus4(b7)", "E:sus4"),
        ("A:min6", "A:min"),  # 6ths fold to the underlying triad
        ("C:maj6", "C:maj"),
        ("C:maj7/3", "C:maj7"),  # inversion dropped
        ("Db:maj", "C#:maj"),  # flats normalized to sharps
        ("N", "N"),
        ("X", "N"),
    ],
)
def test_to_sevenths(label, expected):
    assert to_sevenths(label) == expected


def test_to_sevenths_only_emits_known_qualities():
    labels = ["C:maj", "A:min7", "G:9", "B:hdim7", "F:aug", "D:sus4(b7)", "E:dim7"]
    for lab in labels:
        quality = to_sevenths(lab).split(":", 1)[1]
        assert quality in SEVENTH_QUALITIES


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
