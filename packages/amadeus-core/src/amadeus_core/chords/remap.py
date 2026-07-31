"""
Map raw Billboard/SALAMI chord labels onto the chordgen vocabulary.

``remap_chord_label`` normalizes flats->sharps and handles inversions/extensions;
``CullMode`` controls what happens to out-of-vocabulary labels.
"""

import re
from enum import Enum

from amadeus_core.chords.vocab import CHORDS, FLAT_TO_SHARP, INVERSIONS


class CullMode(Enum):
    LEAVE = 0
    CULL = 1
    REMAP = 2
    BYPASS = 3


REMAP = {
    "1/1": "1",
    "maj/7": "maj7/7",
    "maj/b7": "7/b7",
    "min/b7": "min7/b7",
    "maj/6": "maj6",
    "min/6": "min6",
    "sus4/b7": "sus4(b7)",
}


def remap_chord_label(billboard_chord: str, cull_mode: CullMode) -> tuple[str, str]:
    if billboard_chord == "X" or billboard_chord == "N":
        return ("N", "N")
    root, chord_class = billboard_chord.split(":", 1)
    if chord_class[0] == "(":  # get rid of non-sense annotations
        return ("N", "N")
    if chord_class == "1/1":
        chord_class = "1"
    if root in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[root]
    if cull_mode == CullMode.BYPASS:
        return (root, chord_class)
    if chord_class not in CHORDS and chord_class not in INVERSIONS:
        if chord_class in REMAP:
            chord_class = REMAP[chord_class]
        else:
            chord_class_add_removed = re.sub(r"\(.*?\)", "", chord_class)
            if chord_class_add_removed in REMAP:
                chord_class = REMAP[chord_class_add_removed]
            else:
                chord_class_inv_removed = chord_class.split("/", 1)[
                    0
                ]  # drop unaccounted inversions
                if chord_class not in CHORDS and chord_class not in INVERSIONS:
                    match cull_mode:
                        case CullMode.LEAVE:
                            return (root, chord_class_inv_removed)
                        case CullMode.CULL:
                            return ("N", "N")
                        case CullMode.REMAP:
                            chord_class = chord_class_inv_removed.split("(", 1)[0]
    return (root, chord_class)
