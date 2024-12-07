from enum import Enum
import re
from datagen.chordgen import CHORDS, INVERSIONS


class CullMode(Enum):
    LEAVE = 0,
    CULL = 1,
    REMAP = 2

FLAT_TO_SHARP = {
    "Cb": "B",
    "Db": "C#",
    "Eb": "D#",
    "Fb": "E",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "B#": "C",
    "E#": "F",
}

REMAP = {
    "maj/7": "maj7/7",
    "maj/b7": "7/b7",
    "min/b7": "min7/b7",
    "maj/6": "maj6",
    "min/6": "min6",
    "sus4/b7": "sus4(b7)",
}

def remap_chord_label(billboard_chord:str, cull_mode:CullMode):
    if billboard_chord == 'X' or billboard_chord == 'N':
        return ('N', 'N')
    root, chord_class = billboard_chord.split(':', 1)
    if chord_class[0] == '(': # get rid of non-sense annotations
        return ('N', 'N')
    if root in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[root]
    if chord_class not in CHORDS and chord_class not in INVERSIONS:
        if chord_class in REMAP:
            chord_class = REMAP[chord_class]
        else:
            chord_class_add_removed = re.sub(r'\(.*?\)', '', chord_class)
            if chord_class_add_removed in REMAP:
                chord_class = REMAP[chord_class_add_removed]
            else:
                chord_class_inv_removed = chord_class.split('/', 1)[0] # get rid of unaccounted for inversions
                if chord_class not in CHORDS and chord_class not in INVERSIONS:
                    match cull_mode:
                        case CullMode.LEAVE:
                            return (root, chord_class_inv_removed)
                        case CullMode.CULL:
                            return ('N', 'N')
                        case CullMode.REMAP:
                            chord_class = chord_class_inv_removed.split('(', 1)[0]
    return (root, chord_class)