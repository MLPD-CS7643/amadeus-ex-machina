"""
Chord vocabulary: the single source of truth for chord-class -> interval mappings and
note/enharmonic tables. Previously these lived in ``datagen/chordgen.py`` (and were
partially duplicated in the chord-remap and training code).

Labels are Billboard notation ``root:class`` (e.g. ``C:maj7``); ``N`` == no chord.
Each class maps to semitone intervals relative to the root.
"""

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_INDEX = {n: i for i, n in enumerate(NOTES)}

# Enharmonic normalization: flats (and a couple of odd sharps) -> the canonical sharp.
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

CHORDS = {
    "1": [0],
    "5": [0, 7],
    "maj": [0, 4, 7],
    "maj/2": [0, 4, 7, -10],
    "maj/4": [0, 4, 7, -7],
    "maj(9)": [0, 4, 7, 14],
    "min": [0, 3, 7],
    "min/2": [0, -9, 7],
    "min/4": [0, 3, -5],
    "min(9)": [0, 3, 7, 14],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "sus4(b7)": [0, 5, 7, 10],
    "sus4(9)": [0, 5, 7, 14],
    "sus4(b7,9)": [0, 5, 7, 10, 14],
    "maj6": [0, 4, 7, 9],
    "maj6(9)": [0, 4, 7, 9, 14],
    "maj7": [0, 4, 7, 11],
    "min6": [0, 3, 7, 9],
    "min7": [0, 3, 7, 10],
    "minmaj7": [0, 3, 7, 11],
    "7": [0, 4, 7, 10],
    "hdim7": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "9": [0, 4, 7, 10, 14],
    "7(#9)": [0, 4, 7, 10, 15],
    "11": [0, 4, 7, 10, 14, 17],
    "min11": [0, 3, 7, 10, 14, 17],
    "13": [0, 4, 7, 10, 14, 17, 21],
}

INVERSIONS = {
    "maj/3": [0, -8, 7],
    "maj/5": [0, 4, -5],
    "min/b3": [0, -9, 7],
    "min/5": [0, 3, -5],
    "maj7/3": [0, -8, 7, 11],
    "maj7/5": [0, 4, -5, 11],
    "maj7/7": [0, 4, 7, -1],
    "min7/b3": [0, -9, 7, 10],
    "min7/5": [0, 3, -5, 10],
    "min7/b7": [0, 3, 7, -2],
    "7/3": [0, -8, 7, 10],
    "7/5": [0, 4, -5, 10],
    "7/b7": [0, 4, 7, -2],
}
