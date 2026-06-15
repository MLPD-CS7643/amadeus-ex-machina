"""
MIREX maj/min vocabulary helpers and chord transposition.

Used by training (target reduction + 12-way key augmentation). Kept in core so the
label conventions stay identical between training and any downstream consumer.
"""

from amadeus_core.chords.vocab import FLAT_TO_SHARP, NOTE_INDEX, NOTES


def to_majmin(label: str) -> str:
    """
    Reduce a chord label to the MIREX maj/min vocabulary: ``root:maj``, ``root:min``, or ``N``.

    A chord with a minor third (min/dim/half-dim families) -> min; everything else with a
    defined root -> maj. The standard, far-more-learnable target for frame-level ACR.
    """
    if label in ("N", "X", ""):
        return "N"
    root, _, quality = label.partition(":")
    q = quality.lower()
    if q.startswith("min") or q.startswith("dim") or q.startswith("hdim"):
        return f"{root}:min"
    return f"{root}:maj"


def transpose_label(label: str, k: int) -> str:
    """Transpose a chord label up by ``k`` semitones (root only; quality is invariant)."""
    if label in ("N", "X", ""):
        return "N"
    root, _, quality = label.partition(":")
    root = FLAT_TO_SHARP.get(root, root)
    if root not in NOTE_INDEX:
        return label
    new_root = NOTES[(NOTE_INDEX[root] + k) % 12]
    return f"{new_root}:{quality}" if quality else new_root
