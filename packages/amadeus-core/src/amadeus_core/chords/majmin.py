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


SEVENTH_QUALITIES = (
    "maj",
    "min",
    "7",
    "maj7",
    "min7",
    "dim",
    "aug",
    "sus2",
    "sus4",
    "hdim7",
)


def _seventh_bucket(quality: str) -> str:
    """Fold a raw chord quality onto the nearest 'sevenths' bucket."""
    q = quality.split("/", 1)[0].lower()  # drop inversion, normalize case
    if q.startswith(("maj7", "maj9", "maj11", "maj13")):
        return "maj7"
    if q.startswith(("minmaj7", "min7", "min9", "min11", "min13")):
        return "min7"
    if q.startswith("hdim7"):
        return "hdim7"
    if q.startswith("dim"):  # dim, dim7 -> dim
        return "dim"
    if q.startswith("aug"):
        return "aug"
    if q.startswith("sus2"):
        return "sus2"
    if q.startswith("sus"):  # sus4, sus4(b7), sus -> sus4
        return "sus4"
    if q.startswith("min"):  # min, min6 -> min
        return "min"
    if q.startswith(("7", "9", "11", "13")):  # dominant family -> 7
        return "7"
    return "maj"  # maj, maj6, 6, 5, 1, "", unknown -> maj


def to_sevenths(label: str) -> str:
    """
    Reduce a chord label to a curated ~10-quality "sevenths" vocabulary (+ ``N``).

    Keeps the harmonic colour a songwriter cares about (7 / maj7 / min7 / sus / dim /
    aug) while folding the long tail of extensions and inversions onto the nearest
    bucket. Roots are normalized flats->sharps so enharmonics share a class.
    """
    if label in ("N", "X", ""):
        return "N"
    root, _, quality = label.partition(":")
    root = FLAT_TO_SHARP.get(root, root)
    if root not in NOTE_INDEX:
        return "N"
    return f"{root}:{_seventh_bucket(quality)}"


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
