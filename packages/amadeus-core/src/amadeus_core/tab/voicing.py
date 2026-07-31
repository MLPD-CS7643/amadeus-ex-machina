"""
Deterministic guitar-chord voicing: chord label -> fretboard shape.

Maps a Billboard chord label (e.g. ``C:maj``, ``A:min7``) to a playable six-string
voicing in standard tuning. Common qualities use idiomatic movable barre/open shapes
(the CAGED E- and A-root forms); anything outside that set falls back to a fret-search
solver over the chord's pitch classes, so the full training vocabulary still yields a
playable shape. Pitch-class content comes from :mod:`amadeus_core.chords.vocab` -- the
single source of truth -- so voicings stay consistent with the rest of the pipeline.
"""

import re
from dataclasses import dataclass

from amadeus_core.chords.vocab import CHORDS, FLAT_TO_SHARP, INVERSIONS, NOTE_INDEX

# Standard tuning, low E (6th string) -> high e (1st string).
STANDARD_TUNING_MIDI: tuple[int, ...] = (40, 45, 50, 55, 59, 64)
STANDARD_TUNING_PC: tuple[int, ...] = tuple(m % 12 for m in STANDARD_TUNING_MIDI)
STRING_NAMES: tuple[str, ...] = ("E", "A", "D", "G", "B", "e")  # low -> high

MAX_FRET = 15
MAX_SPAN = 4  # playable hand span, in frets

NO_CHORD_LABELS = frozenset({"N", "X", ""})

# Enharmonic root spellings. Sharps are the canonical/internal form; flats are used
# for display in flat keys (a guitarist reads "Eb", not "D#"). See ``_root_name``.
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


def _root_name(root_pc: int, prefer_flat: bool) -> str:
    return (_FLAT_NAMES if prefer_flat else _SHARP_NAMES)[root_pc]

# Movable shapes keyed by normalized quality. Each entry is a candidate form
# ``(root_string_index, fret_offsets)`` with offsets ordered low E -> high e; ``None``
# marks a muted string and ints are frets *relative to the barre*. The barre fret is
# chosen so the chord root lands on the root string (see :func:`_place_shape`).
_SHAPES: dict[str, list[tuple[int, tuple[int | None, ...]]]] = {
    "maj": [(0, (0, 2, 2, 1, 0, 0)), (1, (None, 0, 2, 2, 2, 0))],
    "min": [(0, (0, 2, 2, 0, 0, 0)), (1, (None, 0, 2, 2, 1, 0))],
    "7": [(0, (0, 2, 0, 1, 0, 0)), (1, (None, 0, 2, 0, 2, 0))],
    "maj7": [(0, (0, 2, 1, 1, 0, 0)), (1, (None, 0, 2, 1, 2, 0))],
    "min7": [(0, (0, 2, 0, 0, 0, 0)), (1, (None, 0, 2, 0, 1, 0))],
    "sus4": [(0, (0, 2, 2, 2, 0, 0)), (1, (None, 0, 2, 2, 3, 0))],
    "sus2": [(1, (None, 0, 2, 2, 0, 0))],
    "5": [(0, (0, 2, 2, None, None, None)), (1, (None, 0, 2, 2, None, None))],
    "1": [
        (0, (0, None, None, None, None, None)),
        (1, (None, 0, None, None, None, None)),
    ],
}

# Idiomatic open-position voicings, keyed by ``(canonical_root, shape_quality)``.
# These are how these chords are actually fingered, so they beat the movable barre
# templates whenever one exists. Frets low E -> high e; ``None`` = muted string.
_OPEN_CHORDS: dict[tuple[str, str], tuple[int | None, ...]] = {
    ("C", "maj"): (None, 3, 2, 0, 1, 0),
    ("A", "maj"): (None, 0, 2, 2, 2, 0),
    ("G", "maj"): (3, 2, 0, 0, 0, 3),
    ("E", "maj"): (0, 2, 2, 1, 0, 0),
    ("D", "maj"): (None, None, 0, 2, 3, 2),
    ("A", "min"): (None, 0, 2, 2, 1, 0),
    ("E", "min"): (0, 2, 2, 0, 0, 0),
    ("D", "min"): (None, None, 0, 2, 3, 1),
    ("C", "7"): (None, 3, 2, 3, 1, 0),
    ("D", "7"): (None, None, 0, 2, 1, 2),
    ("E", "7"): (0, 2, 0, 1, 0, 0),
    ("G", "7"): (3, 2, 0, 0, 0, 1),
    ("A", "7"): (None, 0, 2, 0, 2, 0),
    ("B", "7"): (None, 2, 1, 2, 0, 2),
    ("C", "maj7"): (None, 3, 2, 0, 0, 0),
    ("D", "maj7"): (None, None, 0, 2, 2, 2),
    ("E", "maj7"): (0, 2, 1, 1, 0, 0),
    ("F", "maj7"): (None, None, 3, 2, 1, 0),
    ("G", "maj7"): (3, 2, 0, 0, 0, 2),
    ("A", "maj7"): (None, 0, 2, 1, 2, 0),
    ("A", "min7"): (None, 0, 2, 0, 1, 0),
    ("D", "min7"): (None, None, 0, 2, 1, 1),
    ("E", "min7"): (0, 2, 0, 0, 0, 0),
}

# Root pitch classes that have an open shape, per shape quality -- used by the capo
# suggester to see which chords fall into open-shape range at a given capo.
_open_roots: dict[str, set[int]] = {}
for _root, _key in _OPEN_CHORDS:
    _open_roots.setdefault(_key, set()).add(NOTE_INDEX[_root])
_OPEN_ROOT_PCS: dict[str, frozenset[int]] = {
    k: frozenset(v) for k, v in _open_roots.items()
}

# Chord-name suffix for display (e.g. ``A:min`` -> ``Am``, ``B:hdim7`` -> ``Bm7b5``).
_QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "5": "5",
    "1": "",
    "maj6": "6",
    "min6": "m6",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "minmaj7": "mM7",
    "9": "9",
    "maj9": "maj9",
    "min9": "m9",
    "11": "11",
    "min11": "m11",
    "13": "13",
}


@dataclass(frozen=True)
class GuitarVoicing:
    """A six-string voicing, low E -> high e. ``None`` frets are muted strings."""

    label: str
    name: str
    frets: tuple[int | None, ...]
    is_no_chord: bool = False


def _normalize_root(root: str) -> str | None:
    root = FLAT_TO_SHARP.get(root, root)
    return root if root in NOTE_INDEX else None


def _split_label(label: str) -> tuple[str | None, str]:
    """Return ``(canonical_root, quality)``; root is ``None`` for a no-chord label."""
    if label in NO_CHORD_LABELS:
        return None, ""
    root, sep, quality = label.partition(":")
    canonical = _normalize_root(root)
    if canonical is None:
        return None, ""
    return canonical, (quality if sep else "maj")


def _intervals_for(quality: str) -> list[int]:
    """Semitone intervals for a quality, tolerating inversions/added-tone spellings."""
    intervals = CHORDS.get(quality) or INVERSIONS.get(quality)
    if intervals is None:
        base = quality.split("/", 1)[0]
        intervals = CHORDS.get(base) or INVERSIONS.get(base)
    if intervals is None:
        stripped = re.sub(r"\(.*?\)", "", quality.split("/", 1)[0])
        intervals = CHORDS.get(stripped)
    return list(intervals) if intervals is not None else [0, 4, 7]


def chord_to_pitch_classes(label: str) -> tuple[int | None, frozenset[int]]:
    """``(root_pitch_class, {chord-tone pitch classes})`` for a chord label."""
    root, quality = _split_label(label)
    if root is None:
        return None, frozenset()
    root_pc = NOTE_INDEX[root]
    return root_pc, frozenset((root_pc + iv) % 12 for iv in _intervals_for(quality))


def _shape_key(quality: str) -> str | None:
    """Normalize a quality to a movable-shape key, or ``None`` to use the solver."""
    b = quality.split("/", 1)[0]  # drop inversion suffix; the shape is root-position
    if b.startswith("maj7") or b == "maj9":
        return "maj7"
    if b.startswith("minmaj7"):
        return None
    if b.startswith("min7") or b in ("min9", "min11"):
        return "min7"
    if b.startswith("maj6") or b.startswith("min6"):
        return None  # 6ths: let the solver keep the added tone
    if b in ("7", "9", "11", "13") or b.startswith("7("):
        return "7"
    if b.startswith("sus2"):
        return "sus2"
    if b.startswith("sus4"):
        return "sus4"
    if b in ("dim", "dim7", "hdim7", "aug"):
        return None  # awkward as barre shapes; solve on exact tones
    if b == "5":
        return "5"
    if b == "1":
        return "1"
    if b.startswith("min"):
        return "min"
    if b.startswith("maj"):
        return "maj"
    return None


def _display_name(root: str, quality: str) -> str:
    base = quality.split("/", 1)[0]
    suffix = _QUALITY_SUFFIX.get(base)
    if suffix is None:
        suffix = _QUALITY_SUFFIX.get(re.sub(r"\(.*?\)", "", base))
    if suffix is None:
        suffix = base  # unknown quality: show it verbatim
    return f"{root}{suffix}"


def _place_shape(
    root_pc: int, shapes: list[tuple[int, tuple[int | None, ...]]]
) -> tuple[int | None, ...] | None:
    """Place a movable shape at the lowest playable position for ``root_pc``."""
    best: tuple[int | None, ...] | None = None
    best_score: tuple[int, int] | None = None
    for root_string, offsets in shapes:
        barre = (root_pc - STANDARD_TUNING_PC[root_string]) % 12
        frets = tuple((barre + off) if off is not None else None for off in offsets)
        if any(f is not None and f > MAX_FRET for f in frets):
            continue
        score = (
            barre,
            root_string,
        )  # prefer lower position, then the fuller low-E form
        if best_score is None or score < best_score:
            best_score, best = score, frets
    return best


def _solve(pcs: frozenset[int], bass_pc: int | None = None) -> tuple[int | None, ...]:
    """Search for a playable voicing covering ``pcs`` (fallback for exotic chords)."""
    if not pcs:
        return (None,) * 6
    best: tuple[int | None, ...] | None = None
    best_score: tuple[int, int, int, int] | None = None
    for base in range(0, MAX_FRET - MAX_SPAN + 1):
        frets: list[int | None] = []
        sounded: set[int] = set()
        for s in range(6):
            open_pc = STANDARD_TUNING_PC[s]
            # Only allow open strings in open position, so a voicing never mixes an
            # open string with a high fretted note (unplayable as one shape).
            candidates = [0] if (base == 0 and open_pc in pcs) else []
            for f in range(max(base, 1), base + MAX_SPAN + 1):
                if f > MAX_FRET:
                    break
                if (open_pc + f) % 12 in pcs:
                    candidates.append(f)
            fresh = [f for f in candidates if (open_pc + f) % 12 not in sounded]
            chosen = min(fresh) if fresh else (min(candidates) if candidates else None)
            frets.append(chosen)
            if chosen is not None:
                sounded.add((open_pc + chosen) % 12)

        low_pc: int | None = next(
            (
                (STANDARD_TUNING_PC[i] + f) % 12
                for i, f in enumerate(frets)
                if f is not None
            ),
            None,
        )
        score = (
            len(sounded & pcs),  # coverage of chord tones
            sum(1 for f in frets if f is not None),  # strings sounded
            int(bass_pc is not None and low_pc == bass_pc),  # root in the bass
            -base,  # prefer lower position
        )
        if best_score is None or score > best_score:
            best_score, best = score, tuple(frets)
    return best if best is not None else (None,) * 6


def voice_chord(label: str, *, prefer_flat: bool = False) -> GuitarVoicing:
    """Map a chord label to a playable standard-tuning guitar voicing.

    Prefers an idiomatic open-position shape when one exists, then a movable barre
    template, then a fret-search solve. ``prefer_flat`` only affects the display name
    (e.g. ``A#`` vs ``Bb``); the fingering is identical.
    """
    root, quality = _split_label(label)
    if root is None:
        return GuitarVoicing(
            label=label, name="N.C.", frets=(None,) * 6, is_no_chord=True
        )
    root_pc = NOTE_INDEX[root]
    key = _shape_key(quality)
    frets: tuple[int | None, ...] | None = None
    if key is not None:
        frets = _OPEN_CHORDS.get((root, key))
        if frets is None and key in _SHAPES:
            frets = _place_shape(root_pc, _SHAPES[key])
    if frets is None:
        _, pcs = chord_to_pitch_classes(label)
        frets = _solve(pcs, bass_pc=root_pc)
    name = _display_name(_root_name(root_pc, prefer_flat), quality)
    return GuitarVoicing(label=label, name=name, frets=frets)


def suggest_capo(
    annotations: list[tuple[float, float, str]], *, max_capo: int = 7
) -> int | None:
    """Suggest a capo fret that brings more chords into open-shape range, or ``None``.

    Scores each capo by the total duration of chords that become playable as open
    shapes when fingered at that position. Only suggests a capo for genuinely
    barre-heavy songs (chords are open less than half the time at the nut) -- there's
    no point capoing up several frets just to avoid a single easy barre chord.
    """
    weight: dict[tuple[int, str], float] = {}
    for start, end, label in annotations:
        root, quality = _split_label(label)
        if root is None:
            continue
        key = _shape_key(quality)
        if key is None or key not in _OPEN_ROOT_PCS:
            continue
        dur = max(end - start, 0.0) or 1.0
        weight[(NOTE_INDEX[root], key)] = weight.get((NOTE_INDEX[root], key), 0.0) + dur
    if not weight:
        return None

    def coverage(capo: int) -> float:
        return sum(
            w
            for (pc, key), w in weight.items()
            if ((pc - capo) % 12) in _OPEN_ROOT_PCS[key]
        )

    total = sum(weight.values())
    base = coverage(0)
    if base >= 0.5 * total:  # already mostly open at the nut; a capo isn't worth it
        return None
    best_capo, best = 0, base
    for capo in range(1, max_capo + 1):
        cov = coverage(capo)
        if cov > best:
            best, best_capo = cov, capo
    return best_capo or None


def capo_open_shapes(
    annotations: list[tuple[float, float, str]],
    capo: int,
    *,
    prefer_flat: bool = False,
) -> list[str]:
    """Names of the open shapes a player fingers (at ``capo``) for the open chords."""
    seen: list[str] = []
    for _start, _end, label in annotations:
        root, quality = _split_label(label)
        if root is None:
            continue
        key = _shape_key(quality)
        roots = _OPEN_ROOT_PCS.get(key) if key is not None else None
        if key is None or roots is None:
            continue
        shape_pc = (NOTE_INDEX[root] - capo) % 12
        if shape_pc not in roots:
            continue
        name = _display_name(_root_name(shape_pc, prefer_flat), key)
        if name not in seen:
            seen.append(name)
    return seen
