"""
Render a chord-annotation track as ASCII guitar tablature + chord diagrams.

Produces a guitar-friendly chart from a ``.lab`` chord track:

* a **Chords used** section of open/idiomatic chord-box diagrams (a mini dictionary
  of every grip in the song), and
* a **Progression** -- one voicing per chord change, laid out as six-line tab.

Before rendering, spurious sub-half-second segments are dropped and adjacent
duplicates merged, roots are spelled to match the estimated key (flats in flat keys),
and a capo is suggested when it would turn barre chords into open shapes. Timing is
implicit at chord boundaries -- this is a lead-sheet-style comping chart, not a
note-for-note transcription, because ``.lab`` carries only per-segment harmony.
"""

from collections import defaultdict
from pathlib import Path

from amadeus_core.chords.majmin import transpose_label
from amadeus_core.tab.voicing import (
    STRING_NAMES,
    GuitarVoicing,
    capo_open_shapes,
    chord_to_pitch_classes,
    suggest_capo,
    voice_chord,
)

Annotation = tuple[float, float, str]
Column = tuple[str, list[str], int]  # (display name, per-string cells low->high, width)

_DISPLAY_ORDER = (5, 4, 3, 2, 1, 0)  # top row (high e) -> bottom row (low E)
_PREFIX_WIDTH = 2  # width of the "e|" string-label prefix
_DEFAULT_WIDTH = 72
_MIN_SEGMENT_SEC = 0.5  # drop chord segments shorter than this (likely transients)

_UNIT_DASHES = 2  # sustain dashes per duration unit in the progression
_MAX_UNITS = 8  # cap so one very long chord can't swamp a whole system
_MIN_UNIT_SEC = 0.5  # a chord this short counts as one duration unit

_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_FLAT_KEY_TONICS = frozenset({5, 10, 3, 8, 1})  # F, Bb, Eb, Ab, Db -> flat spelling

# Krumhansl-Schmuckler major key profile (per scale degree, tonic first). Tonic and
# dominant weigh far more than passing degrees, so a brief foreign chord can't define
# the key just because its implied key happens to contain the real chords.
_KRUMHANSL_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)


def parse_lab(text: str) -> list[Annotation]:
    """Parse ``.lab`` text ('start end label' per line) into annotation tuples."""
    annotations: list[Annotation] = []
    for line in text.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        annotations.append((start, end, parts[2]))
    return annotations


def _merge_adjacent(annotations: list[Annotation]) -> list[Annotation]:
    """Collapse consecutive segments that share a label into one span."""
    merged: list[Annotation] = []
    for start, end, label in annotations:
        if merged and merged[-1][2] == label:
            merged[-1] = (merged[-1][0], end, label)
        else:
            merged.append((start, end, label))
    return merged


def clean_annotations(
    annotations: list[Annotation], *, min_duration: float = _MIN_SEGMENT_SEC
) -> list[Annotation]:
    """Merge repeats and drop sub-``min_duration`` transients (detection noise)."""
    merged = _merge_adjacent(annotations)
    kept = _merge_adjacent([a for a in merged if (a[1] - a[0]) >= min_duration])
    if not kept and merged:
        # Everything was short: keep the single longest so output is never empty.
        kept = [max(merged, key=lambda a: a[1] - a[0])]
    return kept


def estimate_key(annotations: list[Annotation]) -> int | None:
    """Best-fitting major-key tonic (pitch class) by duration-weighted diatonic fit."""
    weight: dict[int, float] = defaultdict(float)
    for start, end, label in annotations:
        root_pc, _ = chord_to_pitch_classes(label)
        if root_pc is None:
            continue
        weight[root_pc] += max(end - start, 0.0) or 1.0
    if not weight:
        return None
    best_tonic, best = 0, -1.0
    for tonic in range(12):
        score = sum(w * _KRUMHANSL_MAJOR[(pc - tonic) % 12] for pc, w in weight.items())
        if score > best:
            best, best_tonic = score, tonic
    return best_tonic


def estimate_prefer_flat(annotations: list[Annotation]) -> bool:
    """Guess whether to spell roots with flats, from the estimated key."""
    key = estimate_key(annotations)
    return key is not None and key in _FLAT_KEY_TONICS


def consolidate_root_quality(annotations: list[Annotation]) -> list[Annotation]:
    """Snap every segment to its root's dominant (longest-held) quality.

    A comping chart wants one grip per root, but per-frame detection wobbles between
    qualities of the same held chord -- e.g. a strummed open Em reads as Em/Em7/E across
    frames, and an Asus4 as Asus4/A7/Am7. Relabel every segment of a root to that root's
    best-supported quality so the chart shows one chord per root instead of a smear.
    Unlike a brief-flicker merge this also folds *long* minority stretches (a 16s Em7 in a
    song that is really Em). Rootless segments (``N``) and single-quality roots are left
    unchanged, and distinct roots are never merged with each other.
    """
    dur_by_label: dict[str, float] = defaultdict(float)
    for start, end, label in annotations:
        dur_by_label[label] += max(end - start, 0.0)
    dominant: dict[int, str] = {}
    for label, dur in dur_by_label.items():
        root_pc, _ = chord_to_pitch_classes(label)
        if root_pc is None:
            continue
        if root_pc not in dominant or dur > dur_by_label[dominant[root_pc]]:
            dominant[root_pc] = label

    out: list[Annotation] = []
    for start, end, label in annotations:
        root_pc, _ = chord_to_pitch_classes(label)
        out.append((start, end, dominant.get(root_pc, label) if root_pc is not None else label))
    return _merge_adjacent(out)


def drop_out_of_key(
    annotations: list[Annotation],
    key: int,
    *,
    max_outlier: float = 2.0,
    rare_fraction: float = 0.1,
) -> list[Annotation]:
    """Drop brief, rare chords whose root is foreign to ``key`` (likely mis-detections).

    Conservative: a segment is removed only if its root is non-diatonic AND it is short
    (<= ``max_outlier`` s) AND that root barely appears (< ``rare_fraction`` of the total
    time). A recurring or sustained borrowed chord is kept.
    """
    scale = {(key + d) % 12 for d in _MAJOR_SCALE}
    total = sum(max(e - s, 0.0) for s, e, _ in annotations) or 1.0
    root_total: dict[int, float] = defaultdict(float)
    for start, end, label in annotations:
        root_pc, _ = chord_to_pitch_classes(label)
        if root_pc is not None:
            root_total[root_pc] += max(end - start, 0.0)

    kept: list[Annotation] = []
    for start, end, label in annotations:
        root_pc, _ = chord_to_pitch_classes(label)
        if (
            root_pc is not None
            and root_pc not in scale
            and (end - start) <= max_outlier
            and root_total[root_pc] < rare_fraction * total
        ):
            continue
        kept.append((start, end, label))
    return _merge_adjacent(kept) if kept else list(annotations)


def _prepared(
    annotations: list[Annotation], min_duration: float
) -> tuple[list[Annotation], int | None, bool]:
    """Noise-clean + harmonic-cleanup an annotation track; return it with key context.

    Shared by the ASCII renderer and the structured chart builder so both charts are
    built from exactly the same processed chords.
    """
    cleaned = clean_annotations(annotations, min_duration=min_duration)
    key = estimate_key(cleaned)
    cleaned = consolidate_root_quality(cleaned)
    if key is not None:
        cleaned = drop_out_of_key(cleaned, key)
    prefer_flat = key is not None and key in _FLAT_KEY_TONICS
    return cleaned, key, prefer_flat


def _cells(voicing: GuitarVoicing) -> list[str]:
    return ["x" if f is None else str(f) for f in voicing.frets]


# --- Chord-box diagrams (the "Chords used" dictionary) -----------------------------


def _diagram(voicing: GuitarVoicing, cell_w: int) -> list[str]:
    """A single chord-box: name header over six ``string|-fret-|`` rows (high e top)."""
    cells = _cells(voicing)
    box_w = _PREFIX_WIDTH + (cell_w + 2) + 1  # "e|" + "-0-" + "|"
    w = max(box_w, len(voicing.name))
    lines = [voicing.name.center(w)]
    for s in _DISPLAY_ORDER:
        field = f"-{cells[s].rjust(cell_w)}-"
        lines.append(f"{STRING_NAMES[s]}|{field}|".ljust(w))
    return lines


def _render_diagrams(voicings: list[GuitarVoicing], width: int) -> list[str]:
    """Lay out one box per unique chord, wrapped into rows that fit ``width``."""
    uniq: list[GuitarVoicing] = []
    seen: set[str] = set()
    for v in voicings:
        if v.is_no_chord or v.name in seen:
            continue
        seen.add(v.name)
        uniq.append(v)
    if not uniq:
        return []

    cell_w = max((len(c) for v in uniq for c in _cells(v)), default=1)
    diagrams = [_diagram(v, cell_w) for v in uniq]
    gutter = "  "
    lines: list[str] = []
    row: list[list[str]] = []
    used = 0
    for d in diagrams:
        dw = len(d[0])
        extra = (len(gutter) if row else 0) + dw
        if row and used + extra > width:
            lines.extend(_join_boxes(row, gutter))
            lines.append("")
            row, used = [], 0
            extra = dw
        row.append(d)
        used += extra
    if row:
        lines.extend(_join_boxes(row, gutter))
    return lines


def _join_boxes(row: list[list[str]], gutter: str) -> list[str]:
    return [gutter.join(box[i] for box in row).rstrip() for i in range(7)]


# --- Progression tab (one voicing per chord change) --------------------------------


def _duration_columns(
    voicings: list[GuitarVoicing], durations: list[float]
) -> list[Column]:
    """Build tab columns whose width scales with each chord's ring time.

    The struck frets sit at the left of a column and trailing dashes stand for the
    sustain, so a chord held twice as long is drawn about twice as wide. The shortest
    chord is one duration unit; longer ones scale up to ``_MAX_UNITS``.
    """
    base = max(min(durations, default=_MIN_UNIT_SEC), _MIN_UNIT_SEC)
    columns: list[Column] = []
    for v, dur in zip(voicings, durations, strict=True):
        cells = _cells(v)
        units = min(max(round(dur / base), 1), _MAX_UNITS)
        cell_w = max(len(c) for c in cells)
        width = max(len(v.name) + 1, cell_w + units * _UNIT_DASHES)
        columns.append((v.name, cells, width))
    return columns


def _render_system(columns: list[Column]) -> list[str]:
    header = " " * _PREFIX_WIDTH + "".join(
        name.ljust(w) for name, _, w in columns
    )
    lines = [header.rstrip()]
    for s_idx in _DISPLAY_ORDER:
        row = f"{STRING_NAMES[s_idx]}|" + "".join(
            cells[s_idx].ljust(w, "-") for _, cells, w in columns
        )
        lines.append(row + "|")
    return lines


def _chunk(columns: list[Column], width: int) -> list[list[Column]]:
    """Split columns into systems that each fit within ``width`` characters."""
    avail = max(width - (_PREFIX_WIDTH + 1), 12)
    systems: list[list[Column]] = []
    current: list[Column] = []
    used = 0
    for col in columns:
        w = col[2]
        if current and used + w > avail:
            systems.append(current)
            current, used = [], 0
        current.append(col)
        used += w
    if current:
        systems.append(current)
    return systems


# --- Top-level assembly ------------------------------------------------------------


def _chart_sections(
    voicings: list[GuitarVoicing],
    durations: list[float],
    width: int,
    *,
    suffix: str = "",
) -> list[str]:
    """A chord-box "Chords used" block + a "Progression" tab block for ``voicings``."""
    sections: list[str] = []
    diagrams = _render_diagrams(voicings, width)
    if diagrams:
        sections.append(f"Chords used{suffix}\n" + "\n".join(diagrams))
    columns = _duration_columns(voicings, durations)
    systems = ["\n".join(_render_system(s)) for s in _chunk(columns, width)]
    sections.append(
        f"Progression{suffix}  (column width = how long each chord is held)\n"
        + "\n\n".join(systems)
    )
    return sections


def annotations_to_tab(
    annotations: list[Annotation],
    *,
    title: str | None = None,
    width: int = _DEFAULT_WIDTH,
    min_duration: float = _MIN_SEGMENT_SEC,
) -> str:
    """Render ``(start, end, label)`` annotations as guitar diagrams + a tab chart.

    For barre-heavy songs a suggested capo adds a second, easier-to-play chart whose
    shapes are fingered relative to the capo (open shapes wherever possible).
    """
    cleaned, key, prefer_flat = _prepared(annotations, min_duration)

    header = ["Guitar chord chart - amadeus-ex-machina"]
    if title:
        header.append(f"Source: {title}")
    header.append("Tuning: Standard (EADGBE)   Legend: numbers = fret, x = muted")

    voicings = [voice_chord(label, prefer_flat=prefer_flat) for _, _, label in cleaned]
    if not voicings:
        return "\n".join([*header, "", "(no chords detected)"]) + "\n"

    durations = [max(end - start, 0.0) for start, end, _ in cleaned]
    sections = ["\n".join(header)]
    sections += _chart_sections(voicings, durations, width)

    capo = suggest_capo(cleaned)
    if capo:
        plural = "" if capo == 1 else "s"
        note = [
            f"Capo {capo}  (finger the shapes below at the capo; "
            f"they sound {capo} semitone{plural} higher)"
        ]
        shapes = capo_open_shapes(cleaned, capo, prefer_flat=prefer_flat)
        if shapes:
            note.append(f"Open shapes: {' '.join(shapes)}")
        capo_voicings = [
            voice_chord(transpose_label(label, -capo), prefer_flat=prefer_flat)
            for _, _, label in cleaned
        ]
        sections.append("\n".join(note))
        sections += _chart_sections(
            capo_voicings, durations, width, suffix=f" (capo {capo})"
        )

    return "\n\n".join(sections) + "\n"


def lab_to_tab(
    lab_path: str | Path, *, title: str | None = None, width: int = _DEFAULT_WIDTH
) -> str:
    """Read a ``.lab`` file and render it as guitar diagrams + a tab chart."""
    path = Path(lab_path)
    return annotations_to_tab(
        parse_lab(path.read_text()),
        title=title if title is not None else path.name,
        width=width,
    )
