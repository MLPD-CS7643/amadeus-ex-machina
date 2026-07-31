"""
Structured chord-chart model: the same processed data the ASCII renderer uses, but
as objects instead of text. This is the single source for richer front-end rendering
(SVG chord diagrams, an audio-synced timeline) -- the ASCII tab becomes just one view.
"""

from dataclasses import dataclass
from pathlib import Path

from amadeus_core.chords.majmin import transpose_label
from amadeus_core.tab.render import Annotation, _prepared, parse_lab
from amadeus_core.tab.voicing import (
    GuitarVoicing,
    _root_name,
    capo_open_shapes,
    suggest_capo,
    voice_chord,
)

_MIN_SEGMENT_SEC = 0.5


@dataclass(frozen=True)
class ChordShape:
    """One chord in the chart: a display name + a fretboard shape over a time span."""

    name: str
    label: str  # original Billboard label (e.g. "C:maj")
    frets: tuple[int | None, ...]  # low E -> high e; None = muted string
    start: float
    end: float
    is_no_chord: bool = False


@dataclass(frozen=True)
class Chart:
    """A full chord chart: concert-pitch shapes plus an optional capo variant."""

    title: str | None
    key: str | None  # spelled tonic, e.g. "G" or "Eb"
    prefer_flat: bool
    capo: int | None
    capo_shapes: tuple[str, ...]  # the open shape names fingered under the capo
    duration: float
    chords: tuple[ChordShape, ...]
    capo_chords: tuple[ChordShape, ...]  # same timing, fingered at the capo (or empty)


def _shape(voicing: GuitarVoicing, start: float, end: float) -> ChordShape:
    return ChordShape(
        name=voicing.name,
        label=voicing.label,
        frets=voicing.frets,
        start=start,
        end=end,
        is_no_chord=voicing.is_no_chord,
    )


def build_chart(
    annotations: list[Annotation],
    *,
    title: str | None = None,
    min_duration: float = _MIN_SEGMENT_SEC,
) -> Chart:
    """Turn a chord-annotation track into a structured :class:`Chart`."""
    cleaned, key, prefer_flat = _prepared(annotations, min_duration)
    duration = max((end for _, end, _ in cleaned), default=0.0)

    chords = tuple(
        _shape(voice_chord(label, prefer_flat=prefer_flat), start, end)
        for start, end, label in cleaned
    )

    capo = suggest_capo(cleaned)
    capo_shapes: tuple[str, ...] = ()
    capo_chords: tuple[ChordShape, ...] = ()
    if capo:
        capo_shapes = tuple(capo_open_shapes(cleaned, capo, prefer_flat=prefer_flat))
        capo_chords = tuple(
            _shape(
                voice_chord(transpose_label(label, -capo), prefer_flat=prefer_flat),
                start,
                end,
            )
            for start, end, label in cleaned
        )

    return Chart(
        title=title,
        key=_root_name(key, prefer_flat) if key is not None else None,
        prefer_flat=prefer_flat,
        capo=capo,
        capo_shapes=capo_shapes,
        duration=duration,
        chords=chords,
        capo_chords=capo_chords,
    )


def lab_to_chart(lab_path: str | Path, *, title: str | None = None) -> Chart:
    """Read a ``.lab`` file and build a structured :class:`Chart`."""
    path = Path(lab_path)
    return build_chart(
        parse_lab(path.read_text()),
        title=title if title is not None else path.name,
    )
