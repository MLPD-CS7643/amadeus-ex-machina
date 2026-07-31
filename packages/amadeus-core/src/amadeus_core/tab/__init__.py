"""Guitar chord charts from chord-annotation (.lab) output: ASCII tab + structured model."""

from amadeus_core.tab.chart import Chart, ChordShape, build_chart, lab_to_chart
from amadeus_core.tab.render import (
    annotations_to_tab,
    clean_annotations,
    consolidate_root_quality,
    drop_out_of_key,
    estimate_key,
    estimate_prefer_flat,
    lab_to_tab,
    parse_lab,
)
from amadeus_core.tab.voicing import (
    STANDARD_TUNING_MIDI,
    GuitarVoicing,
    capo_open_shapes,
    chord_to_pitch_classes,
    suggest_capo,
    voice_chord,
)

__all__ = [
    "STANDARD_TUNING_MIDI",
    "Chart",
    "ChordShape",
    "GuitarVoicing",
    "annotations_to_tab",
    "build_chart",
    "capo_open_shapes",
    "chord_to_pitch_classes",
    "clean_annotations",
    "consolidate_root_quality",
    "drop_out_of_key",
    "estimate_key",
    "estimate_prefer_flat",
    "lab_to_chart",
    "lab_to_tab",
    "parse_lab",
    "suggest_capo",
    "voice_chord",
]
