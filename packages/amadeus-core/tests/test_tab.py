"""Tests for guitar-chord voicing and ASCII tab rendering."""

import pytest

from amadeus_core.tab import (
    annotations_to_tab,
    build_chart,
    capo_open_shapes,
    chord_to_pitch_classes,
    clean_annotations,
    consolidate_root_quality,
    drop_out_of_key,
    estimate_key,
    lab_to_tab,
    parse_lab,
    suggest_capo,
    voice_chord,
)
from amadeus_core.tab.voicing import _OPEN_CHORDS, STANDARD_TUNING_PC


def _sounded_pcs(voicing) -> set[int]:
    return {
        (STANDARD_TUNING_PC[i] + f) % 12
        for i, f in enumerate(voicing.frets)
        if f is not None
    }


@pytest.mark.parametrize(
    "label,expected",
    [
        ("E:maj", (0, 2, 2, 1, 0, 0)),  # open E
        ("E:min", (0, 2, 2, 0, 0, 0)),  # open Em
        ("A:maj", (None, 0, 2, 2, 2, 0)),  # open A
        ("A:min", (None, 0, 2, 2, 1, 0)),  # open Am
        ("C:maj", (None, 3, 2, 0, 1, 0)),  # open C (idiomatic, not a barre)
        ("G:maj", (3, 2, 0, 0, 0, 3)),  # open G
        ("D:maj", (None, None, 0, 2, 3, 2)),  # open D
        ("F:maj", (1, 3, 3, 2, 1, 1)),  # E-shape barre at 1 (no open shape)
        ("B:min", (None, 2, 4, 4, 3, 2)),  # A-shape minor barre at 2
    ],
)
def test_idiomatic_shapes(label, expected):
    assert voice_chord(label).frets == expected


def test_open_shape_preferred_over_barre():
    # Open C sounds an open string and stays in the first three frets.
    frets = voice_chord("C:maj").frets
    assert 0 in frets
    assert max(f for f in frets if f is not None) <= 3


def test_barre_fallback_is_lowest_position():
    # Bb has no open shape; the A-root barre at fret 1 beats the E-root barre at 6.
    assert min(f for f in voice_chord("Bb:maj").frets if f) == 1


@pytest.mark.parametrize(
    "label,name",
    [
        ("C:maj", "C"),
        ("A:min", "Am"),
        ("G:7", "G7"),
        ("F:maj7", "Fmaj7"),
        ("D:min7", "Dm7"),
        ("B:hdim7", "Bm7b5"),
        ("Db:maj", "C#"),  # sharp spelling by default
    ],
)
def test_display_names(label, name):
    assert voice_chord(label).name == name


@pytest.mark.parametrize(
    "label,name",
    [("D#:maj", "Eb"), ("A#:min", "Bbm"), ("G#:7", "Ab7"), ("A#:maj7", "Bbmaj7")],
)
def test_flat_spelling(label, name):
    assert voice_chord(label, prefer_flat=True).name == name


def test_no_chord():
    v = voice_chord("N")
    assert v.is_no_chord
    assert v.frets == (None,) * 6
    assert v.name == "N.C."


@pytest.mark.parametrize(
    "label",
    ["C:maj", "A:min", "G:7", "D:maj7", "E:min7", "B:dim", "F:aug", "C:maj6", "D:9"],
)
def test_voicing_uses_only_chord_tones(label):
    voicing = voice_chord(label)
    _, pcs = chord_to_pitch_classes(label)
    sounded = _sounded_pcs(voicing)
    assert sounded, "voicing must sound at least one string"
    assert sounded <= pcs, "every sounded note must be a chord tone"
    assert len(sounded) >= min(3, len(pcs)), "voicing should cover the triad"


@pytest.mark.parametrize("root,quality", list(_OPEN_CHORDS))
def test_open_chords_are_chord_tone_correct(root, quality):
    voicing = voice_chord(f"{root}:{quality}")
    root_pc, pcs = chord_to_pitch_classes(f"{root}:{quality}")
    sounded = _sounded_pcs(voicing)
    assert sounded <= pcs, "open shape must only sound chord tones"
    assert root_pc in sounded, "open shape should sound the root"


def test_solver_covers_diminished_triad():
    # dim/aug have no barre template, so they exercise the solver fallback.
    _, pcs = chord_to_pitch_classes("B:dim")
    assert _sounded_pcs(voice_chord("B:dim")) == pcs


# --- annotation cleaning / key / capo ----------------------------------------------


def test_clean_annotations_drops_transients_and_merges():
    anns = [
        (0.0, 3.0, "C:maj"),
        (3.0, 3.1, "B:maj"),  # 0.1s transient -> dropped
        (3.1, 6.0, "C:maj"),  # ...leaving two C:maj that then merge
        (6.0, 9.0, "G:maj"),
    ]
    assert clean_annotations(anns) == [(0.0, 6.0, "C:maj"), (6.0, 9.0, "G:maj")]


def test_clean_annotations_never_empty():
    # All segments short: keep the single longest rather than returning nothing.
    anns = [(0.0, 0.1, "C:maj"), (0.1, 0.4, "G:maj")]
    assert clean_annotations(anns) == [(0.1, 0.4, "G:maj")]


def test_capo_suggested_for_flat_key():
    anns = [(0, 2, "Eb:maj"), (2, 4, "Ab:maj"), (4, 6, "Bb:maj")]
    assert suggest_capo(anns) == 1
    assert capo_open_shapes(anns, 1) == ["D", "G", "A"]


def test_no_capo_when_already_open():
    anns = [(0, 2, "C:maj"), (2, 4, "G:maj"), (4, 6, "A:min")]
    assert suggest_capo(anns) is None


def test_no_capo_for_single_barre_chord():
    # C, Am, G are open and F is the lone barre -> not worth capoing up 5 frets.
    anns = [(0, 2, "C:maj"), (2, 4, "A:min"), (4, 6, "F:maj"), (6, 8, "G:maj")]
    assert suggest_capo(anns) is None


# --- conservative harmonic cleanup -------------------------------------------------


def test_estimate_key_ignores_brief_outlier():
    # A 1s G# must not make the key Ab just because Ab major also contains C and G.
    anns = [(0, 1, "G#:maj"), (1, 9, "C:maj"), (9, 15, "G:maj"), (15, 21, "C:maj")]
    assert estimate_key(anns) == 0  # C, not 8 (Ab)


def test_estimate_key_prefers_fully_diatonic_major():
    # C, G, F, Am, Dm all fit C major; G major would miss the F.
    anns = [
        (0, 8, "C:maj"),
        (8, 12, "G:maj"),
        (12, 16, "F:maj"),
        (16, 18, "A:min"),
        (18, 20, "D:min"),
    ]
    assert estimate_key(anns) == 0  # C


def test_consolidate_root_quality_snaps_short_wobble():
    # A 0.6s G:min between G:maj-dominant context -> becomes G:maj.
    anns = [(0, 4, "C:maj"), (4, 4.6, "G:min"), (4.6, 9, "G:maj")]
    assert consolidate_root_quality(anns) == [(0, 4, "C:maj"), (4, 9, "G:maj")]


def test_consolidate_root_quality_folds_long_minority_stretch():
    # Same-root quality wobble collapses even when the minority quality is long: a song
    # read as E:min (26%) / E:min7 (21%) / E:maj (6%) is really just Em.
    anns = [(0, 8, "E:min"), (8, 16, "E:min7"), (16, 20, "E:maj"), (20, 26, "E:min")]
    assert consolidate_root_quality(anns) == [(0, 26, "E:min")]


def test_consolidate_root_quality_keeps_distinct_roots_and_no_chord():
    # Different roots are never merged, and N (no chord) is untouched.
    anns = [(0, 4, "E:min"), (4, 7, "A:sus4"), (7, 9, "N"), (9, 11, "A:7")]
    assert consolidate_root_quality(anns) == [
        (0, 4, "E:min"),
        (4, 7, "A:sus4"),
        (7, 9, "N"),
        (9, 11, "A:sus4"),  # A:7 (2s) folds into the A root's dominant A:sus4 (3s)
    ]


def test_drop_out_of_key_drops_rare_outlier():
    anns = [(0, 1, "G#:maj"), (1, 9, "C:maj"), (9, 15, "G:maj")]
    kept = drop_out_of_key(anns, 0)  # key = C major
    assert all(label != "G#:maj" for _, _, label in kept)


def test_drop_out_of_key_keeps_recurring_foreign_chord():
    # Bb is foreign to C major but recurs (>10% of the time) -> kept, not noise.
    anns = [
        (0, 4, "C:maj"),
        (4, 6, "Bb:maj"),
        (6, 10, "C:maj"),
        (10, 12, "Bb:maj"),
        (12, 16, "C:maj"),
        (16, 18, "Bb:maj"),
    ]
    kept = drop_out_of_key(anns, 0)  # force key = C major
    assert any(label == "Bb:maj" for _, _, label in kept)


def test_cleanup_removes_junk_but_keeps_secondary_dominants_in_render():
    tab = annotations_to_tab(
        [
            (0.0, 1.0, "G#:maj"),  # brief foreign outlier -> dropped
            (1.0, 9.0, "C:maj"),
            (9.0, 12.0, "D:maj"),  # diatonic root -> kept (secondary dominant)
            (12.0, 18.0, "G:maj"),
            (18.0, 24.0, "C:maj"),
            (24.0, 24.8, "G:min"),  # quality flicker -> snapped to G
            (24.8, 30.0, "G:maj"),
        ],
        title="song.mp3",
    )
    assert "G#" not in tab  # foreign chord dropped
    assert "Gm" not in tab  # flicker snapped to G
    prog = tab.split("is held)\n", 1)[1]
    assert "D" in prog  # secondary dominant kept
    assert "C" in prog and "G" in prog


# --- structured chart model --------------------------------------------------------


def test_build_chart_has_shapes_key_and_times():
    chart = build_chart(
        [(0, 4, "C:maj"), (4, 8, "A:min"), (8, 12, "G:maj")], title="song.mp3"
    )
    assert chart.title == "song.mp3"
    assert chart.key in {"C", "G"}
    assert [c.name for c in chart.chords] == ["C", "Am", "G"]
    first = chart.chords[0]
    assert first.frets == (None, 3, 2, 0, 1, 0)  # open C
    assert (first.start, first.end) == (0, 4)
    assert chart.duration == 12


def test_build_chart_capo_variant():
    chart = build_chart([(0, 3, "Eb:maj"), (3, 6, "Ab:maj"), (6, 9, "Bb:maj")])
    assert chart.key == "Eb"
    assert chart.capo == 1
    assert chart.capo_shapes == ("D", "G", "A")
    assert [c.name for c in chart.capo_chords] == ["D", "G", "A"]


def test_build_chart_drops_junk_like_the_ascii_chart():
    chart = build_chart(
        [(0, 1, "G#:maj"), (1, 9, "C:maj"), (9, 15, "G:maj"), (15, 21, "C:maj")]
    )
    assert [c.name for c in chart.chords] == ["C", "G", "C"]  # G# outlier gone


# --- rendering ----------------------------------------------------------------------


def test_render_has_sections_and_aligned_progression():
    tab = annotations_to_tab(
        [(0.0, 2.0, "C:maj"), (2.0, 4.0, "A:min"), (4.0, 6.0, "G:maj"), (6.0, 8.0, "N")],
        title="song.wav",
    )
    assert "Guitar chord chart" in tab
    assert "song.wav" in tab
    assert "Chords used" in tab  # chord-box diagram section
    assert "N.C." in tab  # no-chord shown in the progression

    # Within the progression block the six fret rows must all be the same width.
    prog = tab.split("is held)\n", 1)[1]
    fret_rows = [
        ln for ln in prog.splitlines() if len(ln) > 1 and ln[1] == "|" and ln.endswith("|")
    ]
    assert len(fret_rows) == 6
    assert len({len(r) for r in fret_rows}) == 1


def test_flat_key_render_uses_flats_and_capo_tip():
    tab = annotations_to_tab(
        [(0, 2, "Eb:maj"), (2, 4, "Ab:maj"), (4, 6, "Bb:maj")], title="ballad.wav"
    )
    assert "Eb" in tab and "Ab" in tab and "Bb" in tab
    assert "D#" not in tab and "G#" not in tab  # not spelled with sharps
    assert "Capo 1" in tab


def test_capo_variant_uses_open_shapes():
    tab = annotations_to_tab(
        [(0, 2, "Eb:maj"), (2, 4, "Ab:maj"), (4, 6, "Bb:maj")], title="b.wav"
    )
    # A second chart, fingered relative to the capo, follows the concert-pitch one.
    assert "Progression (capo 1)" in tab
    title = "Progression (capo 1)  (column width = how long each chord is held)\n"
    capo_prog = tab.split(title, 1)[1]
    # Eb Ab Bb become D G A shapes at the capo (frets relative to the capo).
    assert capo_prog.splitlines()[0].split() == ["D", "G", "A"]
    # Open shapes fret open strings (a "0"); the concert-pitch barres never do.
    fret_rows = capo_prog.splitlines()[1:7]
    assert any("0" in r for r in fret_rows)
    concert_prog = tab.split(
        "Progression  (column width = how long each chord is held)\n", 1
    )[1].split("\n\nCapo ", 1)[0]
    assert "0" not in concert_prog


def test_progression_width_scales_with_duration():
    # An 8s chord should occupy a much wider column than a 1s chord that follows it.
    tab = annotations_to_tab([(0.0, 8.0, "C:maj"), (8.0, 9.0, "G:maj")])
    header = tab.split("is held)\n", 1)[1].splitlines()[0]
    c_col_width = header.index("G") - header.index("C")
    assert c_col_width >= 12


def test_short_transients_absent_from_render():
    tab = annotations_to_tab(
        [(0.0, 4.0, "C:maj"), (4.0, 4.1, "F#:maj"), (4.1, 8.0, "C:maj")]
    )
    assert "F#" not in tab  # the 0.1s transient is dropped
    # The two C:maj around it merge into a single chord column.
    prog = tab.split("is held)\n", 1)[1]
    assert prog.splitlines()[0].split() == ["C"]


def test_empty_annotations():
    tab = annotations_to_tab([])
    assert "no chords detected" in tab


def test_parse_lab_and_lab_to_tab(tmp_path):
    lab = tmp_path / "chords.lab"
    lab.write_text("0.0000 2.0000 C:maj\n2.0000 4.0000 A:min\ngarbage line\n")
    assert parse_lab(lab.read_text()) == [
        (0.0, 2.0, "C:maj"),
        (2.0, 4.0, "A:min"),
    ]
    tab = lab_to_tab(lab)
    assert "e|" in tab
    assert "Am" in tab
    assert "chords.lab" in tab  # falls back to the filename as the title
