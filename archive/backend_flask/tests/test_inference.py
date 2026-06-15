"""
Tests for the feature extractor and InferenceEngine.

Pure-logic properties run without the trained artifacts; the full pipeline runs
only when the real artifacts are present (see test_integration.py).
"""
import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from app.features import FEATURE_DIM, extract_chroma24
from app.inference import InferenceEngine, majority_smooth, select_device


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
@pytest.mark.unit
class TestChromaExtraction:
    def test_feature_dim_is_24(self):
        assert FEATURE_DIM == 24

    def test_chroma_shape_matches_audio(self, chord_wav):
        # Property 3: Chromagram Extraction Consistency
        features, timestamps, duration = extract_chroma24(str(chord_wav))
        assert features.ndim == 2
        assert features.shape[1] == 24
        assert features.shape[0] == len(timestamps)
        assert features.shape[0] > 0
        assert duration == pytest.approx(2.0, abs=0.2)
        assert features.dtype == np.float32

    def test_smoothing_preserves_shape(self, chord_wav):
        # Temporal feature smoothing keeps shape and reduces frame-to-frame variance.
        raw, _, _ = extract_chroma24(str(chord_wav), smooth_window=0)
        smoothed, _, _ = extract_chroma24(str(chord_wav), smooth_window=15)
        assert smoothed.shape == raw.shape
        assert np.var(np.diff(smoothed, axis=0)) <= np.var(np.diff(raw, axis=0))

    def test_corrupted_audio_raises(self, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not really audio")
        from app.errors import ProcessingError
        with pytest.raises(ProcessingError):
            extract_chroma24(str(bad))


# ----------------------------------------------------------------------------
# Scaling / encoding properties (self-contained, no checkpoint needed)
# ----------------------------------------------------------------------------
@pytest.mark.property
class TestScalingAndEncoding:
    def test_minmax_scaling_in_unit_range(self):
        # Property 4 (adapted): the project uses MinMaxScaler -> features land in [0,1].
        rng = np.random.default_rng(0)
        train = rng.normal(size=(200, 24))
        scaler = MinMaxScaler().fit(train)
        scaled = scaler.transform(train)
        assert scaled.min() >= -1e-9
        assert scaled.max() <= 1 + 1e-9

    @given(labels=st.lists(st.sampled_from(["C:maj", "A:min", "G:7", "N"]), min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_label_encode_decode_round_trip(self, labels):
        # Property 5: Label Encoding Round Trip
        encoder = LabelEncoder().fit(["C:maj", "A:min", "G:7", "N"])
        encoded = encoder.transform(labels)
        decoded = encoder.inverse_transform(encoded)
        assert list(decoded) == labels


# ----------------------------------------------------------------------------
# Decoding / annotation logic (uses a manually-set label encoder, no model)
# ----------------------------------------------------------------------------
def _engine_with_encoder(classes):
    engine = InferenceEngine()
    engine.label_encoder = LabelEncoder().fit(classes)
    return engine


@pytest.mark.unit
class TestDecodePredictions:
    def test_merges_into_gapless_segments(self):
        # Property 6: Annotation Generation Completeness (no gaps/overlaps)
        engine = _engine_with_encoder(["A", "B", "C"])
        enc = engine.label_encoder.transform
        idx = enc(["A", "A", "B", "B", "C"])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        duration = 5.0
        ann = engine.decode_predictions(idx, timestamps, duration)
        assert ann == [(0.0, 2.0, "A"), (2.0, 4.0, "B"), (4.0, 5.0, "C")]
        # full coverage, no gaps
        assert ann[0][0] == 0.0
        assert ann[-1][1] == duration
        for prev, nxt in zip(ann, ann[1:]):
            assert prev[1] == nxt[0]

    def test_empty_predictions_yield_no_annotations(self):
        engine = _engine_with_encoder(["A"])
        assert engine.decode_predictions(np.array([], dtype=int), np.array([]), 0.0) == []

    @given(
        n=st.integers(min_value=1, max_value=40),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=100)
    def test_segments_tile_timeline(self, n, seed):
        # Property 6 (randomized): segments are contiguous, ordered, cover [0, duration]
        rng = np.random.default_rng(seed)
        engine = _engine_with_encoder(["A", "B", "C", "D"])
        labels = rng.choice(["A", "B", "C", "D"], size=n)
        idx = engine.label_encoder.transform(labels)
        timestamps = np.arange(n, dtype=float)
        duration = float(n)
        ann = engine.decode_predictions(idx, timestamps, duration)
        assert ann[0][0] == 0.0
        assert ann[-1][1] == pytest.approx(duration)
        for prev, nxt in zip(ann, ann[1:]):
            assert prev[1] == nxt[0]
            assert prev[2] != nxt[2]  # adjacent segments differ
        for start, end, _ in ann:
            assert start < end


@pytest.mark.property
class TestLabFormat:
    @given(
        rows=st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False),
            ),
            min_size=1, max_size=20,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_lab_lines_have_4_decimals(self, rows, tmp_path):
        # Property 9: Lab File Format Compliance
        engine = InferenceEngine()
        annotations = [(min(a, b), max(a, b) + 0.5, "C:maj") for a, b in rows]
        out = tmp_path / "ann.lab"
        engine.save_annotations(annotations, str(out))
        for line in out.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 3
            # exactly 4 decimal places on both timestamps
            assert len(parts[0].split(".")[1]) == 4
            assert len(parts[1].split(".")[1]) == 4
            assert float(parts[0]) < float(parts[1])


@pytest.mark.unit
class TestSmoothing:
    def test_removes_isolated_spike(self):
        out = majority_smooth(np.array([0, 0, 1, 0, 0]), window=3)
        assert list(out) == [0, 0, 0, 0, 0]

    def test_window_one_is_noop(self):
        arr = np.array([0, 1, 2, 1, 0])
        assert list(majority_smooth(arr, window=1)) == list(arr)

    def test_short_sequence_unchanged(self):
        arr = np.array([3, 7])
        assert list(majority_smooth(arr, window=9)) == list(arr)

    def test_preserves_real_transitions(self):
        # A long run of 0s then 1s should keep the block boundary intact.
        arr = np.array([0] * 6 + [1] * 6)
        out = majority_smooth(arr, window=3)
        assert out[0] == 0 and out[-1] == 1
        assert set(out.tolist()) == {0, 1}


@pytest.mark.unit
class TestDeviceSelection:
    def test_select_device_respects_availability(self):
        # Property 19: GPU Utilization When Available
        device = select_device()
        assert isinstance(device, torch.device)
        if device.type == "mps":
            assert torch.backends.mps.is_available()
        elif device.type == "cuda":
            assert torch.cuda.is_available()
        else:
            assert device.type == "cpu"

    def test_engine_not_loaded_by_default(self):
        assert InferenceEngine().is_loaded is False
