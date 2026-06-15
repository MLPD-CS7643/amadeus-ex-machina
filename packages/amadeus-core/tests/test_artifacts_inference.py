"""Round-trip artifact load + inference tests (synthetic bundle and the real bundle)."""

import numpy as np
import pytest
import soundfile as sf

from amadeus_core.artifacts import load_bundle
from amadeus_core.inference import InferenceEngine
from tests.conftest import REAL_ARTIFACTS, real_artifacts_available


def _write_sine_wav(path, sr=22050, seconds=3.0, freq=440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), y, sr)
    return str(path)


def test_save_load_bundle_round_trip(tiny_bundle_dir):
    bundle = load_bundle(str(tiny_bundle_dir))
    assert bundle.num_classes == 2
    assert bundle.input_features == 24
    assert bundle.arch == "ChromaMLP"
    assert not bundle.is_sequence


def test_engine_inference_on_synthetic_bundle(tiny_bundle_dir, tmp_path):
    engine = InferenceEngine()
    engine.load_from_dir(str(tiny_bundle_dir))
    assert engine.is_loaded

    wav = _write_sine_wav(tmp_path / "tone.wav")
    annotations, duration = engine.process_audio(wav)

    assert duration > 0
    assert len(annotations) >= 1
    # Segments tile the clip with no gaps/overlaps.
    for start, end, label in annotations:
        assert end > start
        assert isinstance(label, str)
    for (_, end_a, _), (start_b, _, _) in zip(
        annotations, annotations[1:], strict=False
    ):
        assert end_a == pytest.approx(start_b)

    lab = tmp_path / "out.lab"
    engine.save_annotations(annotations, str(lab))
    assert lab.read_text().strip() != ""


@pytest.mark.skipif(not real_artifacts_available(), reason="real artifacts not present")
def test_real_artifacts_load_and_infer(tmp_path):
    """The shipped ChromaTransformer bundle must still load and produce a .lab."""
    engine = InferenceEngine()
    engine.load_from_dir(str(REAL_ARTIFACTS))
    assert engine.is_loaded

    wav = _write_sine_wav(tmp_path / "tone.wav", seconds=4.0)
    annotations, duration = engine.process_audio(wav)
    assert duration > 0
    assert len(annotations) >= 1
