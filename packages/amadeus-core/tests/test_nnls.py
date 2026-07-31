"""
Tests for the optional NNLS feature backend (true McGill bothchroma at serve time).

The plugin-dependent tests skip unless the native backend is set up (``just nnls-setup``),
so the default CI/test run stays green without it. The backend-selection and engine
parity-guard tests need no plugin and always run.
"""

import numpy as np
import pytest
import soundfile as sf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from amadeus_core.artifacts import save_bundle
from amadeus_core.errors import ProcessingError
from amadeus_core.features import extract_chroma24
from amadeus_core.features.chroma import chroma24_from_audio
from amadeus_core.features.nnls import nnls_bothchroma
from amadeus_core.inference import InferenceEngine
from amadeus_core.models.chroma_mlp import ChromaMLP

# Note order of the NNLS Chroma bothchroma bins (starts at A, not C).
_NNLS_NOTES = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]


def _nnls_available() -> bool:
    try:
        import vampyhost

        return "nnls-chroma:nnls-chroma" in vampyhost.list_plugins()
    except Exception:
        return False


nnls_required = pytest.mark.skipif(
    not _nnls_available(),
    reason="NNLS Vamp backend not installed (run `just nnls-setup`)",
)


def _triad(sr: int, sec: float = 3.0) -> np.ndarray:
    """C major triad (C4/E4/G4) plus a low C2 to exercise the bass chroma."""
    t = np.arange(int(sr * sec)) / sr
    y = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0, 65.41))
    y = np.asarray(y, dtype=np.float32)
    return (y / np.max(np.abs(y))).astype(np.float32)


def _nnls_bundle_dir(tmp_path, *, feature_backend="nnls", feat_smooth=15):
    """A minimal on-disk bundle whose config selects the given feature backend."""
    rng = np.random.default_rng(0)
    x = rng.random((32, 24)).astype(np.float32)
    labels = np.array(["C:maj", "A:min"] * 16)
    scaler = MinMaxScaler().fit(x)
    encoder = LabelEncoder().fit(labels)
    model = ChromaMLP(input_features=24, num_classes=2, hidden_sizes=(16,))
    model.eval()
    config = {
        "model_class": "ChromaMLP",
        "input_features": 24,
        "num_classes": 2,
        "hidden_sizes": [16],
        "dropout": 0.3,
        "hop_length": 2048,
        "feat_smooth": feat_smooth,
        "feature_backend": feature_backend,
    }
    save_bundle(str(tmp_path), model.state_dict(), scaler, encoder, config)
    return tmp_path


# -- backend selection / validation (no plugin needed) ----------------------


def test_unknown_backend_raises():
    y = _triad(22050, sec=0.5)
    with pytest.raises(ProcessingError, match="Unknown feature backend"):
        chroma24_from_audio(y, 22050, backend="bogus")


def test_nnls_empty_signal_raises():
    with pytest.raises(ProcessingError):
        nnls_bothchroma(np.array([], dtype=np.float32), 44100)


def test_nnls_missing_dependency_is_reported_clearly():
    """When the backend isn't installed, callers get a clean, actionable error."""
    if _nnls_available():
        pytest.skip("NNLS backend is installed")
    with pytest.raises(ProcessingError, match="NNLS feature backend unavailable"):
        nnls_bothchroma(_triad(44100, sec=0.5), 44100)


# -- engine self-describing bundle + parity guard (no plugin needed) --------


def test_engine_nnls_bundle_forces_separation_and_smoothing_off(tmp_path):
    """
    An nnls bundle must serve with separation off and no smoothing, even if the API was
    configured with separation='hpss' and the bundle carries a stray feat_smooth.
    """
    engine = InferenceEngine(separation="hpss")
    engine.load_from_dir(str(_nnls_bundle_dir(tmp_path, feat_smooth=15)))
    assert engine.feature_backend == "nnls"
    assert engine.separation == "none"
    assert engine.feat_smooth == 0


def test_engine_librosa_bundle_keeps_configured_separation(tmp_path):
    engine = InferenceEngine(separation="hpss")
    engine.load_from_dir(str(_nnls_bundle_dir(tmp_path, feature_backend="librosa")))
    assert engine.feature_backend == "librosa"
    assert engine.separation == "hpss"


def test_engine_backend_override_wins(tmp_path):
    """An explicit constructor backend overrides the bundle's declared backend."""
    engine = InferenceEngine(separation="hpss", feature_backend="librosa")
    engine.load_from_dir(str(_nnls_bundle_dir(tmp_path, feature_backend="nnls")))
    assert engine.feature_backend == "librosa"


# -- true bothchroma parity (needs the plugin) ------------------------------


@nnls_required
def test_nnls_bothchroma_shape_framing_and_scale():
    sr = 22050  # deliberately not 44100: exercises the internal resample-to-44100
    feats, ts = nnls_bothchroma(_triad(sr), sr)
    assert feats.ndim == 2 and feats.shape[1] == 24
    assert feats.dtype == np.float32
    # Framing is the plugin's own: stepSize 2048 @ 44100 Hz.
    assert ts.shape[0] == feats.shape[0]
    if len(ts) > 1:
        assert float(ts[1] - ts[0]) == pytest.approx(2048 / 44100, abs=1e-6)
    # Un-normalized NNLS magnitudes, NOT the 0-1 range librosa chroma_cqt returns.
    assert feats.max() > 1.0


@nnls_required
def test_nnls_bothchroma_bass_first_and_pitch_content():
    feats, _ = nnls_bothchroma(_triad(44100), 44100)
    mid = feats[len(feats) // 3 : 2 * len(feats) // 3].mean(axis=0)
    bass, treble = mid[:12], mid[12:]
    # Treble chroma should peak on the triad tones C, E, G.
    top3 = {_NNLS_NOTES[i] for i in treble.argsort()[::-1][:3]}
    assert {"C", "E", "G"} == top3
    # The only strong bass note is the C2 fundamental.
    assert _NNLS_NOTES[int(bass.argmax())] == "C"


@nnls_required
def test_extract_chroma24_nnls_backend_from_file(tmp_path):
    sr = 44100
    path = tmp_path / "clip.wav"
    sf.write(str(path), _triad(sr, sec=2.0), sr, subtype="PCM_16")
    feats, ts, duration = extract_chroma24(str(path), backend="nnls")
    assert feats.shape[1] == 24
    assert feats.max() > 1.0
    assert duration == pytest.approx(2.0, abs=0.1)
