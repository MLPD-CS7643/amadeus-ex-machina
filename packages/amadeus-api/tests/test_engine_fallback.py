"""
Startup fallback: when the deployed model needs the NNLS feature backend but it isn't
installed, the app must fall back to the librosa model instead of booting broken.
"""

import types

import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from amadeus_api.main import _load_engine
from amadeus_core.artifacts import save_bundle
from amadeus_core.models.chroma_mlp import ChromaMLP


def _make_bundle(path, *, num_classes, feature_backend):
    rng = np.random.default_rng(0)
    x = rng.random((32, 24)).astype(np.float32)
    labels = np.array([f"c{i % num_classes}" for i in range(32)])
    scaler = MinMaxScaler().fit(x)
    encoder = LabelEncoder().fit(labels)
    model = ChromaMLP(input_features=24, num_classes=num_classes, hidden_sizes=(16,))
    model.eval()
    save_bundle(
        str(path),
        model.state_dict(),
        scaler,
        encoder,
        {
            "model_class": "ChromaMLP",
            "input_features": 24,
            "num_classes": num_classes,
            "hidden_sizes": [16],
            "dropout": 0.3,
            "feature_backend": feature_backend,
        },
    )
    return path


def _config(tmp_path):
    primary = _make_bundle(
        tmp_path / "primary", num_classes=7, feature_backend="nnls"
    )
    fallback = _make_bundle(
        tmp_path / "fallback", num_classes=3, feature_backend="librosa"
    )
    return types.SimpleNamespace(
        MODEL_PATH=str(primary / "model.pt"),
        SCALER_PATH=str(primary / "scaler.pkl"),
        ENCODER_PATH=str(primary / "label_encoder.pkl"),
        MODEL_CONFIG_PATH=str(primary / "model_config.json"),
        FALLBACK_MODEL_PATH=str(fallback / "model.pt"),
        FALLBACK_SCALER_PATH=str(fallback / "scaler.pkl"),
        FALLBACK_ENCODER_PATH=str(fallback / "label_encoder.pkl"),
        FALLBACK_MODEL_CONFIG_PATH=str(fallback / "model_config.json"),
        SEPARATION_METHOD="hpss",
        FEATURE_BACKEND=None,
    )


def test_falls_back_to_librosa_when_nnls_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "amadeus_core.features.nnls.nnls_backend_available", lambda: False
    )
    engine = _load_engine(_config(tmp_path))
    # The librosa fallback bundle (3 classes) is served, not the nnls primary (7 classes).
    assert engine.is_loaded
    assert engine.feature_backend == "librosa"
    assert engine.bundle is not None and engine.bundle.num_classes == 3


def test_serves_nnls_model_when_backend_available(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "amadeus_core.features.nnls.nnls_backend_available", lambda: True
    )
    engine = _load_engine(_config(tmp_path))
    # The nnls primary bundle (7 classes) is served; parity guard forces separation off.
    assert engine.feature_backend == "nnls"
    assert engine.separation == "none"
    assert engine.bundle is not None and engine.bundle.num_classes == 7
