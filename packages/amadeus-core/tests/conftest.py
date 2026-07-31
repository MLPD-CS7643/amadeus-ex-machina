"""Shared fixtures for amadeus-core tests."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from amadeus_core.artifacts import save_bundle
from amadeus_core.models.chroma_mlp import ChromaMLP

# packages/amadeus-core/tests/conftest.py -> repo root is parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_ARTIFACTS = REPO_ROOT / "artifacts" / "guitarset_chroma"


@pytest.fixture
def tiny_bundle_dir(tmp_path):
    """
    Build a minimal but valid ChromaMLP artifact bundle on disk.

    Two classes, 24-dim features, a scaler fit on random data and a label encoder over
    {'C:maj', 'A:min'} -- enough to exercise load + inference end to end without the real
    (gitignored) model.
    """
    rng = np.random.default_rng(0)
    x = rng.random((64, 24)).astype(np.float32)
    labels = np.array(["C:maj", "A:min"] * 32)

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
        "feat_smooth": 0,
    }
    save_bundle(str(tmp_path), model.state_dict(), scaler, encoder, config)
    return tmp_path


def real_artifacts_available() -> bool:
    return (REAL_ARTIFACTS / "model.pt").exists()
