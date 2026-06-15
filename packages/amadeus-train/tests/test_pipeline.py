"""
Train-pipeline tests on synthetic data (no GuitarSet download).

These exercise the augmentation helpers, the train loop, and the artifact round-trip so
we know the training package integrates with amadeus-core end to end without pulling the
multi-GB dataset.
"""

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from amadeus_core.artifacts import load_bundle
from amadeus_core.inference import InferenceEngine
from amadeus_core.models import ChromaMLP
from amadeus_train.cli import build_parser
from amadeus_train.guitarset import (
    augment_transpositions,
    augment_windows,
    fold_majmin,
    make_windows,
    train,
)


def _synthetic(n_per_track=60, n_tracks=2):
    rng = np.random.default_rng(0)
    X = rng.random((n_per_track * n_tracks, 24)).astype(np.float32)
    base = np.array(["C:maj", "A:min", "N"])
    y = base[rng.integers(0, 3, size=X.shape[0])]
    groups = np.repeat([f"t{t}" for t in range(n_tracks)], n_per_track)
    return X, y, groups


def test_cli_parser_defaults():
    args = build_parser().parse_args(
        ["--max-tracks", "4", "--epochs", "1", "--no-leadsheet"]
    )
    assert args.max_tracks == 4
    assert args.epochs == 1
    assert args.leadsheet is False
    assert args.arch == "mlp"


def test_fold_majmin_collapses_qualities():
    y = np.array(["C:maj7", "A:min7", "G:7", "N"])
    folded = fold_majmin(y)
    assert set(folded) <= {"C:maj", "A:min", "G:maj", "N"}


def test_augment_transpositions_multiplies_by_12():
    X, y, _ = _synthetic()
    aug_X, aug_y = augment_transpositions(X, y)
    assert aug_X.shape == (X.shape[0] * 12, 24)
    assert aug_y.shape[0] == y.shape[0] * 12


def test_make_and_augment_windows():
    X, y, groups = _synthetic()
    win_X, win_y, win_g = make_windows(X, y, groups, seq_length=8, stride=4)
    assert win_X.shape[1:] == (8, 24)
    assert len(win_y) == len(win_g) == win_X.shape[0]
    aug_X, aug_y = augment_windows(win_X, win_y)
    assert aug_X.shape[0] == win_X.shape[0] * 12


def test_train_and_artifact_round_trip(tmp_path):
    from amadeus_core.artifacts import save_bundle

    X, y, _ = _synthetic(n_per_track=80)
    y = fold_majmin(y)
    scaler = MinMaxScaler().fit(X)
    encoder = LabelEncoder().fit(y)
    Xs = scaler.transform(X).astype(np.float32)
    ye = encoder.transform(y)
    num_classes = len(encoder.classes_)

    model = ChromaMLP(input_features=24, num_classes=num_classes, hidden_sizes=(16,))
    loader = DataLoader(
        TensorDataset(torch.tensor(Xs), torch.tensor(ye, dtype=torch.long)),
        batch_size=16,
        shuffle=True,
        drop_last=True,
    )
    device = torch.device("cpu")
    best_state, best_acc = train(
        model, loader, Xs, ye, Xs, ye, device, epochs=2, lr=1e-3, patience=0
    )
    assert best_state is not None
    assert 0.0 <= best_acc <= 1.0

    config = {
        "model_class": "ChromaMLP",
        "input_features": 24,
        "num_classes": num_classes,
        "hidden_sizes": [16],
        "dropout": 0.3,
        "hop_length": 2048,
        "feat_smooth": 0,
    }
    save_bundle(str(tmp_path), best_state, scaler, encoder, config)

    bundle = load_bundle(str(tmp_path))
    assert bundle.num_classes == num_classes
    engine = InferenceEngine()
    engine.attach_bundle(bundle)
    assert engine.is_loaded
