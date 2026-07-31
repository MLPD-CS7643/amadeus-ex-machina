"""
Tests for the constructor-injection training loop (`Solver`).

Uses a tiny ChromaMLP + synthetic learnable data on CPU, with ``verbose=False`` so the
loop runs without tqdm. Covers train/eval, history recording, checkpointing, the warmup
path, and both TrialMetric return modes.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amadeus_core.models import ChromaMLP
from amadeus_train.solver import Solver, TrialMetric

DIM = 8
BATCH = 8


def _loader(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.random((n, DIM)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.int64)  # a learnable signal
    return DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=BATCH)


def _make_solver(tmp_path, epochs=3, warmup_epochs=0, early_stop_epochs=0, lr=0.1):
    train_loader = _loader(64, seed=0)
    valid_loader = _loader(24, seed=1)
    model = ChromaMLP(input_features=DIM, num_classes=2, hidden_sizes=(8,))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return Solver(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        batch_size=BATCH,
        epochs=epochs,
        device="cpu",
        warmup_epochs=warmup_epochs,
        early_stop_epochs=early_stop_epochs,
        checkpoint_dir=tmp_path / "ckpts",
    )


def test_trial_metric_enum_distinct():
    assert TrialMetric.LOSS != TrialMetric.ACCURACY


def test_train_and_evaluate_records_history_and_checkpoints(tmp_path):
    solver = _make_solver(tmp_path, epochs=3)
    best_loss = solver.train_and_evaluate(verbose=False)

    assert isinstance(best_loss, float)
    assert len(solver.train_loss_history) == 3
    assert len(solver.valid_loss_history) == 3
    assert len(solver.train_accuracy_history) == 3
    assert len(solver.valid_accuracy_history) == 3
    assert solver.best_model is not None
    # A best-model checkpoint was written to the configured dir (the path bug fix).
    assert list((tmp_path / "ckpts").glob("*.pth"))


def test_evaluate_returns_loss_and_accuracy(tmp_path):
    solver = _make_solver(tmp_path, epochs=1)
    solver.train_and_evaluate(verbose=False)
    total_loss, avg_loss, accuracy = solver.evaluate(solver.valid_dataloader)
    assert total_loss >= 0.0
    assert avg_loss >= 0.0
    assert 0.0 <= accuracy <= 1.0


def test_accuracy_metric_return_in_unit_range(tmp_path):
    solver = _make_solver(tmp_path, epochs=2)
    best_acc = solver.train_and_evaluate(
        trial_metric=TrialMetric.ACCURACY, verbose=False
    )
    assert 0.0 <= best_acc <= 1.0


def test_warmup_path_runs(tmp_path):
    # warmup_epochs>0 exercises the private __lr_warmup branch without error.
    solver = _make_solver(tmp_path, epochs=2, warmup_epochs=2, lr=0.1)
    solver.train_and_evaluate(verbose=False)
    assert len(solver.train_loss_history) == 2
