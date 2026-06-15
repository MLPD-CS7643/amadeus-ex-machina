"""
Console entry point: train the GuitarSet chroma chord model.

Usage:
    amadeus-train --download --no-leadsheet --vocab majmin --arch transformer
    amadeus-train --max-tracks 4 --epochs 1 --no-leadsheet     # quick smoke run
"""

import argparse

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from amadeus_core.artifacts import save_bundle
from amadeus_core.device import get_device, set_seed
from amadeus_core.features import DEFAULT_HOP_LENGTH
from amadeus_core.models import ChromaMLP, ChromaTransformerModel
from amadeus_train.guitarset import (
    DEFAULT_DATA_HOME,
    DEFAULT_OUT,
    augment_transpositions,
    augment_windows,
    fold_majmin,
    load_or_build_features,
    make_windows,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a GuitarSet chroma chord model.")
    ap.add_argument(
        "--download", action="store_true", help="download GuitarSet (partial)"
    )
    ap.add_argument("--data-home", default=str(DEFAULT_DATA_HOME))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", default="256,256", help="comma-separated hidden sizes")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    ap.add_argument(
        "--feat-smooth",
        type=int,
        default=15,
        help="temporal chroma averaging window in frames (~0.7s at default hop)",
    )
    ap.add_argument(
        "--leadsheet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use instructed leadsheet chords (--no-leadsheet for performed)",
    )
    ap.add_argument(
        "--vocab",
        choices=["full", "majmin"],
        default="majmin",
        help="chord vocabulary: 'majmin' folds qualities to maj/min/N (recommended)",
    )
    ap.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="augment train set with 12 chroma transpositions (--no-augment to disable)",
    )
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true", help="ignore cached features")
    # Architecture.
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="mlp")
    ap.add_argument(
        "--seq-length", type=int, default=16, help="transformer window size"
    )
    ap.add_argument("--stride", type=int, default=8, help="training window stride")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=1024)
    ap.add_argument("--tf-dropout", type=float, default=0.1, help="transformer dropout")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    set_seed()
    hidden_sizes = tuple(int(h) for h in args.hidden.split(",") if h)

    X, y, groups = load_or_build_features(args)
    if args.vocab == "majmin":
        y = fold_majmin(y)

    # Split by track (GroupShuffleSplit guarantees no track spans both sets).
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.val_size, random_state=args.seed
    )
    train_idx, val_idx = next(splitter.split(X, y, groups))
    train_tracks = set(np.unique(groups[train_idx]))
    val_tracks = set(np.unique(groups[val_idx]))
    print(f"Split by track: train {len(train_idx)}, val {len(val_idx)} frames")

    device = get_device()
    rng = np.random.default_rng(args.seed)

    if args.arch == "transformer":
        win_X, win_y, win_g = make_windows(X, y, groups, args.seq_length, args.stride)
        tr = np.array([g in train_tracks for g in win_g])
        va = np.array([g in val_tracks for g in win_g])
        Xtr, ytr, Xva, yva = win_X[tr], win_y[tr], win_X[va], win_y[va]
        print(
            f"Windows (seq={args.seq_length}, stride={args.stride}): "
            f"train {len(Xtr)}, val {len(Xva)}"
        )
        if args.augment:
            Xtr, ytr = augment_windows(Xtr, ytr)
            print(f"Augmented train windows -> {len(Xtr)}")

        scaler = MinMaxScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))

        def _scale(W):
            shape = W.shape
            return (
                scaler.transform(W.reshape(-1, shape[-1]))
                .reshape(shape)
                .astype(np.float32)
            )

        Xtr_s, Xva_s = _scale(Xtr), _scale(Xva)
        encoder = LabelEncoder().fit(np.concatenate([ytr, yva]))
        num_classes = len(encoder.classes_)
        ytr_e, yva_e = encoder.transform(ytr), encoder.transform(yva)
        print(f"Encoded {num_classes} chord classes.")

        model: torch.nn.Module = ChromaTransformerModel(
            input_dim=X.shape[1],
            seq_length=args.seq_length,
            num_classes=num_classes,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.tf_dropout,
        )
        print(
            f"Training ChromaTransformer(d_model={args.d_model}, "
            f"layers={args.num_layers}) on {device}..."
        )
        config = {
            "model_class": "ChromaTransformerModel",
            "input_features": int(X.shape[1]),
            "input_dim": int(X.shape[1]),
            "seq_length": int(args.seq_length),
            "num_classes": int(num_classes),
            "d_model": int(args.d_model),
            "nhead": int(args.nhead),
            "num_layers": int(args.num_layers),
            "dim_feedforward": int(args.dim_feedforward),
            "dropout": float(args.tf_dropout),
        }
    else:
        Xtr, ytr = X[train_idx], y[train_idx]
        Xva, yva = X[val_idx], y[val_idx]
        if args.augment:
            Xtr, ytr = augment_transpositions(Xtr, ytr)
            print(f"Augmented train with 12 transpositions -> {Xtr.shape[0]} frames")
        scaler = MinMaxScaler().fit(Xtr)
        encoder = LabelEncoder().fit(np.concatenate([ytr, yva]))
        num_classes = len(encoder.classes_)
        Xtr_s = scaler.transform(Xtr).astype(np.float32)
        Xva_s = scaler.transform(Xva).astype(np.float32)
        ytr_e, yva_e = encoder.transform(ytr), encoder.transform(yva)
        print(f"Encoded {num_classes} chord classes.")
        model = ChromaMLP(
            input_features=X.shape[1],
            num_classes=num_classes,
            hidden_sizes=hidden_sizes,
            dropout=args.dropout,
        )
        print(f"Training ChromaMLP{hidden_sizes} on {device}...")
        config = {
            "model_class": "ChromaMLP",
            "input_features": int(X.shape[1]),
            "num_classes": int(num_classes),
            "hidden_sizes": list(hidden_sizes),
            "dropout": float(args.dropout),
        }

    train_ds = TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr_e, dtype=torch.long))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    sample = rng.choice(len(Xtr_s), size=min(20000, len(Xtr_s)), replace=False)
    best_state, best_acc = train(
        model,
        train_loader,
        Xtr_s[sample],
        ytr_e[sample],
        Xva_s,
        yva_e,
        device,
        args.epochs,
        args.lr,
        args.patience,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )
    print(f"Best val accuracy: {best_acc:.4f}")

    config.update(
        {
            "hop_length": int(args.hop_length),
            "feat_smooth": int(args.feat_smooth),
            "val_accuracy": round(float(best_acc), 4),
            "trained_on": "guitarset",
            "vocab": args.vocab,
        }
    )
    out = save_bundle(args.out_dir, best_state, scaler, encoder, config)
    print(f"Saved artifacts to {out}")


if __name__ == "__main__":
    main()
