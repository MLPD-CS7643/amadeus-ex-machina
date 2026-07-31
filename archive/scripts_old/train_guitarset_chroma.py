"""
Train a self-consistent chord-recognition model on GuitarSet.

Pipeline: GuitarSet audio (mirdata) -> the SAME 24-dim chroma the web backend uses at
inference (``app.features.chroma24_from_audio``) -> JAMS chord labels aligned per
frame -> ChromaMLP. Artifacts are written where the backend loads them by default
(``backend/artifacts/guitarset_chroma/``), so the web app serves this model with no
train/inference feature mismatch.

Usage:
    python scripts/train_guitarset_chroma.py --download            # full run
    python scripts/train_guitarset_chroma.py --download --max-tracks 20 --epochs 15
"""
import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import mirdata  # noqa: E402
import mirdata.datasets.guitarset  # noqa: E402
import mir_eval  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from app.features import DEFAULT_HOP_LENGTH, N_CHROMA, chroma24_from_audio  # noqa: E402
from app.model import ChromaMLP, ChromaTransformerModel  # noqa: E402
from utils.model_utils import get_device, set_seed  # noqa: E402

DEFAULT_OUT = BACKEND_DIR / "artifacts" / "guitarset_chroma"
DEFAULT_DATA_HOME = REPO_ROOT / "data" / "raw" / "guitarset"

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_INDEX = {n: i for i, n in enumerate(NOTES)}
# Inlined (avoids importing utils.chord_remap, which pulls in the tinysoundfont dep).
FLAT_TO_SHARP = {
    "Cb": "B", "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "B#": "C", "E#": "F",
}


def to_majmin(label: str) -> str:
    """
    Reduce a chord label to the MIREX maj/min vocabulary: root:maj, root:min, or N.

    A chord with a minor third (min/dim/half-dim families) -> min; everything else
    with a defined root -> maj. This is the standard, far-more-learnable target for
    frame-level chord recognition.
    """
    if label in ("N", "X", ""):
        return "N"
    root, _, quality = label.partition(":")
    q = quality.lower()
    if q.startswith("min") or q.startswith("dim") or q.startswith("hdim"):
        return f"{root}:min"
    return f"{root}:maj"


def transpose_label(label: str, k: int) -> str:
    """Transpose a chord label up by ``k`` semitones (root only; quality is invariant)."""
    if label in ("N", "X", ""):
        return "N"
    root, _, quality = label.partition(":")
    root = FLAT_TO_SHARP.get(root, root)
    if root not in NOTE_INDEX:
        return label
    new_root = NOTES[(NOTE_INDEX[root] + k) % 12]
    return f"{new_root}:{quality}" if quality else new_root


def augment_transpositions(X: np.ndarray, y: np.ndarray):
    """
    Expand a (frames, 24) chroma set + labels with all 12 pitch transpositions.

    Each 12-bin chroma register (bass, treble) is rolled by k semitones and the chord
    root is shifted by the same k, so the model sees every chord in every key.
    """
    feats, labels = [], []
    for k in range(12):
        rolled = X.copy()
        rolled[:, :N_CHROMA] = np.roll(X[:, :N_CHROMA], k, axis=1)
        rolled[:, N_CHROMA:] = np.roll(X[:, N_CHROMA:], k, axis=1)
        feats.append(rolled)
        labels.append(np.array([transpose_label(lbl, k) for lbl in y]))
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels)


def make_windows(X, y, groups, seq_length, stride):
    """
    Build overlapping windows over each track's contiguous frames.

    Returns (windows[m, seq_length, 24] float32, labels[m] str, win_groups[m] track id),
    where each window's label is the mode chord over its frames.
    """
    win_X, win_y, win_g = [], [], []
    n = len(groups)
    i = 0
    while i < n:
        j = i
        while j < n and groups[j] == groups[i]:
            j += 1
        Xg, yg = X[i:j], y[i:j]
        for s in range(0, len(Xg) - seq_length + 1, stride):
            window_labels = yg[s:s + seq_length]
            vals, counts = np.unique(window_labels, return_counts=True)
            win_X.append(Xg[s:s + seq_length])
            win_y.append(vals[int(np.argmax(counts))])
            win_g.append(groups[i])
        i = j
    if not win_X:
        raise SystemExit("No windows built; seq_length too large for the tracks.")
    return np.stack(win_X).astype(np.float32), np.array(win_y), np.array(win_g)


def augment_windows(win_X, win_y):
    """Expand windows with all 12 pitch transpositions (roll both chroma registers)."""
    feats, labels = [], []
    for k in range(12):
        rolled = win_X.copy()
        rolled[:, :, :N_CHROMA] = np.roll(win_X[:, :, :N_CHROMA], k, axis=2)
        rolled[:, :, N_CHROMA:] = np.roll(win_X[:, :, N_CHROMA:], k, axis=2)
        feats.append(rolled)
        labels.append(np.array([transpose_label(lbl, k) for lbl in win_y]))
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels)


def build_dataset(dataset, hop_length, leadsheet, max_tracks, smooth_window=0):
    """Return (X[N,24] float32, y[N] str labels, groups[N] track ids)."""
    track_ids = dataset.track_ids
    if max_tracks:
        track_ids = track_ids[:max_tracks]
    print(f"Building features from {len(track_ids)} GuitarSet tracks...")

    feats_all, labels_all, groups_all = [], [], []
    for i, track_id in enumerate(track_ids, 1):
        track = dataset.track(track_id)
        try:
            audio, sr = track.audio_mix
            features, timestamps = chroma24_from_audio(
                audio, sr, hop_length=hop_length, smooth_window=smooth_window
            )
            chords = mirdata.datasets.guitarset.load_chords(track.jams_path, leadsheet)
            if chords is None or len(chords.intervals) == 0:
                print(f"  [{i}/{len(track_ids)}] {track_id}: no chords, skipping")
                continue
            frame_labels = np.array(
                mir_eval.util.interpolate_intervals(
                    chords.intervals, chords.labels, timestamps, fill_value="N"
                )
            )
            frame_labels = np.where(frame_labels == "X", "N", frame_labels)
        except Exception as exc:  # noqa: BLE001 - skip unreadable tracks
            print(f"  [{i}/{len(track_ids)}] {track_id}: error ({exc}), skipping")
            continue

        feats_all.append(features)
        labels_all.append(frame_labels)
        groups_all.append(np.full(features.shape[0], track_id))
        if i % 20 == 0 or i == len(track_ids):
            print(f"  [{i}/{len(track_ids)}] processed")

    if not feats_all:
        raise SystemExit("No usable GuitarSet tracks found. Did you pass --download?")

    X = np.concatenate(feats_all).astype(np.float32)
    y = np.concatenate(labels_all)
    groups = np.concatenate(groups_all)
    print(f"Total frames: {X.shape[0]}, raw label classes: {len(set(y))}")
    return X, y, groups


def _accuracy(model, X, y, device):
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        yb = torch.tensor(y, dtype=torch.long, device=device)
        return (model(xb).argmax(1) == yb).float().mean().item()


def train(
    model, train_loader, train_X, train_y, val_X, val_y, device, epochs, lr, patience,
    weight_decay=0.0, label_smoothing=0.0,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_acc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        train_acc = _accuracy(model, train_X, train_y, device)
        val_acc = _accuracy(model, val_X, val_y, device)
        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch:3d}/{epochs}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  lr={lr_now:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"  early stop at epoch {epoch}")
                break
    return best_state, best_acc


def main():
    ap = argparse.ArgumentParser(description="Train a GuitarSet chroma chord model.")
    ap.add_argument("--download", action="store_true", help="download GuitarSet (partial)")
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
        "--feat-smooth", type=int, default=15,
        help="temporal chroma averaging window in frames (~0.7s at default hop)",
    )
    ap.add_argument(
        "--leadsheet", action=argparse.BooleanOptionalAction, default=True,
        help="use instructed leadsheet chords (--no-leadsheet for performed)",
    )
    ap.add_argument(
        "--vocab", choices=["full", "majmin"], default="majmin",
        help="chord vocabulary: 'majmin' folds qualities to maj/min/N (recommended)",
    )
    ap.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=True,
        help="augment train set with 12 chroma transpositions (--no-augment to disable)",
    )
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true", help="ignore cached features")
    # Architecture
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="mlp")
    ap.add_argument("--seq-length", type=int, default=16, help="transformer window size")
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
    args = ap.parse_args()

    set_seed()
    hidden_sizes = tuple(int(h) for h in args.hidden.split(",") if h)

    dataset = mirdata.initialize("guitarset", data_home=args.data_home)
    if args.download:
        # "index" pulls the remote dataset index (required in mirdata 0.3.9);
        # annotations + audio_mix skip the multi-GB hex audio. force_overwrite=False
        # makes re-runs skip already-downloaded files.
        print("Downloading GuitarSet (index + annotations + audio_mix)...")
        dataset.download(partial_download=["index", "annotations", "audio_mix"])

    # Cache the (slow) feature extraction so vocab/model experiments don't re-extract.
    cache_dir = REPO_ROOT / "data" / "processed" / "guitarset_chroma_cache"
    tag = (
        f"mt{args.max_tracks or 'all'}_hop{args.hop_length}"
        f"_ls{int(args.leadsheet)}_sm{args.feat_smooth}"
    )
    cache_path = cache_dir / f"{tag}.npz"
    if cache_path.exists() and not args.no_cache:
        data = np.load(cache_path, allow_pickle=True)
        X, y, groups = data["X"], data["y"], data["groups"]
        print(f"Loaded cached features {X.shape} from {cache_path}")
    else:
        X, y, groups = build_dataset(
            dataset, args.hop_length, args.leadsheet, args.max_tracks, args.feat_smooth
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, y=y, groups=groups)
        print(f"Cached features to {cache_path}")

    if args.vocab == "majmin":
        y = np.array([to_majmin(lbl) for lbl in y])
        print(f"Reduced to maj/min vocabulary: {len(set(y))} raw classes")

    # Split by track (GroupShuffleSplit guarantees no track spans both sets).
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
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
        print(f"Windows (seq={args.seq_length}, stride={args.stride}): "
              f"train {len(Xtr)}, val {len(Xva)}")
        if args.augment:
            Xtr, ytr = augment_windows(Xtr, ytr)
            print(f"Augmented train windows -> {len(Xtr)}")

        scaler = MinMaxScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))

        def _scale(W):
            shape = W.shape
            return scaler.transform(W.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

        Xtr_s, Xva_s = _scale(Xtr), _scale(Xva)
        encoder = LabelEncoder().fit(np.concatenate([ytr, yva]))
        num_classes = len(encoder.classes_)
        ytr_e, yva_e = encoder.transform(ytr), encoder.transform(yva)
        print(f"Encoded {num_classes} chord classes.")

        model = ChromaTransformerModel(
            input_dim=X.shape[1], seq_length=args.seq_length, num_classes=num_classes,
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward, dropout=args.tf_dropout,
        )
        print(f"Training ChromaTransformer(d_model={args.d_model}, "
              f"layers={args.num_layers}) on {device}...")
        config = {
            "model_class": "ChromaTransformerModel",
            "input_features": int(X.shape[1]), "input_dim": int(X.shape[1]),
            "seq_length": int(args.seq_length), "num_classes": int(num_classes),
            "d_model": int(args.d_model), "nhead": int(args.nhead),
            "num_layers": int(args.num_layers),
            "dim_feedforward": int(args.dim_feedforward), "dropout": float(args.tf_dropout),
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
            input_features=X.shape[1], num_classes=num_classes,
            hidden_sizes=hidden_sizes, dropout=args.dropout,
        )
        print(f"Training ChromaMLP{hidden_sizes} on {device}...")
        config = {
            "model_class": "ChromaMLP", "input_features": int(X.shape[1]),
            "num_classes": int(num_classes), "hidden_sizes": list(hidden_sizes),
            "dropout": float(args.dropout),
        }

    train_ds = TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr_e, dtype=torch.long))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    sample = rng.choice(len(Xtr_s), size=min(20000, len(Xtr_s)), replace=False)
    best_state, best_acc = train(
        model, train_loader, Xtr_s[sample], ytr_e[sample], Xva_s, yva_e,
        device, args.epochs, args.lr, args.patience,
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
    )
    print(f"Best val accuracy: {best_acc:.4f}")

    config.update({
        "hop_length": int(args.hop_length),
        "feat_smooth": int(args.feat_smooth),
        "val_accuracy": round(float(best_acc), 4),
        "trained_on": "guitarset",
        "vocab": args.vocab,
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
