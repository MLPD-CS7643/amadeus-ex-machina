"""
Train a self-consistent chord-recognition model on GuitarSet.

Pipeline: GuitarSet audio (mirdata) -> the SAME 24-dim chroma the web backend uses at
inference (``amadeus_core.features.chroma24_from_audio``) -> JAMS chord labels aligned per
frame -> ChromaMLP or ChromaTransformer. Artifacts are written where the API loads them by
default (the repo-level ``artifacts/guitarset_chroma/``), so the web app serves this model
with no train/inference feature mismatch.
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from amadeus_core.chords.majmin import to_majmin, transpose_label
from amadeus_core.features import N_CHROMA, chroma24_from_audio

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "guitarset_chroma"
DEFAULT_DATA_HOME = REPO_ROOT / "data" / "raw" / "guitarset"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "guitarset_chroma_cache"


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
            window_labels = yg[s : s + seq_length]
            vals, counts = np.unique(window_labels, return_counts=True)
            win_X.append(Xg[s : s + seq_length])
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
    import mir_eval
    import mirdata.datasets.guitarset

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
    model,
    train_loader,
    train_X,
    train_y,
    val_X,
    val_y,
    device,
    epochs,
    lr,
    patience,
    weight_decay=0.0,
    label_smoothing=0.0,
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
            best_state = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            no_improve = 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"  early stop at epoch {epoch}")
                break
    return best_state, best_acc


def load_or_build_features(args):
    """Load cached features or build them from GuitarSet; returns (X, y, groups)."""
    import mirdata

    dataset = mirdata.initialize("guitarset", data_home=args.data_home)
    if args.download:
        print("Downloading GuitarSet (index + annotations + audio_mix)...")
        dataset.download(partial_download=["index", "annotations", "audio_mix"])

    tag = (
        f"mt{args.max_tracks or 'all'}_hop{args.hop_length}"
        f"_ls{int(args.leadsheet)}_sm{args.feat_smooth}"
    )
    cache_path = CACHE_DIR / f"{tag}.npz"
    if cache_path.exists() and not args.no_cache:
        data = np.load(cache_path, allow_pickle=True)
        X, y, groups = data["X"], data["y"], data["groups"]
        print(f"Loaded cached features {X.shape} from {cache_path}")
    else:
        X, y, groups = build_dataset(
            dataset, args.hop_length, args.leadsheet, args.max_tracks, args.feat_smooth
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, y=y, groups=groups)
        print(f"Cached features to {cache_path}")
    return X, y, groups


def fold_majmin(y):
    folded = np.array([to_majmin(lbl) for lbl in y])
    print(f"Reduced to maj/min vocabulary: {len(set(folded))} classes")
    return folded
