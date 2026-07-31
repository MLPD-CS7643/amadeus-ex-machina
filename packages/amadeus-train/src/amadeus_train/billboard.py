"""
McGill-Billboard dataset builder for produced-music chord training.

Uses the precomputed NNLS **bothchroma** features + full chord labels that ship with
the dataset (via ``mirdata``) -- no audio, no scraping, gold-aligned. bothchroma is the
same 24-dim (bass|treble) NNLS representation the serving model expects, so training on
it (and serving with an NNLS front-end) keeps train and serve features consistent.

Mirrors :func:`amadeus_train.guitarset.build_dataset`, returning
``(X[N,24] float32, y[N] str labels, groups[N] track ids)``.
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_HOME = REPO_ROOT / "data" / "raw" / "billboard"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "billboard_bothchroma_cache"

_EXPECTED_COLS = 25  # 1 time column + 24 chroma dims (bass 12 | treble 12)


def build_dataset_billboard(data_home, max_tracks=None):
    """Return (X[N,24] float32, y[N] str, groups[N] track id) from Billboard bothchroma."""
    import mir_eval
    import mirdata

    ds = mirdata.initialize("billboard", data_home=str(data_home))
    track_ids = ds.track_ids
    if max_tracks:
        track_ids = track_ids[:max_tracks]
    print(f"Building features from {len(track_ids)} Billboard tracks (bothchroma)...")

    feats_all, labels_all, groups_all = [], [], []
    kept = 0
    for i, track_id in enumerate(track_ids, 1):
        track = ds.track(track_id)
        try:
            chroma = np.asarray(track.chroma, dtype=np.float32)
            chords = track.chords_full
            if chroma.ndim != 2 or chroma.shape[1] < _EXPECTED_COLS:
                continue
            if chords is None or len(chords.intervals) == 0:
                continue
            times = chroma[:, 0].astype(np.float64)
            features = chroma[:, 1:_EXPECTED_COLS]  # (frames, 24)
            frame_labels = np.array(
                mir_eval.util.interpolate_intervals(
                    chords.intervals, chords.labels, times, fill_value="N"
                )
            )
            feats_all.append(features)
            labels_all.append(frame_labels)
            groups_all.append(np.full(features.shape[0], track_id))
            kept += 1
            if i % 100 == 0:
                print(f"  [{i}/{len(track_ids)}] kept {kept}")
        except Exception as exc:  # noqa: BLE001 - skip unreadable tracks, keep going
            print(f"  [{i}/{len(track_ids)}] {track_id}: skipped ({exc})")
            continue

    if not feats_all:
        raise RuntimeError("No Billboard tracks yielded features; check the download.")
    X = np.concatenate(feats_all).astype(np.float32)
    y = np.concatenate(labels_all)
    groups = np.concatenate(groups_all)
    print(f"Built {X.shape} from {kept} tracks.")
    return X, y, groups


def load_or_build_billboard(args):
    """Cache wrapper mirroring guitarset.load_or_build_features."""
    tag = f"billboard_mt{args.max_tracks or 'all'}"
    cache_path = CACHE_DIR / f"{tag}.npz"
    if cache_path.exists() and not args.no_cache:
        data = np.load(cache_path, allow_pickle=True)
        X, y, groups = data["X"], data["y"], data["groups"]
        print(f"Loaded cached features {X.shape} from {cache_path}")
        return X, y, groups
    X, y, groups = build_dataset_billboard(args.data_home, args.max_tracks)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, y=y, groups=groups)
    print(f"Cached features to {cache_path}")
    return X, y, groups
