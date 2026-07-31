"""
mir_eval evaluation of a chord-recognition bundle on held-out Billboard songs.

Reproduces the training group-split (by song) so we score exactly the songs the model
never saw, runs the bundle's serve pipeline per song, and reports duration-weighted
WCSR at several granularities (root / majmin / sevenths / mirex). Run with:

    uv run python -m amadeus_train.evaluate --bundle artifacts/billboard_sevenths
"""

import argparse

import mir_eval
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import amadeus_core.chords.eval  # noqa: F401 - imported for its mir_eval numpy-compat shim
from amadeus_core.artifacts import load_bundle
from amadeus_core.inference import InferenceEngine, majority_smooth
from amadeus_train.billboard import DEFAULT_DATA_HOME, load_or_build_billboard

_GRANULARITIES = ("root", "majmin", "sevenths", "mirex")


def _held_out_track_ids(seed: int, val_size: float) -> list[str]:
    """The song ids in the validation split (same split the trainer used)."""
    ns = argparse.Namespace(max_tracks=None, no_cache=False, data_home=str(DEFAULT_DATA_HOME))
    _X, y, groups = load_or_build_billboard(ns)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    _train_idx, val_idx = next(splitter.split(_X, y, groups))
    return sorted(set(groups[val_idx].tolist()))


def _sanitize(intervals: np.ndarray, labels: list[str]):
    """Sort and truncate overlaps so mir_eval accepts them (Billboard refs overlap)."""
    intervals = np.asarray(intervals, dtype=float)
    order = np.argsort(intervals[:, 0])
    iv = intervals[order].copy()
    lab = [labels[i] for i in order]
    for i in range(len(iv) - 1):
        if iv[i, 1] > iv[i + 1, 0]:  # overlap -> hand the region to the later chord
            iv[i, 1] = iv[i + 1, 0]
    keep = iv[:, 1] > iv[:, 0]
    return iv[keep], [lab for lab, k in zip(lab, keep, strict=True) if k]


def _predict(engine: InferenceEngine, features: np.ndarray, times: np.ndarray):
    """Run the bundle's serve pipeline on precomputed features -> (intervals, labels)."""
    scaled = engine.scale_features(features)
    idx = majority_smooth(engine.run_inference(scaled), engine.smoothing_window)
    anns = engine.decode_predictions(idx, times, float(times[-1]))
    intervals = np.array([[s, e] for s, e, _ in anns], dtype=float)
    labels = [lab for _, _, lab in anns]
    return intervals, labels


def evaluate_bundle(bundle_dir: str, seed: int, val_size: float) -> dict[str, float]:
    import mirdata

    engine = InferenceEngine()
    engine.attach_bundle(load_bundle(bundle_dir))
    ds = mirdata.initialize("billboard", data_home=str(DEFAULT_DATA_HOME))

    totals = dict.fromkeys(_GRANULARITIES, 0.0)
    weight_sum = 0.0
    val_ids = _held_out_track_ids(seed, val_size)
    print(f"Scoring {len(val_ids)} held-out songs with {bundle_dir} ...")
    for i, tid in enumerate(val_ids, 1):
        track = ds.track(tid)
        try:
            chroma = np.asarray(track.chroma, dtype=np.float32)
            chords = track.chords_full
            if chroma.ndim != 2 or chroma.shape[1] < 25 or chords is None:
                continue
            times = chroma[:, 0].astype(np.float64)
            est_int, est_lab = _predict(engine, chroma[:, 1:25], times)
            ref_int, ref_lab = _sanitize(chords.intervals, list(chords.labels))
            scores = mir_eval.chord.evaluate(ref_int, ref_lab, est_int, est_lab)
            dur = float(ref_int[-1][1] - ref_int[0][0])
            for g in _GRANULARITIES:
                totals[g] += scores[g] * dur
            weight_sum += dur
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(val_ids)}] {tid}: skipped ({exc})")
            continue
    return {g: (totals[g] / weight_sum if weight_sum else 0.0) for g in _GRANULARITIES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, help="artifact bundle directory to score")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.2)
    args = ap.parse_args()

    scores = evaluate_bundle(args.bundle, args.seed, args.val_size)
    print("\nHeld-out Billboard WCSR (duration-weighted):")
    for g in _GRANULARITIES:
        print(f"  {g:9s} {scores[g]:.4f}")


if __name__ == "__main__":
    main()
