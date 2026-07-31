"""
mir_eval-based chord scoring of two ``.lab`` files (reference vs. estimate).

The pinned ``mir_eval`` uses NumPy aliases (``np.int`` / ``np.float``) removed in modern
NumPy, so we monkeypatch them back before importing ``evaluate``. Keep that shim.
"""

import mir_eval
import numpy as np
from mir_eval.chord import evaluate

# Monkeypatch the removed NumPy aliases the pinned mir_eval still references.
mir_eval.chord.np = np
mir_eval.chord.np.int = int
mir_eval.chord.np.float = float


def compute_chord_annotation_scores(ref_fpath, est_fpath):
    """Score an estimated annotation file against a reference; returns mir_eval metrics."""
    ref_intervals, ref_labels = _load_lab_file(ref_fpath)
    est_intervals, est_labels = _load_lab_file(est_fpath)

    ref_intervals = _round_intervals(ref_intervals, 6)
    est_intervals = _round_intervals(est_intervals, 6)

    return evaluate(ref_intervals, ref_labels, est_intervals, est_labels)


def _load_lab_file(file_path):
    intervals = []
    labels = []
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                break
            intervals.append([float(parts[0]), float(parts[1])])
            labels.append(parts[2])
    return intervals, labels


def _round_intervals(intervals, decimal_places=6):
    """mir_eval has floating-point sensitivity; rounding resolves those headaches."""
    return np.array(
        [
            [round(start, decimal_places), round(end, decimal_places)]
            for start, end in intervals
        ]
    )
