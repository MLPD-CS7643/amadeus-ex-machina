import numpy as np
import mir_eval
from mir_eval.chord import evaluate

# Monkey patch their **old** lib code
mir_eval.chord.np = np
mir_eval.chord.np.int = int
mir_eval.chord.np.float = float


def compute_chord_annotation_scores(ref_fpath, est_fpath):
    """Provide the reference annotation file and the model created annotation file and compute MIR eval scores"""
    ref_intervals, ref_labels = _load_lab_file(ref_fpath)
    est_intervals, est_labels = _load_lab_file(est_fpath)

    ref_intervals = _round_intervals(ref_intervals, 6)
    est_intervals = _round_intervals(est_intervals, 6)

    scores = evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
    return scores


def _load_lab_file(file_path):
    intervals = []
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                break
            start_time = float(parts[0])
            end_time = float(parts[1])
            chord_label = parts[2]
            intervals.append([start_time, end_time])
            labels.append(chord_label)
    return intervals, labels


def _round_intervals(intervals, decimal_places=6):
    """The mir_eval lib has some floating point sensitivity, this is used to resolve those headaches"""
    return np.array(
        [
            [round(start, decimal_places), round(end, decimal_places)]
            for start, end in intervals
        ]
    )
