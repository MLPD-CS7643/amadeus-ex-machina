"""
Light coverage for MirDataProcessor's CSV -> model-data path.

Drives ``prepare_model_data`` over a small synthetic ``combined_data.csv`` so the
scaling + label-encoding + train/test split are exercised without any mirdata download.
Skips if MirDataProcessor can't be constructed offline.
"""

import numpy as np
import pandas as pd
import pytest

from amadeus_train.datasets import MirDataProcessor


@pytest.fixture
def processor(tmp_path):
    try:
        return MirDataProcessor(
            output_dir=tmp_path / "processed", batch_size=8, process_sequential=False
        )
    except Exception as exc:  # noqa: BLE001 - offline / missing index -> skip
        pytest.skip(f"MirDataProcessor unavailable offline: {exc}")


def test_prepare_model_data_tabular(processor):
    rng = np.random.default_rng(0)
    feats = rng.random((40, 4))
    labels = np.array(["C:maj", "A:min"] * 20)
    pd.DataFrame(np.column_stack([feats, labels])).to_csv(
        processor.combined_csv_path, index=False, header=False
    )

    X_train, X_test, y_train, y_test = processor.prepare_model_data("combined")

    # Scaler + encoder were fit and persisted.
    assert processor.scaler.n_features_in_ == 4
    assert set(processor.label_encoder.classes_) == {"A:min", "C:maj"}
    assert processor.scaler_path.exists()
    assert processor.label_encoder_path.exists()

    # 80/20 split over 40 rows, features stay 4-dim.
    assert len(X_train) + len(X_test) == 40
    assert len(y_train) == len(X_train)
    assert X_train.shape[1] == 4
