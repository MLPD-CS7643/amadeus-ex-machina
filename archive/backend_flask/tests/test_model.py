"""
Tests for the chroma model, the build_model factory, config-driven artifact loading,
and the GuitarSet training feature/label assembly.
"""
import json
import pickle

import numpy as np
import mir_eval
import pytest
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from app.features import chroma24_from_audio
from app.inference import InferenceEngine
from app.model import (
    ChromaMLP,
    ChromaTransformerModel,
    CRNNChordModel,
    build_model,
)


@pytest.mark.unit
class TestChromaMLP:
    def test_forward_shape(self):
        model = ChromaMLP(input_features=24, num_classes=7, hidden_sizes=(32, 16))
        out = model(torch.randn(8, 24))
        assert out.shape == (8, 7)


@pytest.mark.unit
class TestChromaTransformer:
    def test_forward_shape(self):
        model = ChromaTransformerModel(
            input_dim=24, seq_length=8, num_classes=5,
            d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        )
        out = model(torch.randn(4, 8, 24))
        assert out.shape == (4, 5)

    def test_sequence_inference_round_trip(self, tmp_path):
        classes = ["C:maj", "A:min", "N"]
        scaler = MinMaxScaler().fit(np.random.rand(50, 24))
        encoder = LabelEncoder().fit(classes)
        model = ChromaTransformerModel(
            input_dim=24, seq_length=8, num_classes=len(classes),
            d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        )
        (tmp_path / "scaler.pkl").write_bytes(pickle.dumps(scaler))
        (tmp_path / "label_encoder.pkl").write_bytes(pickle.dumps(encoder))
        torch.save(model.state_dict(), tmp_path / "model.pt")
        (tmp_path / "model_config.json").write_text(
            json.dumps({
                "model_class": "ChromaTransformerModel", "input_dim": 24,
                "seq_length": 8, "num_classes": len(classes), "d_model": 32,
                "nhead": 4, "num_layers": 2, "dim_feedforward": 64, "dropout": 0.1,
            })
        )
        engine = InferenceEngine()
        engine.load_artifacts(
            str(tmp_path / "model.pt"), str(tmp_path / "scaler.pkl"),
            str(tmp_path / "label_encoder.pkl"),
            config_path=str(tmp_path / "model_config.json"),
        )
        assert engine.is_loaded and engine.is_sequence and engine.seq_length == 8
        # One prediction per frame, even for a clip shorter than seq_length.
        preds = engine.run_inference(np.random.rand(50, 24).astype("float32"))
        assert preds.shape == (50,)
        assert engine.run_inference(np.random.rand(3, 24).astype("float32")).shape == (3,)


@pytest.mark.unit
class TestBuildModel:
    def test_builds_chroma_mlp(self):
        model = build_model(
            {"model_class": "ChromaMLP", "num_classes": 5, "hidden_sizes": [16]}
        )
        assert isinstance(model, ChromaMLP)
        assert model(torch.randn(4, 24)).shape == (4, 5)

    def test_builds_legacy_crnn(self):
        model = build_model(
            {"model_class": "CRNNChordModel", "num_classes": 10, "hidden_size": 4}
        )
        assert isinstance(model, CRNNChordModel)

    def test_unknown_model_class_raises(self):
        with pytest.raises(ValueError):
            build_model({"model_class": "NopeNet", "num_classes": 3})


@pytest.mark.unit
class TestLoadArtifactsConfig:
    def test_loads_chroma_mlp_from_config(self, tmp_path):
        classes = ["C:maj", "A:min", "N"]
        scaler = MinMaxScaler().fit(np.random.rand(50, 24))
        encoder = LabelEncoder().fit(classes)
        model = ChromaMLP(input_features=24, num_classes=len(classes), hidden_sizes=(16,))

        (tmp_path / "scaler.pkl").write_bytes(pickle.dumps(scaler))
        (tmp_path / "label_encoder.pkl").write_bytes(pickle.dumps(encoder))
        torch.save(model.state_dict(), tmp_path / "model.pt")
        (tmp_path / "model_config.json").write_text(
            json.dumps(
                {
                    "model_class": "ChromaMLP",
                    "input_features": 24,
                    "num_classes": len(classes),
                    "hidden_sizes": [16],
                    "dropout": 0.3,
                }
            )
        )

        engine = InferenceEngine()
        engine.load_artifacts(
            str(tmp_path / "model.pt"),
            str(tmp_path / "scaler.pkl"),
            str(tmp_path / "label_encoder.pkl"),
            config_path=str(tmp_path / "model_config.json"),
        )
        assert engine.is_loaded
        assert isinstance(engine.model, ChromaMLP)
        preds = engine.run_inference(np.random.rand(5, 24).astype("float32"))
        assert preds.shape == (5,)
        assert preds.max() < len(classes)


@pytest.mark.unit
class TestTrainingPipeline:
    """Validates the exact feature+label assembly the training script uses."""

    def test_chroma_and_label_alignment(self):
        sr = 22050
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        # C-major triad for first half, A-minor for second half.
        cmaj = sum(np.sin(2 * np.pi * f * t[: len(t) // 2]) for f in (261.63, 329.63, 392.0))
        amin = sum(np.sin(2 * np.pi * f * t[len(t) // 2:]) for f in (220.0, 261.63, 329.63))
        y = np.concatenate([cmaj, amin]).astype(np.float32)
        y /= np.max(np.abs(y))

        features, timestamps = chroma24_from_audio(y, sr)
        assert features.shape[1] == 24
        assert features.shape[0] == len(timestamps)

        intervals = np.array([[0.0, 1.0], [1.0, 2.0]])
        labels = ["C:maj", "A:min"]
        frame_labels = np.array(
            mir_eval.util.interpolate_intervals(
                intervals, labels, timestamps, fill_value="N"
            )
        )
        assert frame_labels.shape[0] == features.shape[0]
        assert frame_labels[0] == "C:maj"
        assert frame_labels[-1] == "A:min"
