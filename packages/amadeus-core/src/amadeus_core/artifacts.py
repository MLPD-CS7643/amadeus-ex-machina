"""
The artifact bundle contract: ``model.pt`` + ``scaler.pkl`` + ``label_encoder.pkl`` +
``model_config.json``.

A bundle is fully self-describing: ``model_config.json`` names the architecture (and its
hyperparameters + the feature params used at training), so :func:`load_bundle` can rebuild
the exact model and the inference engine can extract matching features. Training writes a
bundle with :func:`save_bundle`; the API loads one with :func:`load_bundle`.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from amadeus_core.device import get_device
from amadeus_core.errors import ProcessingError
from amadeus_core.features.chroma import DEFAULT_HOP_LENGTH, FEATURE_DIM
from amadeus_core.models.crnn_legacy import CHECKPOINT_HIDDEN_SIZE, CRNNChordModel
from amadeus_core.models.registry import SEQUENCE_MODELS, build_model

MODEL_FILE = "model.pt"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"
CONFIG_FILE = "model_config.json"


@dataclass
class ArtifactBundle:
    """A loaded, ready-to-serve model with its preprocessing + feature parameters."""

    model: nn.Module
    scaler: Any
    label_encoder: Any
    config: dict
    device: torch.device
    arch: str
    num_classes: int
    input_features: int
    is_sequence: bool
    seq_length: int
    hop_length: int
    feat_smooth: int


def load_bundle(
    artifact_dir: str | None = None,
    *,
    model_path: str | None = None,
    scaler_path: str | None = None,
    encoder_path: str | None = None,
    config_path: str | None = None,
    device: torch.device | None = None,
) -> ArtifactBundle:
    """
    Load + rebuild a model bundle.

    Pass either ``artifact_dir`` (uses the standard filenames) or explicit paths. If a
    ``model_config.json`` is present the architecture is rebuilt via :func:`build_model`;
    otherwise it falls back to the legacy checkpoint-faithful ``CRNNChordModel``.

    Raises:
        ProcessingError: if a required artifact is missing or fails to load.
    """
    if artifact_dir is not None:
        base = Path(artifact_dir)
        model_path = model_path or str(base / MODEL_FILE)
        scaler_path = scaler_path or str(base / SCALER_FILE)
        encoder_path = encoder_path or str(base / ENCODER_FILE)
        if config_path is None and (base / CONFIG_FILE).exists():
            config_path = str(base / CONFIG_FILE)

    for path, name in (
        (model_path, "model checkpoint"),
        (scaler_path, "scaler"),
        (encoder_path, "label encoder"),
    ):
        if not path or not Path(path).exists():
            raise ProcessingError(
                f"Required {name} artifact is missing; the server is misconfigured.",
                stage="load_artifacts",
            )
    # All three are present and non-None past this point.
    assert (
        model_path is not None and scaler_path is not None and encoder_path is not None
    )

    try:
        with open(scaler_path, "rb") as fin:
            scaler = pickle.load(fin)
        with open(encoder_path, "rb") as fin:
            label_encoder = pickle.load(fin)

        num_classes = len(label_encoder.classes_)
        input_features = int(getattr(scaler, "n_features_in_", FEATURE_DIM))
        device = device or get_device()

        if config_path and Path(config_path).exists():
            with open(config_path) as fin:
                config = json.load(fin)
            config.setdefault("num_classes", num_classes)
            config.setdefault("input_features", input_features)
            model = build_model(config)
            arch = config.get("model_class", "unknown")
            hop_length = int(config.get("hop_length", DEFAULT_HOP_LENGTH))
            feat_smooth = int(config.get("feat_smooth", 0))
            is_sequence = arch in SEQUENCE_MODELS
            seq_length = int(config.get("seq_length", 0))
        else:
            config = {}
            model = CRNNChordModel(
                input_features=input_features,
                num_classes=num_classes,
                hidden_size=CHECKPOINT_HIDDEN_SIZE,
            )
            arch = "CRNNChordModel (legacy)"
            hop_length = DEFAULT_HOP_LENGTH
            feat_smooth = 0
            is_sequence = False
            seq_length = 0

        state = torch.load(model_path, map_location="cpu")
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        model.load_state_dict(state)
        model.to(device).eval()
    except ProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ProcessingError(
            "Failed to load model artifacts.",
            stage="load_artifacts",
            original_error=exc,
        ) from exc

    return ArtifactBundle(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        config=config,
        device=device,
        arch=arch,
        num_classes=num_classes,
        input_features=input_features,
        is_sequence=is_sequence,
        seq_length=seq_length,
        hop_length=hop_length,
        feat_smooth=feat_smooth,
    )


def save_bundle(
    out_dir: str,
    model_state: dict,
    scaler: Any,
    label_encoder: Any,
    config: dict,
) -> Path:
    """Write a complete artifact bundle to ``out_dir`` and return the directory path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, out / MODEL_FILE)
    with open(out / SCALER_FILE, "wb") as fout:
        pickle.dump(scaler, fout)
    with open(out / ENCODER_FILE, "wb") as fout:
        pickle.dump(label_encoder, fout)
    with open(out / CONFIG_FILE, "w") as fout:
        json.dump(config, fout, indent=2)
    return out
