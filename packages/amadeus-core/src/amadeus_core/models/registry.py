"""
Model registry + builder.

``build_model`` reconstructs a model from a saved ``model_config.json`` dict so the
artifact files alone fully determine the architecture. ``SEQUENCE_MODELS`` flags which
architectures consume a window of frames (and thus need sliding-window inference)
rather than a single frame.
"""

import torch.nn as nn

from amadeus_core.models.chroma_mlp import ChromaMLP
from amadeus_core.models.chroma_transformer import ChromaTransformerModel
from amadeus_core.models.crnn_legacy import CHECKPOINT_HIDDEN_SIZE, CRNNChordModel

# Architectures whose forward expects (batch, seq_length, input_dim) and emit one label
# per window -- inference feeds sliding windows rather than single frames.
SEQUENCE_MODELS = {"ChromaTransformerModel"}

_MODEL_REGISTRY = {
    "ChromaMLP": ChromaMLP,
    "CRNNChordModel": CRNNChordModel,
    "ChromaTransformerModel": ChromaTransformerModel,
}


def build_model(config: dict) -> nn.Module:
    """
    Construct a model from a saved ``model_config.json`` dict.

    Required keys: ``model_class`` and ``num_classes``. Other keys are
    architecture-specific and fall back to sensible defaults.
    """
    name = config.get("model_class")
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_class: {name!r}")

    if name == "ChromaMLP":
        return ChromaMLP(
            input_features=int(config.get("input_features", 24)),
            num_classes=int(config["num_classes"]),
            hidden_sizes=tuple(config.get("hidden_sizes", (256, 256))),
            dropout=float(config.get("dropout", 0.3)),
        )
    if name == "ChromaTransformerModel":
        return ChromaTransformerModel(
            input_dim=int(config.get("input_features", config.get("input_dim", 24))),
            seq_length=int(config["seq_length"]),
            num_classes=int(config["num_classes"]),
            d_model=int(config.get("d_model", 128)),
            nhead=int(config.get("nhead", 8)),
            num_layers=int(config.get("num_layers", 4)),
            dim_feedforward=int(config.get("dim_feedforward", 1024)),
            dropout=float(config.get("dropout", 0.1)),
        )
    # CRNNChordModel
    return CRNNChordModel(
        input_features=int(config.get("input_features", 24)),
        num_classes=int(config["num_classes"]),
        hidden_size=int(config.get("hidden_size", CHECKPOINT_HIDDEN_SIZE)),
    )
