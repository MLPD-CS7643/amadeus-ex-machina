"""
Model zoo for chord recognition.

The serving-relevant architectures (the ones the artifact registry can rebuild) are
re-exported here. Research-only architectures live alongside in their own modules
(e.g. ``amadeus_core.models.cnn``, ``amadeus_core.models.rnn``).
"""

from amadeus_core.models.chroma_mlp import ChromaMLP
from amadeus_core.models.chroma_transformer import (
    ChromaTransformerModel,
    PositionalEncoding,
)
from amadeus_core.models.crnn_legacy import (
    CHECKPOINT_HIDDEN_SIZE,
    CHECKPOINT_INPUT_FEATURES,
    CRNNChordModel,
)
from amadeus_core.models.registry import SEQUENCE_MODELS, build_model

__all__ = [
    "CHECKPOINT_HIDDEN_SIZE",
    "CHECKPOINT_INPUT_FEATURES",
    "ChromaMLP",
    "ChromaTransformerModel",
    "CRNNChordModel",
    "PositionalEncoding",
    "SEQUENCE_MODELS",
    "build_model",
]
