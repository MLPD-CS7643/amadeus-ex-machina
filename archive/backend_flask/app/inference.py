"""
InferenceEngine: orchestrates audio -> chord-annotation inference.

Pipeline (per the design doc, simplified to per-frame classification because the
shipped checkpoint was trained on tabular chroma):

    audio -> extract_chroma24 -> scaler.transform -> CRNN (batched, per frame)
          -> argmax -> label_encoder.inverse_transform
          -> merge consecutive labels into gap-free segments -> .lab

A single engine instance is shared across worker threads; the model runs in
``eval()`` and forward passes are serialized with a lock for thread-safety.
"""
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from app.errors import ProcessingError
from app.features import DEFAULT_HOP_LENGTH, FEATURE_DIM, extract_chroma24
from app.model import CHECKPOINT_HIDDEN_SIZE, SEQUENCE_MODELS, CRNNChordModel, build_model

# Default majority-vote smoothing window (frames) over per-frame predictions.
# ~9 frames at hop 2048 / sr 22050 ≈ 0.8s; odd number; <=1 disables.
DEFAULT_SMOOTHING_WINDOW = 9


def majority_smooth(indices: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window majority vote over a sequence of class indices."""
    n = len(indices)
    if window <= 1 or n <= 2:
        return indices
    half = window // 2
    out = indices.copy()
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals, counts = np.unique(indices[lo:hi], return_counts=True)
        out[i] = vals[int(np.argmax(counts))]
    return out

logger = logging.getLogger(__name__)

# (start_time, end_time, chord_label)
Annotation = Tuple[float, float, str]
ProgressCallback = Callable[[int, str], None]


def select_device() -> torch.device:
    """Prefer MPS, then CUDA, then CPU (design Property 19)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class InferenceEngine:
    def __init__(self, smoothing_window: int = DEFAULT_SMOOTHING_WINDOW) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.scaler = None
        self.label_encoder = None
        self.device: Optional[torch.device] = None
        self.smoothing_window = smoothing_window
        self.hop_length = DEFAULT_HOP_LENGTH
        self.feat_smooth = 0
        self.is_sequence = False
        self.seq_length = 0
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_artifacts(
        self,
        model_path: str,
        scaler_path: str,
        encoder_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Load model weights + scaler + label encoder. Raises on misconfiguration.

        If ``config_path`` points to a ``model_config.json``, the architecture is
        rebuilt via :func:`build_model` (the retrained ChromaMLP path). Otherwise it
        falls back to the legacy checkpoint-faithful ``CRNNChordModel``.
        """
        for path, name in (
            (model_path, "model checkpoint"),
            (scaler_path, "scaler"),
            (encoder_path, "label encoder"),
        ):
            if not Path(path).exists():
                raise ProcessingError(
                    f"Required {name} artifact is missing; the server is misconfigured.",
                    stage="load_artifacts",
                )

        try:
            with open(scaler_path, "rb") as fin:
                self.scaler = pickle.load(fin)
            with open(encoder_path, "rb") as fin:
                self.label_encoder = pickle.load(fin)

            num_classes = len(self.label_encoder.classes_)
            input_features = int(getattr(self.scaler, "n_features_in_", FEATURE_DIM))
            self.device = device or select_device()

            if config_path and Path(config_path).exists():
                with open(config_path) as fin:
                    config = json.load(fin)
                config.setdefault("num_classes", num_classes)
                config.setdefault("input_features", input_features)
                model = build_model(config)
                arch = config.get("model_class")
                # Feature params must match training for self-consistency.
                self.hop_length = int(config.get("hop_length", DEFAULT_HOP_LENGTH))
                self.feat_smooth = int(config.get("feat_smooth", 0))
                self.is_sequence = arch in SEQUENCE_MODELS
                self.seq_length = int(config.get("seq_length", 0))
            else:
                model = CRNNChordModel(
                    input_features=input_features,
                    num_classes=num_classes,
                    hidden_size=CHECKPOINT_HIDDEN_SIZE,
                )
                arch = "CRNNChordModel (legacy)"

            state = torch.load(model_path, map_location="cpu")
            if hasattr(state, "state_dict"):
                state = state.state_dict()
            model.load_state_dict(state)
            model.to(self.device).eval()

            self.model = model
            self._loaded = True
            logger.info(
                "InferenceEngine loaded: %s, %d classes, %d features, device=%s",
                arch, num_classes, input_features, self.device,
            )
        except ProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                "Failed to load model artifacts.",
                stage="load_artifacts",
                original_error=exc,
            ) from exc

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise ProcessingError(
                "Inference engine is not initialized.", stage="load_artifacts"
            )

    # -- pipeline stages -------------------------------------------------

    def extract_chromagram(self, audio_path: str):
        """Returns (features[n,24], timestamps[n], duration)."""
        return extract_chroma24(
            audio_path, hop_length=self.hop_length, smooth_window=self.feat_smooth
        )

    def scale_features(self, features: np.ndarray) -> np.ndarray:
        try:
            return self.scaler.transform(features).astype(np.float32, copy=False)
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                "Failed to scale features.", stage="feature_scaling", original_error=exc
            ) from exc

    def run_inference(self, scaled: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Per-frame class indices for the whole clip."""
        self._ensure_loaded()
        if len(scaled) == 0:
            return np.array([], dtype=int)
        if self.is_sequence:
            return self._run_sequence_inference(scaled)
        preds: List[np.ndarray] = []
        try:
            with torch.no_grad():
                for start in range(0, len(scaled), batch_size):
                    chunk = torch.as_tensor(
                        scaled[start:start + batch_size],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    with self._lock:
                        logits = self.model(chunk)
                    preds.append(logits.argmax(dim=1).cpu().numpy())
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                "Model inference failed.", stage="model_inference", original_error=exc
            ) from exc
        return np.concatenate(preds)

    def _run_sequence_inference(self, scaled: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """
        Sliding-window prediction for sequence models (e.g. ChromaTransformer).

        Each window predicts one chord, assigned to its center frame; per-frame labels
        come from a majority vote of overlapping windows, with edges filled from the
        nearest decided frame. Returns one class index per input frame.
        """
        seq = self.seq_length
        n = len(scaled)
        # Pad short clips so at least one full window exists.
        work = scaled
        if n < seq:
            work = np.concatenate([scaled, np.repeat(scaled[-1:], seq - n, axis=0)])
        num_windows = len(work) - seq + 1
        votes: List[List[int]] = [[] for _ in range(len(work))]
        try:
            with torch.no_grad():
                for start in range(0, num_windows, batch_size):
                    idxs = list(range(start, min(start + batch_size, num_windows)))
                    batch = np.stack([work[i:i + seq] for i in idxs]).astype(np.float32)
                    tensor = torch.as_tensor(batch, device=self.device)
                    with self._lock:
                        logits = self.model(tensor)
                    window_preds = logits.argmax(dim=1).cpu().numpy()
                    for local, i in enumerate(idxs):
                        votes[i + seq // 2].append(int(window_preds[local]))
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                "Model inference failed.", stage="model_inference", original_error=exc
            ) from exc

        preds = np.full(len(work), -1, dtype=int)
        for f, frame_votes in enumerate(votes):
            if frame_votes:
                vals, counts = np.unique(frame_votes, return_counts=True)
                preds[f] = int(vals[int(np.argmax(counts))])

        decided = np.where(preds != -1)[0]
        if len(decided) == 0:
            preds[:] = 0
        else:
            preds[: decided[0]] = preds[decided[0]]  # backfill leading edge
            last = preds[decided[0]]
            for f in range(decided[0], len(preds)):
                if preds[f] == -1:
                    preds[f] = last
                else:
                    last = preds[f]
        return preds[:n]

    def decode_predictions(
        self, class_indices: np.ndarray, timestamps: np.ndarray, duration: float
    ) -> List[Annotation]:
        """
        Merge consecutive identical predictions into segments that tile the whole
        clip with no gaps or overlaps (design Property 6).
        """
        if len(class_indices) == 0:
            return []
        labels = self.label_encoder.inverse_transform(class_indices)

        annotations: List[Annotation] = []
        seg_start = float(timestamps[0])
        seg_label = labels[0]
        for i in range(1, len(labels)):
            if labels[i] != seg_label:
                seg_end = float(timestamps[i])
                annotations.append((seg_start, seg_end, str(seg_label)))
                seg_start = seg_end
                seg_label = labels[i]

        final_end = float(max(duration, float(timestamps[-1])))
        if final_end <= seg_start:
            final_end = seg_start + 1e-3  # guarantee start < end
        annotations.append((seg_start, final_end, str(seg_label)))
        return annotations

    def save_annotations(self, annotations: List[Annotation], output_path: str) -> None:
        """Write a .lab file: 'start end label' with 4-decimal timestamps."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fout:
            for start, end, label in annotations:
                fout.write(f"{start:.4f} {end:.4f} {label}\n")

    # -- orchestration ---------------------------------------------------

    def process_audio(
        self, audio_path: str, progress_cb: Optional[ProgressCallback] = None
    ) -> Tuple[List[Annotation], float]:
        """Full pipeline; returns (annotations, duration_seconds)."""
        self._ensure_loaded()

        def report(pct: int, stage: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(pct, stage)
                except Exception:  # noqa: BLE001 - progress reporting must never break inference
                    logger.debug("progress callback failed", exc_info=True)

        report(5, "Loading audio")
        features, timestamps, duration = self.extract_chromagram(audio_path)
        report(40, "Extracting features")
        scaled = self.scale_features(features)
        report(55, "Scaling features")
        class_indices = self.run_inference(scaled)
        class_indices = majority_smooth(class_indices, self.smoothing_window)
        report(85, "Decoding predictions")
        annotations = self.decode_predictions(class_indices, timestamps, duration)
        report(100, "Complete")
        return annotations, duration
