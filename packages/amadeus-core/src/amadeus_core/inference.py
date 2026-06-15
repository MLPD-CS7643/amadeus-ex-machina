"""
InferenceEngine: orchestrates audio -> chord-annotation inference.

Pipeline:

    audio -> extract_chroma24 -> scaler.transform -> model (batched / sliding-window)
          -> argmax -> majority smooth -> label_encoder.inverse_transform
          -> merge consecutive labels into gap-free segments -> .lab

Framework-agnostic: a single engine instance is shared across worker threads; the model
runs in ``eval()`` and forward passes are serialized with a lock for thread-safety.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from amadeus_core.artifacts import ArtifactBundle, load_bundle
from amadeus_core.errors import ProcessingError
from amadeus_core.features.chroma import DEFAULT_HOP_LENGTH, extract_chroma24

logger = logging.getLogger(__name__)

# Default majority-vote smoothing window (frames) over per-frame predictions.
# ~9 frames at hop 2048 / sr 22050 ~ 0.8s; odd number; <=1 disables.
DEFAULT_SMOOTHING_WINDOW = 9

# (start_time, end_time, chord_label)
Annotation = tuple[float, float, str]
ProgressCallback = Callable[[int, str], None]


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


class InferenceEngine:
    def __init__(self, smoothing_window: int = DEFAULT_SMOOTHING_WINDOW) -> None:
        self.bundle: ArtifactBundle | None = None
        self.smoothing_window = smoothing_window
        self.hop_length = DEFAULT_HOP_LENGTH
        self.feat_smooth = 0
        self.is_sequence = False
        self.seq_length = 0
        self._lock = threading.Lock()

    # -- loading ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    @property
    def model(self) -> torch.nn.Module | None:
        return self.bundle.model if self.bundle else None

    @property
    def device(self) -> torch.device | None:
        return self.bundle.device if self.bundle else None

    def attach_bundle(self, bundle: ArtifactBundle) -> None:
        """Adopt an already-loaded :class:`ArtifactBundle`."""
        self.bundle = bundle
        self.hop_length = bundle.hop_length
        self.feat_smooth = bundle.feat_smooth
        self.is_sequence = bundle.is_sequence
        self.seq_length = bundle.seq_length
        logger.info(
            "InferenceEngine loaded: %s, %d classes, %d features, device=%s",
            bundle.arch,
            bundle.num_classes,
            bundle.input_features,
            bundle.device,
        )

    def load_artifacts(
        self,
        model_path: str,
        scaler_path: str,
        encoder_path: str,
        config_path: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Load model weights + scaler + label encoder from explicit paths."""
        self.attach_bundle(
            load_bundle(
                model_path=model_path,
                scaler_path=scaler_path,
                encoder_path=encoder_path,
                config_path=config_path,
                device=device,
            )
        )

    def load_from_dir(
        self, artifact_dir: str, device: torch.device | None = None
    ) -> None:
        """Load a bundle from a directory using the standard artifact filenames."""
        self.attach_bundle(load_bundle(artifact_dir, device=device))

    def _ensure_loaded(self) -> None:
        if self.bundle is None:
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
        assert self.bundle is not None
        try:
            return self.bundle.scaler.transform(features).astype(np.float32, copy=False)
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                "Failed to scale features.", stage="feature_scaling", original_error=exc
            ) from exc

    def run_inference(self, scaled: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Per-frame class indices for the whole clip."""
        self._ensure_loaded()
        assert self.bundle is not None
        if len(scaled) == 0:
            return np.array([], dtype=int)
        if self.is_sequence:
            return self._run_sequence_inference(scaled)
        preds: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for start in range(0, len(scaled), batch_size):
                    chunk = torch.as_tensor(
                        scaled[start : start + batch_size],
                        dtype=torch.float32,
                        device=self.bundle.device,
                    )
                    with self._lock:
                        logits = self.bundle.model(chunk)
                    preds.append(logits.argmax(dim=1).cpu().numpy())
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                "Model inference failed.", stage="model_inference", original_error=exc
            ) from exc
        return np.concatenate(preds)

    def _run_sequence_inference(
        self, scaled: np.ndarray, batch_size: int = 2048
    ) -> np.ndarray:
        """
        Sliding-window prediction for sequence models (e.g. ChromaTransformer).

        Each window predicts one chord, assigned to its center frame; per-frame labels
        come from a majority vote of overlapping windows, with edges filled from the
        nearest decided frame. Returns one class index per input frame.
        """
        assert self.bundle is not None
        seq = self.seq_length
        n = len(scaled)
        # Pad short clips so at least one full window exists.
        work = scaled
        if n < seq:
            work = np.concatenate([scaled, np.repeat(scaled[-1:], seq - n, axis=0)])
        num_windows = len(work) - seq + 1
        votes: list[list[int]] = [[] for _ in range(len(work))]
        try:
            with torch.no_grad():
                for start in range(0, num_windows, batch_size):
                    idxs = list(range(start, min(start + batch_size, num_windows)))
                    batch = np.stack([work[i : i + seq] for i in idxs]).astype(
                        np.float32
                    )
                    tensor = torch.as_tensor(batch, device=self.bundle.device)
                    with self._lock:
                        logits = self.bundle.model(tensor)
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
    ) -> list[Annotation]:
        """
        Merge consecutive identical predictions into segments that tile the whole clip
        with no gaps or overlaps.
        """
        if len(class_indices) == 0:
            return []
        assert self.bundle is not None
        labels = self.bundle.label_encoder.inverse_transform(class_indices)

        annotations: list[Annotation] = []
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

    def save_annotations(self, annotations: list[Annotation], output_path: str) -> None:
        """Write a .lab file: 'start end label' with 4-decimal timestamps."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fout:
            for start, end, label in annotations:
                fout.write(f"{start:.4f} {end:.4f} {label}\n")

    # -- orchestration ---------------------------------------------------

    def process_audio(
        self, audio_path: str, progress_cb: ProgressCallback | None = None
    ) -> tuple[list[Annotation], float]:
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
