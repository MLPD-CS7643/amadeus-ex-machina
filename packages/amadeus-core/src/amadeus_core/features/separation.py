"""
Optional harmonic source separation, applied to the waveform before chroma extraction.

The shipped model was trained on clean, single-instrument harmonic audio, so a full band
mix (vocals + drums + bass + effects) sits well outside its domain -- a major source of
wrong-chord predictions on produced songs. Separating out the harmonic / accompaniment
content first pulls a produced track back toward that training domain. Because train-time
and inference-time features must match, this lives in core and is shared by both.

Methods:
  ``none``   -- passthrough (default; preserves the original behaviour).
  ``hpss``   -- librosa harmonic/percussive separation. Cheap, always available; strips
                drum transients (which smear energy across every pitch class) but not
                vocals.
  ``demucs`` -- Demucs stems, keeping ``bass`` + ``other`` and dropping vocals and drums.
                Best quality on full mixes; needs the optional ``demucs`` dependency.
"""

import numpy as np

from amadeus_core.errors import ProcessingError

SEPARATION_METHODS = ("none", "hpss", "demucs")
_STAGE = "source_separation"


def separate_harmonic(y: np.ndarray, sr: int, method: str = "none") -> np.ndarray:
    """Return the harmonic/accompaniment component of ``y`` per ``method``."""
    if y is None or np.size(y) == 0 or method in (None, "", "none"):
        return y
    if method == "hpss":
        return _hpss(y)
    if method == "demucs":
        return _demucs(y, sr)
    raise ProcessingError(
        f"Unknown separation method {method!r}; expected one of {SEPARATION_METHODS}.",
        stage=_STAGE,
    )


def _hpss(y: np.ndarray) -> np.ndarray:
    import librosa

    harmonic = librosa.effects.hpss(y)[0]
    return harmonic.astype(np.float32, copy=False)


def _demucs(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        import librosa
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as exc:  # noqa: BLE001 - optional heavy dependency
        raise ProcessingError(
            "Demucs separation was requested but the optional 'demucs' dependency is "
            "not installed. Install it with `uv sync --all-packages --extra separation`.",
            stage=_STAGE,
            original_error=exc,
        ) from exc

    try:
        model = get_model("htdemucs")
        model.eval()
        wav = np.asarray(y, dtype=np.float32)
        if sr != model.samplerate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=model.samplerate)
        # demucs wants (batch, channels=2, samples); upmix mono -> stereo.
        tensor = torch.from_numpy(wav)[None, None, :].repeat(1, 2, 1)
        with torch.no_grad():
            stems = apply_model(model, tensor, progress=False)[0]  # (sources, 2, T)
        keep = [i for i, name in enumerate(model.sources) if name in ("bass", "other")]
        accompaniment = stems[keep].sum(dim=0).mean(dim=0)  # sum kept stems -> mono
        out = accompaniment.cpu().numpy().astype(np.float32)
        if sr != model.samplerate:
            out = librosa.resample(out, orig_sr=model.samplerate, target_sr=sr)
        return out
    except ProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ProcessingError(
            "Demucs source separation failed.", stage=_STAGE, original_error=exc
        ) from exc
