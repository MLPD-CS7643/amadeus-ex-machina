"""
True NNLS ``bothchroma`` feature extraction (the McGill-Billboard representation).

The Billboard-trained model + its ``MinMaxScaler`` were fit on the 24-dim NNLS
``bothchroma`` features that ship with the McGill-Billboard dataset, produced by Queen
Mary's **NNLS Chroma** Vamp plugin. ``features/chroma.py`` can only *approximate* those
with band-limited CQT chromagrams; this module reproduces them exactly by calling the
same plugin, so there is **no train/serve feature mismatch** for the Billboard model.

Parity is by construction: same plugin, same parameters (the plugin defaults McGill used),
same framing (stepSize 2048 / blockSize 16384 at 44100 Hz -> ~0.0464 s/frame), same
un-normalized magnitude scale, and the same bass(12)|treble(12) bin order (note order
starts at A, matching :data:`amadeus_core.features.chroma.BASS_FIRST`).

This is an **optional backend**. It needs the ``vamp`` Python host (``uv sync --extra
nnls`` with the Vamp SDK on the include path) plus the ``nnls-chroma`` plugin installed in
a Vamp plugin directory (``~/Library/Audio/Plug-Ins/Vamp`` on macOS). librosa remains the
default backend; this is only used when a bundle/config selects ``backend="nnls"``.
"""

import numpy as np

from amadeus_core.errors import ProcessingError

# McGill-Billboard extraction settings. These match the NNLS Chroma plugin defaults, i.e.
# the canonical Billboard "bothchroma" features. Frame period = STEP / SAMPLE_RATE.
NNLS_SAMPLE_RATE = 44100
NNLS_STEP = 2048
NNLS_BLOCK = 16384
NNLS_PLUGIN_KEY = "nnls-chroma:nnls-chroma"
NNLS_OUTPUT = "bothchroma"

# Explicit parameter dict == the plugin defaults McGill used. Passed explicitly so parity
# is documented and robust to any future change in the plugin's own defaults.
#   useNNLS=1        NNLS approximate transcription (vs. plain chroma)
#   rollon=0         no bass roll-on high-pass
#   tuningmode=0     global tuning estimate
#   whitening=1      spectral whitening on
#   s=0.7            whitening spectral-shape parameter
#   chromanormalize=0  no per-frame chroma normalization (keeps the ~0-4.5 NNLS scale)
NNLS_PARAMETERS: dict[str, float] = {
    "useNNLS": 1.0,
    "rollon": 0.0,
    "tuningmode": 0.0,
    "whitening": 1.0,
    "s": 0.7,
    "chromanormalize": 0.0,
}

FEATURE_DIM = 24
_STAGE = "chromagram_extraction"

_INSTALL_HINT = (
    "NNLS feature backend unavailable. Install the Python host with "
    "`uv sync --all-packages --extra nnls` (needs the Vamp SDK: `brew install "
    "vamp-plugin-sdk`) and place the `nnls-chroma` Vamp plugin in a Vamp plugin "
    "directory (e.g. ~/Library/Audio/Plug-Ins/Vamp on macOS)."
)


def nnls_backend_available() -> bool:
    """
    True if the NNLS backend is usable: the ``vamp`` host is importable and the
    ``nnls-chroma`` plugin is installed in a Vamp plugin directory. Cheap enough to probe
    at app startup (so a server can fall back to the librosa model if it's missing).
    """
    try:
        import vampyhost  # the C host extension that `vamp` builds on

        return NNLS_PLUGIN_KEY in vampyhost.list_plugins()
    except Exception:
        return False


def nnls_bothchroma(
    y: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute true NNLS ``bothchroma`` (24-dim bass|treble) from an in-memory signal.

    Args:
        y: Mono audio samples.
        sr: Sample rate of ``y`` (resampled to 44100 Hz internally to match McGill).

    Returns:
        ``(features, timestamps)`` where ``features`` is ``(n_frames, 24)`` float32
        un-normalized NNLS magnitudes (bass 12 then treble 12, note order from A) and
        ``timestamps`` is ``(n_frames,)`` seconds at each frame's start.

    Raises:
        ProcessingError: if the signal is empty, the plugin/host is unavailable, or
            feature extraction fails.
    """
    if y is None or np.size(y) == 0:
        raise ProcessingError("Audio file contains no samples.", stage="audio_load")

    try:
        import vamp  # lazy: optional dependency
    except Exception as exc:  # noqa: BLE001
        raise ProcessingError(
            _INSTALL_HINT, stage=_STAGE, original_error=exc
        ) from exc

    # The plugin's framing (0.0464 s/frame) assumes 44100 Hz; resample to match McGill.
    y = np.asarray(y, dtype=np.float32)
    if sr != NNLS_SAMPLE_RATE:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=NNLS_SAMPLE_RATE).astype(
            np.float32, copy=False
        )

    try:
        result = vamp.collect(
            y,
            NNLS_SAMPLE_RATE,
            NNLS_PLUGIN_KEY,
            output=NNLS_OUTPUT,
            parameters=NNLS_PARAMETERS,
            step_size=NNLS_STEP,
            block_size=NNLS_BLOCK,
        )
    except Exception as exc:  # noqa: BLE001
        # Most commonly: the nnls-chroma plugin isn't installed in the Vamp path.
        raise ProcessingError(_INSTALL_HINT, stage=_STAGE, original_error=exc) from exc

    matrix = result.get("matrix") if isinstance(result, dict) else None
    if matrix is None:
        raise ProcessingError(
            "NNLS plugin returned no 'bothchroma' matrix output.", stage=_STAGE
        )
    step_time, features = matrix
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ProcessingError(
            f"Expected {FEATURE_DIM}-dim NNLS features, got shape {features.shape}.",
            stage=_STAGE,
        )

    step = float(step_time)
    timestamps = np.arange(features.shape[0], dtype=np.float64) * step
    return features, timestamps
