# amadeus-core

The shared chord-recognition library. Owns the pieces that **both** the web API and
the training package must agree on, so there is a single source of truth:

- `amadeus_core.features` — audio → 24-dim chroma (`chroma24_from_audio`, `extract_chroma24`).
- `amadeus_core.models` — the model zoo (`ChromaTransformerModel`, `ChromaMLP`,
  `CRNNChordModel`, plus research nets) and `build_model` / the model registry.
- `amadeus_core.chords` — chord vocabulary, label remapping, maj/min folding, `mir_eval` scoring.
- `amadeus_core.artifacts` — load/save the artifact bundle (`model.pt` + scaler + encoder + config).
- `amadeus_core.inference` — framework-agnostic `InferenceEngine` (audio → `.lab`).
- `amadeus_core.device` — device selection + seeding.

Nothing here imports the web framework or the training stack; both depend on this package.
