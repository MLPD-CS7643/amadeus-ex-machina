# amadeus-train

Training, synthetic-data generation, and hyperparameter search for the amadeus chord
models. Depends on `amadeus-core` for the shared feature/model/chord code, so training
features are identical to what the API uses at inference time.

## Train the deployed model (GuitarSet chroma)

```bash
just train --download --no-leadsheet --vocab majmin --arch transformer
# quick smoke run:
just train --max-tracks 4 --epochs 1 --no-leadsheet
```

Artifacts are written to the repo-level `artifacts/guitarset_chroma/` where the API
loads them by default.

## Contents

- `amadeus_train.cli` / `amadeus_train.guitarset` — the live GuitarSet training pipeline.
- `amadeus_train.datasets` — `MirDataProcessor` / `ChordDataProcessor` (Billboard + audio).
- `amadeus_train.solver` — the training `Solver` used by the hyperparameter search.
- `amadeus_train.griddy` — Optuna-based hyperparameter search.
- `amadeus_train.datagen` — synthetic chord-audio + effects for the texture-bias study.
