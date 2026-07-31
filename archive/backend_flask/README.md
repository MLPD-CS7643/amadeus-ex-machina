# Chord Recognition Backend

Flask REST API backend for the web-based chord recognition system.

## Setup

### Prerequisites

- Python 3.11
- PyTorch 2.5.1
- Trained CRNNModel checkpoint at `../models/checkpoints/CRNNModel_best_model.pth`
- Preprocessor artifacts at `../data/processed/scaler.pkl` and `../data/processed/label_encoder.pkl`

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Server

Development mode:
```bash
python -m flask --app app.main run --debug
```

Production mode with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app.main:app
```

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app --cov-report=html
```

Run specific test categories:
```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m property      # Property-based tests only
```

## Project Structure

```
backend/
├── app/                    # Application code
│   ├── __init__.py
│   ├── main.py            # Flask app factory
│   ├── routes.py          # API endpoints
│   ├── job_manager.py     # Job lifecycle management
│   ├── file_handler.py    # File operations
│   └── inference.py       # Model inference pipeline
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_routes.py
│   ├── test_job_manager.py
│   ├── test_file_handler.py
│   └── test_inference.py
├── config/                 # Configuration
│   ├── __init__.py
│   └── settings.py
├── requirements.txt        # Python dependencies
├── pytest.ini             # Pytest configuration
├── .env.example           # Environment variables template
└── README.md              # This file
```

## API Endpoints

### POST /api/upload
Upload an audio file for chord recognition.

**Request:**
- Content-Type: multipart/form-data
- Body: `file` (audio file)

**Response:**
```json
{
  "jobId": "uuid-string",
  "status": "queued",
  "message": "File uploaded successfully"
}
```

### GET /api/status/{job_id}
Get processing status for a job.

**Response:**
```json
{
  "jobId": "uuid-string",
  "status": "processing",
  "progress": 45,
  "estimatedTimeRemaining": 12,
  "message": "Extracting features"
}
```

### GET /api/download/{job_id}
Download chord annotation file (.lab format).

### GET /api/results/{job_id}
Get chord annotations as JSON for visualization.

**Response:**
```json
{
  "jobId": "uuid-string",
  "filename": "song.wav",
  "duration": 180.5,
  "annotations": [
    {
      "startTime": 0.0,
      "endTime": 2.5,
      "chordLabel": "C:maj"
    }
  ]
}
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

- `MODEL_PATH`: Path to trained PyTorch model
- `SCALER_PATH`: Path to fitted StandardScaler
- `ENCODER_PATH`: Path to fitted LabelEncoder
- `MAX_FILE_SIZE`: Maximum upload size in bytes (default: 50MB)
- `UPLOAD_DIR`: Directory for temporary file storage
- `MAX_CONCURRENT_JOBS`: Maximum concurrent processing jobs
- `JOB_TIMEOUT_HOURS`: Hours before job cleanup

## Development

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting
- Type hints for better code documentation

Format code:
```bash
black app/ tests/
```

Lint code:
```bash
flake8 app/ tests/
```

### Adding New Features

1. Write tests first (TDD approach)
2. Implement the feature
3. Ensure all tests pass
4. Update documentation

## Implementation notes (backend)

The backend is **self-contained** (no imports from the repo root). The deployed model
is a **`ChromaTransformerModel`, retrained on GuitarSet** (the team's best-performing
architecture). It's self-consistent: training features come from the exact same
`app/features.py::chroma24_from_audio` used at inference, so there's no train/inference
feature mismatch.

- **Model + artifacts.** Default artifacts live in
  `backend/artifacts/guitarset_chroma/` (`model.pt`, `scaler.pkl`,
  `label_encoder.pkl`, `model_config.json`). `model_config.json` names the
  architecture, sequence length, and feature params (`hop_length`, `feat_smooth`); the
  engine rebuilds the model via `app/model.py::build_model`. Two simpler models remain
  available: `ChromaMLP` (per-frame, `--arch mlp`) and a legacy `CRNNChordModel`
  fallback (used by the legacy integration test against the old Billboard checkpoint).
- **Features (24-dim chroma).** Two band-limited CQT chromagrams (bass ‖ treble).
  Optional temporal averaging (`feat_smooth`) is used by the per-frame MLP but **off
  for the transformer** (`feat_smooth=0`), which models temporal context itself across
  a 16-frame window.
- **Sequence inference.** For the transformer the engine slides a `seq_length` window
  (currently 24 frames ≈ 1.1 s, stride 1) over the chroma frames, predicts one chord
  per window, assigns it to the window's center frame, and majority-votes per frame.
  Per-frame models (MLP/CRNN) classify each frame directly. Either way, predictions are
  majority-smoothed and consecutive identical labels are merged into gap-free `.lab`
  segments.
- **Vocabulary.** MIREX **maj/min** (12 roots × {maj, min} + N = 24 classes) — the
  standard, learnable target. The model is **correct on clean chords** (verified 6/6:
  C/A/G/E/D/F maj/min, end-to-end through the API); held-out accuracy on real guitar
  comping is ~0.56 (a hard task — arpeggiation, transitions, partial chords). Tuning
  that got there: longer context window (24 frames), 12× transposition augmentation,
  AdamW + weight decay + label smoothing + LR plateau scheduling.

### Retraining the model

```bash
conda activate amadeus-ex-machina
# first run downloads GuitarSet (index + annotations + mono mix, ~0.5–1 GB)
# this is the deployed config (val ~0.56):
python ../scripts/train_guitarset_chroma.py --arch transformer --download \
    --no-leadsheet --vocab majmin --feat-smooth 0 \
    --seq-length 24 --stride 6 --weight-decay 1e-4 --label-smoothing 0.1 \
    --epochs 50 --patience 8 --batch-size 256
# subsequent runs reuse cached features (data/processed/guitarset_chroma_cache/)
```

Key flags: `--arch {transformer,mlp}`; `--seq-length`/`--stride` and
`--d-model/--num-layers/--dim-feedforward` (transformer); `--no-leadsheet` (performed
annotations — better audio alignment than the instructed leadsheet); `--vocab
{majmin,full}`; `--feat-smooth N` (per-frame temporal averaging, use 0 for the
transformer); `--augment/--no-augment` (12 chroma transpositions for key invariance);
`--max-tracks N`. Artifacts are written where the backend loads them by default.

### Running with the project conda env

The conda env `amadeus-ex-machina` already has the heavy ML stack (torch, librosa,
scikit-learn). Install only the web/test extras into it:

```bash
conda activate amadeus-ex-machina
pip install "flask>=3,<4" flask-cors python-dotenv hypothesis pytest-cov pytest-flask pytest-mock
pytest                                   # 84 tests (integration tests run when artifacts are present)
flask --app app.main run --port 5000     # serve the API
```

(`requirements.txt` pins a full standalone set incl. torch 2.5.1 for a fresh venv
or Docker; reusing the conda env avoids a multi-GB reinstall.)

## License

See main project LICENSE file.
