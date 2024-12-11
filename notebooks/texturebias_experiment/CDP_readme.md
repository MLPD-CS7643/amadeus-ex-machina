# Technical Documentation for Chord Recognition Experiment

## Data Processing Pipeline
Detailed documentation for preprocessing chord data and preparing it for model training.

### ChordDataProcessor Class
The `ChordDataProcessor` class handles dataset preprocessing and transforms audio into formats suitable for deep learning models. It supports:
- Chromagram generation
- Spectrogram generation
- Multiple JSON metadata formats
- Batch processing
- Sequential or individual sample processing

### Requirements
```bash
pip install torch torchaudio numpy scikit-learn
```

### Basic Usage

```python
from pathlib import Path

# Initialize processor
processor = ChordDataProcessor(
    chord_json_path=Path("data/chords.json"),
    batch_size=64,
    seq_length=16,
    device="cuda",
    mode="chroma",
    json="keyed",
    audio_path=Path("data/audio"),
)

# Process data and build loaders
train_loader, test_loader, num_classes = processor.process_all_and_build_loaders(
    target_features_shape=(None, None, 12, 128),  # For chromagrams
    test_size=0.2,
    random_state=42,
)
```

### Supported Metadata Formats

#### Keyed JSON Format
```json
{
    "Fmaj7": {
        "filename": "Fmaj7.mp3",
        "billboard_notation": "F:maj7"
    },
    "Cmin": {
        "filename": "Cmin.mp3",
        "billboard_notation": "C:min"
    }
}
```

#### Entries JSON Format
```json
[
    {
        "processed_path": "train/Fmaj7_processed.mp3",
        "billboard_notation": "F:maj7"
    },
    {
        "processed_path": "test/Cmin_processed.mp3",
        "billboard_notation": "C:min"
    }
]
```

### Configuration Options

1. **Initialization Parameters**
   - `chord_json_path`: Path to JSON metadata
   - `batch_size`: DataLoader batch size (default: 64)
   - `seq_length`: Sequence length for sequential processing (default: 16)
   - `device`: Processing device ("cpu" or "cuda")
   - `process_sequential`: Sequential or individual processing
   - `mode`: Feature type ("chroma" or "spectrogram")
   - `json`: Metadata format ("keyed" or "entries")
   - `audio_path`: Base path for audio files

2. **Processing Parameters**
   - `target_features_shape`: Expected feature dimensions
   - `target_labels_shape`: Expected label dimensions
   - `test_size`: Train/test split ratio
   - `random_state`: Random seed for reproducibility

### Common Issues and Solutions

1. **File Path Issues**
   - Ensure `audio_path` points to correct parent directory
   - Check relative paths in metadata JSON
   - Verify file extensions match

2. **Dimension Mismatches**
   - Verify `target_features_shape` matches model input
   - Check batch size compatibility
   - Ensure consistent sequence lengths

3. **Metadata Issues**
   - Validate JSON format matches specified type
   - Check for missing required keys
   - Verify billboard notation format

### Example Training Workflow

```python
# 1. Initialize processor
processor = ChordDataProcessor(
    chord_json_path="data/chords.json",
    mode="chroma",
    json="keyed",
    audio_path="data/audio",
)

# 2. Process data
chord_train_loader, chord_test_loader, chord_num_classes = training_data_processor.process_all_and_build_loaders(
    target_features_shape=None,
    target_labels_shape=None,
    test_size=0.2,
    random_state=42,
)

# 3. Initialize model and training components
model = CRNNModel(
    input_features=12,
    num_classes=num_classes,
    hidden_size=128,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 4. Set up solver
solver = Solver(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    train_dataloader=train_loader,
    valid_dataloader=test_loader,
    batch_size=64,
    epochs=10,
    device="cuda",
    early_stop_epochs=3,
)

# 5. Train and evaluate
solver.train_and_evaluate(plot_results=True)
#or maybe
solver.run_inference 
#idk how solver works lol
```
