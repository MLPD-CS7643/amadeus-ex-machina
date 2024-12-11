# Investigating Timbral Bias in Chord Recognition Models

## Overview
This experiment investigates potential timbral biases in deep learning models trained for chord recognition tasks. Inspired by the paper "IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE", we explore whether chord recognition models exhibit similar biases towards timbre rather than fundamental chord structure.

## Hypothesis
Chord recognition models may develop biases based on common timbral associations in music genres. For example:
- Power chords may be more readily recognized with distorted guitar timbres (metal/rock)
- Extended chords may be better recognized with clean, chorus/reverb-affected timbres (jazz)
- Basic triads may show bias towards synth/piano timbres (pop/electronic)

## Dataset Generation
The experiment uses procedurally generated datasets with controlled timbral characteristics to test these biases.

### Training Dataset (2000 samples)
Intentionally biased dataset split into three genre-based groups:
1. Metal-style samples (40%):
   - Chord types: Power chords, perfect 5ths, sus4 chords
   - Instruments: Overdriven/distorted electric guitars
   - Effects: Heavy distortion, room noise, minimal reverb

2. Pop-style samples (35%):
   - Chord types: Major/minor triads, sus2/4 chords, dominant 7ths
   - Instruments: Synthesizers, electric pianos, clean guitars
   - Effects: Plate reverb, chorus, subtle drive

3. Jazz-style samples (25%):
   - Chord types: maj7, min7, extended chords (9, 11, 13)
   - Instruments: Acoustic piano, nylon guitar, jazz guitar
   - Effects: Hall reverb, subtle chorus/delay

### Validation Dataset (500 samples)
Maintains the same biased distribution as the training set to monitor training progress.

### Test Dataset (1000 samples)
Balanced dataset featuring:
- All chord types with all timbral combinations
- Even distribution of instruments and effects
- Designed to expose any learned biases

## Implementation Details

### Dependencies
- Python 3.8+
- PyTorch
- torchaudio
- Custom modules:
  - chordgen: Chord generation and synthesis
  - fxgen_torch: Audio effect processing
  - Effect modules: Distortion, Chorus, Reverb, Noise

### Directory Structure
```
chord_datasets/
├── train/
│   ├── processed/
│   └── dataset_metadata.json
├── val/
│   ├── processed/
│   └── dataset_metadata.json
└── test/
    ├── processed/
    └── dataset_metadata.json
```

### Metadata Format
Each audio file includes metadata with:
- Root note
- Chord class
- Billboard notation
- Octave
- Instrument
- GM preset ID
- Effect chain details
- Audio specifications (format, duration, sample rate, bit depth)

## Experiment Design Notes

### Genre-Timbre Associations
The dataset intentionally reinforces common genre-timbre associations:
- Metal: Emphasizes high-gain timbres with power chord structures
- Pop: Clean, processed timbres with basic chord structures
- Jazz: Warm, clean timbres with complex chord structures

### Testing for Bias
The test set deliberately breaks these associations to reveal if models have learned to rely on timbral cues rather than harmonic content. Examples include:
- Clean power chords
- Distorted extended chords
- Basic triads with jazz voicings

## Future Improvements
Potential enhancements to consider:
1. More sophisticated jazz chord voicing selection
2. Weighted probability distributions for effect chains
3. Extended metadata for genre grouping analysis
4. Data augmentation via pitch shifting

## Usage
[Implementation details and usage instructions to be added after code completion]

## References
- "IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE" (Geirhos et al.)
- [Additional references to be added]