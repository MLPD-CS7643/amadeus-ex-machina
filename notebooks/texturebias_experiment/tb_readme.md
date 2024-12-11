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
1. Metal-style samples (33%):
   - Primary chords: Power chords, perfect 5ths, sus4 chords, dim
   - Secondary chords: maj, min7, 7, aug (used in prog metal)
   - Instruments: Overdriven/distorted electric guitars
   - Effects: Classic distortion/fuzz, room noise, small room reverb

2. Pop-style samples (34%):
   - Primary chords: Major/minor triads, sus2/4, maj6, dominant 7ths
   - Secondary chords: maj7, min7, maj(9), min(9)
   - Instruments: Synthesizers, electric pianos, clean guitars
   - Effects: Plate reverb, classic/subtle chorus, subtle drive

3. Jazz-style samples (33%):
   - Primary chords: maj7, min7, extended chords (9, 11, 13), hdim7, dim7
   - Secondary chords: 7(#9), sus4(b7,9), aug, min11
   - Instruments: Acoustic piano, nylon guitar, jazz guitar
   - Effects: Hall reverb, subtle chorus/delay

### Validation Dataset (500 samples)
Maintains the same biased distribution as the training set to monitor training progress.

### Test Dataset
Balanced dataset featuring:
- Every chord type processed with each genre's timbral characteristics
- Comprehensive coverage of all possible chord-timbre combinations
- There are currently 50 combinations that are expressed for every chord type 

## Implementation Details

### Dependencies
- Python 3.8+ (I am running 3.11)
- PyTorch
- torchaudio
- mido and a bunch of other stuff, it's everything that's required for chordgen, fxgen, the pedals, and the notebook itself
- Custom modules:
  - chordgen: Chord generation and synthesis
  - fxgen_torch: Audio effect processing
  - Effect modules: Distortion, Chorus, Reverb, Noise

### Audio Processing Pipeline
1. MIDI Generation
   - Generates chord progressions using specified intervals
   - Controls note duration and velocity
   
2. Audio Synthesis
   - Converts MIDI to audio using FluidSynth soundfonts
   - Genre-appropriate instrument selection
   
3. Effect Processing
   - Direct application of effect chains using custom modules
   - Genre-specific effect combinations

4. Feature Extraction
   - Conversion to spectrograms using torchaudio
   - Chromagram generation for harmonic analysis
   - Optional visualization functionality

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
- Applied effects chain
- Genre association
- Audio specifications (format, duration, sample rate, bit depth)

## Feature Processing
The experiment includes functionality for:
1. Converting audio to spectrograms:
   - Using torchaudio's built-in transforms
   - Configurable FFT and hop length parameters

2. Generating chromagrams:
   - Using torchaudio.prototype.transforms.ChromaSpectrogram
   - 12 pitch classes
   - Configurable for different analysis requirements

3. Visualization options:
   - Spectrogram plots
   - Chromagram visualizations
   - Configurable color maps and scaling

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

### Validation Requirements
The test set ensures:
- Every chord type appears with every genre's timbral characteristics
- Minimum instrument and effect coverage per genre
- Balanced representation across all combinations

## Future Improvements
Potential enhancements to consider:
1. More sophisticated jazz chord voicing selection
2. Weighted probability distributions for effect chains
3. Extended metadata for genre grouping analysis
4. Data augmentation via pitch shifting
5. Additional audio feature extraction methods
6. Enhanced visualization capabilities

## References
- "IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE" (Geirhos et al.)
- [Additional references to be added]