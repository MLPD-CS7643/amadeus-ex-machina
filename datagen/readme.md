# Chord Synthesis Pipeline

A Python-based pipeline for generating synthetic chord audio samples used in deep learning and data science applications, particularly for chord transcription tasks.

## Overview

The pipeline consists of three main components:

- `chordgen.py`: Generates MIDI files for various chord types across octaves
- `wavgen.py`: Converts MIDI files to WAV using SoundFont synthesis
- `pedal_board.py`: Applies audio effects for data augmentation

## Getting Started

1. Install dependencies:
```bash
pip install mido librosa numpy scipy soundfile tinysoundfont #TODO: upgrade to a requirements.txt
```

2. Set up directory structure:
```bash
mkdir -p ./chordgen/{wav,sf2} #Does this work automatically if they don't exist?
mkdir -p ./fx_out/{reverb,delay,distort,compress,chorus}
```

3. Download SoundFonts by running:
```python
from datagen.chordgen import generate_all_chords
generate_all_chords(download_sf2=True)
```

4. Generate chord samples:
```python
# Generate chords for octaves 3-5
generate_all_chords(start_octave=2, end_octave=4) #reccomend using no lower than 1 and no higher than 5
```

5. Apply effects:
Currently, to apply effects please use samplegen.py. Some fx are excluded from the pipeline and some parameters will result in not so great results. 
Samplegen uses a bank of presets to randomly select parameter and effect combinations from the list of working effeccts to create random processing chains for each chord input.  

## Technical Details

### Chord Generation
- Defines 16 chord types including major, minor, diminished, augmented, and extended chords
- Generates MIDI files using specified intervals from root notes
- Supports octave ranges 0-7
- Creates comprehensive metadata in JSON format

### Audio Synthesis
- Uses tinysoundfont for MIDI to audio conversion
- Supports multiple SoundFonts for timbre variation
- Generates 16-bit stereo WAV files at 44.1kHz
- Processes files in batches for efficiency

### Effects Processing
Available effects:
- Reverb 
- Delay 
- Distortion 
- Chorus
- Noise

  See the spec for each effect in the "pedals" folder. Note that there are additional pedlas that are unused, these are currently bugged so avoid including them. 

## Experimental Approach

The pipeline creates a diverse dataset through:
1. Systematic chord generation across multiple octaves
2. Timbre variation using different SoundFonts
3. Audio effect augmentation for acoustic diversity
4. Consistent metadata tracking for experiment reproducibility

## Output Structure

```
project/
├── chordgen/
│   ├── wav/          # Raw synthesized samples
│   └── sf2/          # SoundFont files
├── fx_out/           # Processed audio
│   ├── reverb/
│   ├── delay/
│   └── ...
└── chord_ref.json    # Sample metadata
```
