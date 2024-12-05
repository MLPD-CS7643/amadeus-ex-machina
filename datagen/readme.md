# Chord Synthesis Pipeline

A Python-based pipeline for generating synthetic chord audio samples used in deep learning and data science applications, particularly for chord transcription tasks.

## Overview

The pipeline consists of three main components:

- `chordgen.py`: Defines chord classes and generates .wav files using library of .sf2 soundfonts
- `fxgen.py`: Applies audio effects to base .wav files for data augmentation
- `pedal_board.py`: Defines and implements the effects used by fxgen.py

## Getting Started

### Chordgen

1. Make sure your environment packages are up to date with the ones defined in `environment.yaml`

2. Import to your notebook using `from datagen.chordgen import generate_all_chords`

3. Choose an output folder, you can set `make_dir=True` when calling `generate_all_chords` to create the folder but you must be aware of your working directory location
    - You may want to pass an absolute path to `out_dir` to be safe

4. If this is your first time running, you will need to download the soundfonts by setting `download_sf2=True` when calling `generate_all_chords`
    - For this to work, your `{project_root}/secrets/gdrive.json` file must be present and up to date (see `#links-only` channel)
    - .sf2 files will be saved to `{out_dir}/sf2/`
    - Set `download_sf2=False` afterwards

5. Call `generate_all_chords` 
    - Example: `generate_all_chords(out_dir="/data/audio/chords", download_sf2=True)`
    - By default chords are generated only for the 4th octave - this can be configured with the `start_octave` and `end_octave` params (recommend no lower than 1 and no higher than 6)
    - Audio will be saved to `{out_dir}/wav/` with `chord_ref.json` lookup table saved to `{out_dir}/`

6. Load `chord_ref.json` output to use as reference for generating your dataset
    - keyed with unique id for each chord
    - Example of how you might want to use it:
    ```python
    chord_table = json.loads(f"{out_dir}/chord_ref.json")
    for key, obj in chord_table.items():
      # ex key: Cmaj7_O4_piano
      billboard_chord = obj["billboard_notation"] # ex: C:maj7
      #do something with ground truth
      filename = obj["filename"] #: ex Cmaj7_O4_piano.wav
      fileformat = obj["format"] # ex: wav
      path_to_audio_file = out_dir / fileformat / filename # using Pathlib notation
      audio = your_audio_loader.load(path_to_audio_file)
      # do something with audio
    ```
    - other useful subkeys: `root` `chord_class` `octave` `instrument`


### FXGen

- NOTE: Effects integration WIP, recommend use chords for now
- Currently, to apply effects please use `fxgen.py`. Some fx are excluded from the pipeline and some parameters will result in not so great results. 
- fxgen uses a bank of presets to randomly select parameter and effect combinations from the list of working effects to create random processing chains for each chord input.  

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
