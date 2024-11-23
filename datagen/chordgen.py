import os
import json
from pathlib import Path
from mido import Message, MidiFile, MidiTrack
from zipfile import ZipFile
from datagen.wavgen import synthesize_to_wav
from utils.gdrive import download_from_gdrive


JSON_FILE = "chord_ref.json"
SF2_ARCHIVE = "sf2.zip"

BASE_DIR = "./chordgen"
WAV_DIR = "wav"
SF2_DIR = "sf2"

SAMPLE_RATE = 44100
BIT_DEPTH = 16

C0 = 12

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORDS = {
    "5": [0, 7],
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "9": [0, 4, 7, 10, 14],
    "7b9": [0, 4, 7, 10, 13]
}

def generate_all_chords(download_sf2:bool=False, start_octave:int=4, end_octave:int=4, out_dir=BASE_DIR):
    """
    Generates .wav files for all defined chords using all available SoundFonts.
    Also saves chord_ref.json lookup table with metadata.

    Args:
        download_sf2 (bool): download repository of SoundFonts
        start_octave (int): first octave (min 0)
        end_octave (int): last octave (max 7)
        out_dir (str): output directory
    
    Returns:
        None
    """
    base_path = Path(out_dir)
    sf2_path = base_path / SF2_DIR
    wav_path = base_path / WAV_DIR
    if download_sf2:
        __fetch_sf2_archive(sf2_path)
    soundfont_names = [f[:-4] for f in os.listdir(sf2_path) if f.endswith('.sf2')]
    if not soundfont_names:
        print(f"No SoundFonts found in {sf2_path}!")
        return
    os.makedirs(wav_path, exist_ok=True)
    json_out = {}
    print("Starting chord generation...")
    for octave in range(start_octave, end_octave+1):
        for i in range(12):
            root = C0 + octave * 12 + i
            for chord_class, intervals in CHORDS.items():
                midi = __generate_midi_chord(root, intervals)
                note_name = __note_lookup(root)
                mid_filename = f"{note_name}{chord_class}_O{octave}"
                mid_filepath = wav_path / f"{mid_filename}.mid"
                midi.save(mid_filepath)
                for sf_name in soundfont_names:
                    wav_filename = f"{mid_filename}_{sf_name}"
                    wav_filepath = wav_path / f"{wav_filename}.wav"
                    sf_filepath = sf2_path / f"{sf_name}.sf2"
                    synthesize_to_wav(str(mid_filepath.absolute()), str(sf_filepath.absolute()), str(wav_filepath.absolute()), sample_rate=SAMPLE_RATE, bit_depth=BIT_DEPTH)
                    json_out[wav_filename] = {
                        "root": note_name,
                        "chord_class": chord_class,
                        "billboard_notation": f"{note_name}:{chord_class}",
                        "octave": octave,
                        "instrument": sf_name,
                        "filename": f"{wav_filename}.wav",
                        "format": "wav",
                        "sample_rate": SAMPLE_RATE,
                        "bit_depth": BIT_DEPTH
                    }
                    print(wav_filename)
                os.remove(mid_filepath)
    print("Saving lookup table...")
    __save_json(json_out, base_path)
    print("Fin~")

def __generate_midi_chord(root_note:int, intervals:list, velocity=64, ticks=1920):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    for interval in intervals:
        track.append(Message("note_on", note=root_note + interval, velocity=velocity, time=0))
    track.append(Message("note_off", note=root_note + intervals[0], velocity=velocity, time=ticks))
    for interval in intervals[1:]:
        track.append(Message("note_off", note=root_note + interval, velocity=velocity, time=0))
    return midi

def __note_lookup(midi_note:int, include_octave=False):
    note_index = midi_note % 12  # Get the position of the note within the octave
    if include_octave:
        octave = (midi_note // 12) - 1  # Calculate the octave number
        return f"{NOTES[note_index]}{octave}"
    return f"{NOTES[note_index]}"

def __save_json(data:dict, path):
    dumps = json.dumps(data)
    os.makedirs(path, exist_ok=True)
    with open(path / JSON_FILE, 'w') as outfile:
        outfile.write(dumps)

def __fetch_sf2_archive(path:Path):
    print("Downloading sf2.zip...")
    dl_path = path / SF2_ARCHIVE
    download_from_gdrive(SF2_ARCHIVE, str(dl_path.absolute()))
    print("Extracting sf2.zip...")
    with ZipFile(dl_path, 'r') as zf:
        zf.extractall(path)
    os.remove(dl_path)