import os
import json
import argparse
from mido import Message, MidiFile, MidiTrack
from zipfile import ZipFile
from wavgen import batch_process_midi
#from utils.gdrive import download_from_gdrive


JSON_FILE = "chord_ref.json"
SF2_ARCHIVE = "sf2.zip"

BASE_DIR = "datagen/chords/"
MIDI_DIR = f"{BASE_DIR}midi/"
WAV_DIR = f"./{BASE_DIR}wav/"
SF2_DIR = "datagen/sf2/"

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


def generate_all_chords(start_octave:int=4, end_octave:int=4):
    json_out = {}
    if not os.path.exists(MIDI_DIR):
        os.makedirs(MIDI_DIR)
    for octave in range(start_octave, end_octave+1):
        for i in range(12):
            root = C0 + octave * 12 + i
            for chord_class, intervals in CHORDS.items():
                midi = __generate_chord(root, intervals)
                note_name = __note_lookup(root)
                filename = f"oct{octave}_{note_name}{chord_class}"
                filepath = f"{MIDI_DIR}{filename}.mid"
                midi.save(filepath)
                json_out[filename] = {
                    "root": note_name,
                    "chord_class": chord_class,
                    "billboard_notation": f"{note_name}:{chord_class}",
                    "octave": octave
                }
    __save_json(json_out)

def __generate_chord(root_note:int, intervals:list, velocity=64, ticks=1920):
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

def __save_json(out:dict):
    dumps = json.dumps(out)
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    with open(f"{BASE_DIR}{JSON_FILE}", 'w') as outfile:
        outfile.write(dumps)

def __fetch_sf2_archive():
    print("Downloading sf2.zip from gdrive...")
    dl_path = f"{SF2_DIR}{SF2_ARCHIVE}"
    download_from_gdrive(SF2_ARCHIVE, dl_path)
    print("Extracting sf2.zip...")
    with ZipFile(dl_path, 'r') as zf:
        zf.extractall(SF2_DIR)
    os.remove(dl_path)
    print("DONE!")

def main(args=None):
    print("Generating midi chords...")
    generate_all_chords()
    print("DONE!")
    if args and args.wav:
        if args.download:
            __fetch_sf2_archive()
        print("Generating wav from midi...")
        if not os.path.exists(WAV_DIR):
            os.makedirs(WAV_DIR)
        # Load SoundFonts
        soundfonts = [os.path.join(SF2_DIR, f) for f in os.listdir(SF2_DIR) if f.endswith('.sf2')]
        if not soundfonts:
            print("No SoundFonts found in the soundfonts directory!")
            return
        # Process MIDI files with each SoundFont
        batch_process_midi(MIDI_DIR, soundfonts, WAV_DIR)
        print("DONE!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wav', action='store_true', help='generate wav after generating midi')
    parser.add_argument('-d', '--download', action='store_true', help='download sf2 archive from gdrive')
    args = parser.parse_args()
    main(args)