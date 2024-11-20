import os
from mido import Message, MidiFile, MidiTrack

BASE_DIR = "datagen/chords/midi/"

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

def note_lookup(midi_note:int, include_octave=False):
    note_index = midi_note % 12  # Get the position of the note within the octave
    if include_octave:
        octave = (midi_note // 12) - 1  # Calculate the octave number
        return f"{NOTES[note_index]}{octave}"
    return f"{NOTES[note_index]}"

def generate_chord(root_note:int, intervals:list, velocity=64, ticks=1920):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    for interval in intervals:
        track.append(Message('note_on', note=root_note + interval, velocity=velocity, time=0))
    track.append(Message('note_off', note=root_note + intervals[0], velocity=velocity, time=ticks))
    for interval in intervals[1:]:
        track.append(Message('note_off', note=root_note + interval, velocity=velocity, time=0))
    return midi

def generate_all_chords(start_octave=0, end_octave=7):
    for octave in range(start_octave, end_octave+1):
        for i in range(12):
            root = C0 + octave * 12 + i
            for chord_type, intervals in CHORDS.items():
                midi = generate_chord(root, intervals)
                dir = f"{BASE_DIR}octave_{octave}"
                note_name = note_lookup(root)
                filename = f"oct{octave}_{note_name}{chord_type}.mid"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                midi.save(f"{dir}/{filename}")

generate_all_chords()
