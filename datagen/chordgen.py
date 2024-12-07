import os
import copy
import json
import wave
import numpy as np
import tinysoundfont as tsf
from pathlib import Path
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
from zipfile import ZipFile
from utils.gdrive import download_from_gdrive
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

JSON_FILE = "chord_ref.json"
SF2_ARCHIVE = "FluidR3_GM.sf2"

WAV_SUBDIR = "wav"
SF2_SUBDIR = "sf2"

SAMPLE_RATE = 44100
BIT_DEPTH = 16

C0 = 12

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORDS = {
    "1": [0],
    "5": [0, 7],
    "maj": [0, 4, 7],
    "maj/2": [0, 4, 7, -10],
    "maj/4": [0, 4, 7, -7],
    "maj(9)": [0, 4, 7, 14],
    "min": [0, 3, 7],
    "min/2": [0, -9, 7],
    "min/4": [0, 3, -5],
    "min(9)": [0, 3, 7, 14],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "sus4(b7)": [0, 5, 7, 10],
    "sus4(9)": [0, 5, 7, 14],
    "sus4(b7,9)": [0, 5, 7, 10, 14],
    "maj6": [0, 4, 7, 9],
    "maj6(9)": [0, 4, 7, 9, 14],
    "maj7": [0, 4, 7, 11],
    "min6": [0, 3, 7, 9],
    "min7": [0, 3, 7, 10],
    "minmaj7": [0, 3, 7, 11],
    "7": [0, 4, 7, 10],
    "hdim7": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "9": [0, 4, 7, 10, 14],
    "7(#9)": [0, 4, 7, 10, 15],
    "11": [0, 4, 7, 10, 14, 17],
    "min11": [0, 3, 7, 10, 14, 17],
    "13": [0, 4, 7, 10, 14, 17, 21],
}

INVERSIONS = {
    "maj/3": [0, -8, 7],
    "maj/5": [0, 4, -5],
    "min/b3": [0, -9, 7],
    "min/5": [0, 3, -5],
    "maj7/3": [0, -8, 7, 11],
    "maj7/5": [0, 4, -5, 11],
    "maj7/7": [0, 4, 7, -1],
    "min7/b3": [0, -9, 7, 10],
    "min7/5": [0, 3, -5, 10],
    "min7/b7": [0, 3, 7, -2],
    "7/3": [0, -8, 7, 10],
    "7/5": [0, 4, -5, 10],
    "7/b7": [0, 4, 7, -2]
}

GM_INSTRUMENTS = {
    0: "acoustic_grand_piano",
    #1: "bright_acoustic_piano",
    #2: "electric_grand_piano",
    3: "honky-tonk_piano",
    4: "electric_piano_1",
    5: "electric_piano_2",
    6: "harpsichord",
    7: "clavi",
    8: "celesta",
    9: "glockenspiel",
    #10: "music_box",
    11: "vibraphone",
    12: "marimba",
    13: "xylophone",
    #14: "tubular_bells",
    15: "dulcimer",
    #16: "drawbar_organ",
    #17: "percussive_organ",
    18: "rock_organ",
    #19: "church_organ",
    #20: "reed_organ",
    21: "accordion",
    22: "harmonica",
    #23: "tango_accordion",
    24: "acoustic_guitar_(nylon)",
    25: "acoustic_guitar_(steel)",
    26: "electric_guitar_(jazz)",
    27: "electric_guitar_(clean)",
    29: "overdriven_guitar",
    30: "distortion_guitar",
    32: "acoustic_bass",
    33: "electric_bass_(finger)",
    34: "electric_bass_(pick)",
    #35: "fretless_bass",
    36: "slap_bass_1",
    #37: "slap_bass_2",
    38: "synth_bass_1",
    #39: "synth_bass_2",
    40: "violin",
    #41: "viola",
    42: "cello",
    44: "tremolo_strings",
    45: "pizzicato_strings",
    46: "orchestral_harp",
    48: "string_ensemble_1",
    #49: "string_ensemble_2",
    50: "synthstrings_1",
    #51: "synthstrings_2",
    52: "choir_aahs",
    53: "voice_oohs",
    #54: "synth_voice",
    55: "orchestra_hit",
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    59: "muted_trumpet",
    60: "french_horn",
    61: "brass_section",
    62: "synthbrass_1",
    #63: "synthbrass_2",
    64: "soprano_sax",
    #65: "alto_sax",
    66: "tenor_sax",
    #67: "baritone_sax",
    68: "oboe",
    #69: "english_horn",
    70: "bassoon",
    #71: "clarinet",
    #72: "piccolo",
    73: "flute",
    74: "recorder",
    75: "pan_flute",
    #76: "blown_bottle",
    77: "shakuhachi",
    #78: "whistle",
    #79: "ocarina",
    80: "lead_1_(square)",
    81: "lead_2_(sawtooth)",
    84: "lead_5_(charang)",
    85: "lead_6_(voice)",
    88: "pad_1_(new_age)",
    89: "pad_2_(warm)",
    #90: "pad_3_(polysynth)",
    91: "pad_4_(choir)",
    #92: "pad_5_(bowed)",
    104: "sitar",
    105: "banjo",
    106: "shamisen",
    #107: "koto",
    108: "kalimba",
    109: "bag_pipe",
    110: "fiddle",
    111: "shanai",
    #112: "tinkle_bell",
    #113: "agogo",
    114: "steel_drums",
}

def generate_all_chords(out_dir, download_sf2, inversions, start_octave:int=4, end_octave:int=4, duration=2.0, make_dir=False, n_jobs=1):
    """
    Generates .wav files for all defined chords using all available SoundFonts.
    Also saves chord_ref.json lookup table with metadata.

    Args:
        out_dir (str): output directory
        download_sf2 (bool): download repository of SoundFonts
        start_octave (int): first octave (min 0)
        end_octave (int): last octave (max 7)
        make_dir (bool): automatically create output directory if it doesn't exist (make sure you know where your working directory is set)
    
    Returns:
        None
    """
    chord_definitions = copy.deepcopy(CHORDS)
    if inversions:
        chord_definitions.update(INVERSIONS)
    base_path = Path(out_dir)
    if make_dir:
        base_path.mkdir(parents=True, exist_ok=True)
    sf2_dir = base_path / SF2_SUBDIR
    wav_dir = base_path / WAV_SUBDIR
    sf_filepath = sf2_dir / SF2_ARCHIVE
    if download_sf2:
        __fetch_sf2_archive(sf2_dir)
    # soundfont_names = [f[:-4] for f in os.listdir(sf2_dir) if f.endswith('.sf2')]
    # if not soundfont_names:
    #     print(f"No SoundFonts found in {sf2_dir}!")
    #     return
    os.makedirs(wav_dir, exist_ok=True)
    json_out = {}
    print("Starting chord generation...")

    for octave in range(start_octave, end_octave + 1):
        for i in range(12):
            print(f"Processing octave {octave}, note {i+1}/12...")
            root = C0 + octave * 12 + i
            note_name = __note_lookup(root)
            tasks = []
            mid_filepaths = []
            for chord_class, intervals in chord_definitions.items():
                midi = __generate_midi_chord(root, intervals)
                mid_filename = f"{note_name}{chord_class.replace('/','inv')}_O{octave}"
                mid_filepath = wav_dir / f"{mid_filename}.mid"
                midi.save(mid_filepath)
                mid_filepaths.append(mid_filepath)
                for preset_id, instrument_name in GM_INSTRUMENTS.items():
                    wav_filename = f"{mid_filename}_{instrument_name}"
                    wav_filepath = wav_dir / f"{wav_filename}.wav"
                    tasks.append((str(mid_filepath.absolute()), str(sf_filepath.absolute()), str(wav_filepath.absolute()), preset_id, duration, note_name, chord_class, octave, instrument_name, wav_filename))
            
            with tqdm_joblib(tqdm(desc="Processing chords", total=len(tasks))) as progress_bar:
                results = Parallel(n_jobs=n_jobs)(delayed(__process_chord_task)(*t) for t in tasks)
            
            for path in mid_filepaths:
                if path.exists():
                    os.remove(path)
            
            for res in results:
                json_out.update(res)

    print("Saving lookup table...")
    __save_json(json_out, base_path)
    print("Fin~")

def __process_chord_task(mid_filepath, sf_filepath, wav_filepath, preset_id, duration, note_name, chord_class, octave, instrument_name, wav_filename):
    __synthesize_to_wav(
        mid_filepath,
        sf_filepath,
        wav_filepath,
        preset_id=preset_id,
        seconds_to_generate=duration,
        sample_rate=SAMPLE_RATE,
        bit_depth=BIT_DEPTH,
        gain=-6 if preset_id == 29 or preset_id == 30 else -3
    )
    return {
        wav_filename: {
            "root": note_name,
            "chord_class": chord_class,
            "billboard_notation": f"{note_name}:{chord_class}",
            "octave": octave,
            "instrument": instrument_name,
            "gm_preset_id": preset_id,
            "filename": f"{wav_filename}.wav",
            "format": "wav",
            "duration(s)": duration,
            "sample_rate": SAMPLE_RATE,
            "bit_depth": BIT_DEPTH
        }
    }


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
    print("Downloading sf2...")
    dl_path = path / SF2_ARCHIVE
    download_from_gdrive(SF2_ARCHIVE, str(dl_path.absolute()))
    # print("Extracting sf2.zip...")
    # with ZipFile(dl_path, 'r') as zf:
    #     zf.extractall(path)
    # os.remove(dl_path)

def __synthesize_to_wav(midi_path, soundfont_path, output_file, instrument_id=0, preset_id=0, seconds_to_generate=2.0, sample_rate=44100, bit_depth=16, gain=-3):   
    # Initialize the synthesizer and load the soundfont
    synth = tsf.Synth(gain=-gain)
    soundfont_id = synth.sfload(soundfont_path)
    synth.program_select(instrument_id, soundfont_id, 0, preset_id)

    # Start the synthesizer (assumes correct setup prior to this point)
    synth.start()

    # Load the MIDI file into the sequencer
    seq = tsf.Sequencer(synth)
    seq.midi_load(midi_path)

    # Collect audio samples into a list
    audio_samples = []
    samples_per_chunk = 4096  # Number of samples per chunk
    total_samples = int(sample_rate * seconds_to_generate)  # Total samples to generate

    # Generating and collecting audio samples
    for _ in range(0, total_samples+1, samples_per_chunk):
        buffer = synth.generate(samples_per_chunk)  # Generates and returns a memoryview
        audio_data = np.frombuffer(buffer, dtype=np.float32).reshape(-1, 2)
        audio_samples.append(audio_data)

    # Concatenate all collected audio data
    full_audio = np.concatenate(audio_samples)

    if bit_depth == 16:
        # Convert float32 to int16
        int_audio = np.int16(full_audio * 32767)
    else:
        print(f"Unsupported bit depth of {bit_depth}")
        return

    # Save the synthesized audio to a WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # Bytes per sample, int16
        wf.setframerate(sample_rate)  # Sample rate in Hz
        wf.writeframes(int_audio.tobytes())

    synth.stop()