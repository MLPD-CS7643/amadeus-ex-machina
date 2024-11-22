import os
import numpy as np
from scipy.io.wavfile import write
import tinysoundfont as tsf
import wave

# Paths
MIDI_FILES_PATH = "./midi_files/"
SOUNDFONTS_PATH = "./soundfonts/"
OUTPUT_PATH = "./output/"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

def synthesize_to_wav(midi_path, soundfont_path, instrument_id, output_file):
    # Initialize the synthesizer and load the soundfont
    synth = tsf.Synth()
    soundfont_id = synth.sfload(soundfont_path)
    synth.program_select(0, soundfont_id, 0, 0)

    # Start the synthesizer (assumes correct setup prior to this point)
    synth.start()

    # Load the MIDI file into the sequencer
    seq = tsf.Sequencer(synth)
    seq.midi_load(midi_path)

    # Collect audio samples into a list
    audio_samples = []
    sample_rate = 44100  # Define the sample rate
    seconds_to_generate = 2  # Define how many seconds of audio to generate based on MIDI length estimation
    samples_per_chunk = 4096  # Number of samples per chunk
    total_samples = sample_rate * seconds_to_generate  # Total samples to generate

    # Generating and collecting audio samples
    for _ in range(0, total_samples, samples_per_chunk):
        buffer = synth.generate(samples_per_chunk)  # Generates and returns a memoryview
        audio_data = np.frombuffer(buffer, dtype=np.float32).reshape(-1, 2)
        audio_samples.append(audio_data)

    # Concatenate all collected audio data
    full_audio = np.concatenate(audio_samples)

    # Save the synthesized audio to a WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(4)  # Bytes per sample, float32
        wf.setframerate(sample_rate)  # Sample rate in Hz
        wf.writeframes(full_audio.tobytes())

    synth.stop()


def batch_process_midi(midi_path, soundfonts, output_path):
    """Batch process MIDI files with multiple SoundFonts."""
    for soundfont in soundfonts:
        sf_name = os.path.basename(soundfont).replace('.sf2', '')
        print(f"Using SoundFont: {sf_name}")
        
        #for instrument in range(0, 128):
        for midi_file in os.listdir(midi_path):
            if midi_file.endswith('.mid'):
                input_file = os.path.join(midi_path, midi_file)
                output_file = os.path.join(output_path, f"{midi_file[:-4]}_{sf_name}.wav")

                print(f"Converting {midi_file} with {sf_name} to {output_file}")
                synthesize_to_wav(input_file, soundfont, 0, output_file)
                #synthesize_to_wav(input_file, soundfont, instrument, output_file)


def main():
    # Load SoundFonts
    soundfonts = [os.path.join(SOUNDFONTS_PATH, f) for f in os.listdir(SOUNDFONTS_PATH) if f.endswith('.sf2')]
    if not soundfonts:
        print("No SoundFonts found in the soundfonts directory!")
        return

    # Process MIDI files with each SoundFont
    batch_process_midi(MIDI_FILES_PATH, soundfonts, OUTPUT_PATH)
    print("Batch processing complete. Check the output directory.")

if __name__ == "__main__":
    main()
