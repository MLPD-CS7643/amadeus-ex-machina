import librosa
import numpy as np
from scipy.signal import convolve
from scipy.signal import lfilter, butter
import soundfile as sf
import os
from pedals import Compressor, Delay, Distortion, Reverb, Chorus, Noise, Flanger

INPUT_PATH = "./wav_out/"
OUTPUT_PATH = "./fx_out/"

def reverb(y, sr):
    r = Reverb.Reverb(sr=sr)
    reverb = r.reverb(y)

    return reverb

def delay(y, sr):
    d = Delay.Delay(sr=sr)
    delayed = d.delay(y)

    return delayed

def distort(y, sr):
    d = Distortion.Distortion(sr=sr)
    distorted = d.distort(y)

    return distorted

def compress(y, sr):
    c = Compressor.Compressor(sr=sr)
    compressed = c.process(y)

    return compressed

def chorus(y, sr):
    c = Chorus.Chorus(sr=sr)
    chorused = c.process(y)

    return chorused

def noise(y, sr):
    n = Noise.NoiseGenerator(sr=sr)
    noised = n.add_noise(y)

    return noised

def flanger(y, sr):
    f = Flanger.Flanger(sr=sr)
    flanged = f.process(y)

    return flanged

def save_audio(audio_data, sr, file_path):
    sf.write(file_path, audio_data, sr)

def process_directory(input_path, output_path, effect_func):
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Loop over all files in the input directory
    for file_name in os.listdir(input_path):
        if file_name.endswith(".wav"):
            # Load the audio file
            file_path = os.path.join(input_path, file_name)
            y, sr = librosa.load(file_path, sr=None) #sample, sample rate
            
            # Apply the effect
            processed_audio = effect_func(y, sr)
            
            # Save the processed audio to the output directory
            output_file_path = os.path.join(output_path, file_name)
            save_audio(processed_audio, sr, output_file_path)

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    verb_out = os.path.join(OUTPUT_PATH, "reverb")
    os.makedirs(verb_out, exist_ok=True)
    process_directory(INPUT_PATH, verb_out, reverb)
    delay_out = os.path.join(OUTPUT_PATH, "delay")
    os.makedirs(delay_out, exist_ok=True)
    process_directory(INPUT_PATH, delay_out, delay)
    distort_out = os.path.join(OUTPUT_PATH, "distort")
    os.makedirs(distort_out, exist_ok=True)
    process_directory(INPUT_PATH, distort_out, distort)
    compress_out = os.path.join(OUTPUT_PATH, "compress")
    os.makedirs(compress_out, exist_ok=True)
    process_directory(INPUT_PATH, compress_out, compress)
    chorus_out = os.path.join(OUTPUT_PATH, "chorus")
    os.makedirs(chorus_out, exist_ok=True)
    process_directory(INPUT_PATH, chorus_out, chorus)
    noise_out = os.path.join(OUTPUT_PATH, "noise")
    os.makedirs(noise_out, exist_ok=True)
    process_directory(INPUT_PATH, noise_out, noise)
    flanger_out = os.path.join(OUTPUT_PATH, "flanger")
    os.makedirs(flanger_out, exist_ok=True)
    process_directory(INPUT_PATH, flanger_out, flanger)


if __name__ == "__main__":
    main()
