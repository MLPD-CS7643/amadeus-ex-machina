import librosa
import numpy as np
from scipy.signal import convolve
from scipy.signal import lfilter, butter
import soundfile as sf
import os

INPUT_PATH = "./wav_out/"
OUTPUT_PATH = "./fx_out/"

def generate_synthetic_ir(duration=1.0, sr=44100, decay_rate=0.1):
    # Generate white noise
    noise = np.random.normal(0, 1, int(sr * duration))
    
    # Apply exponential decay
    decay = np.exp(-decay_rate * np.arange(len(noise)))
    ir = noise * decay
    
    # Apply a low-pass filter to simulate high frequency absorption
    b, a = butter(4, 0.2, 'low')  # Low-pass filter with cutoff frequency at 20% of Nyquist
    ir = lfilter(b, a, ir)
    
    return ir

def reverb(y, sr):
    # Generate or load an impulse response
    # For simplicity, we'll create a simple reverb impulse here
    ir = generate_synthetic_ir()

    # Apply convolution to simulate reverb
    reverbed = convolve(y, ir, mode='full')[:len(y)]

    return reverbed

def delay(y, sr):
    delay_time = 0.5  # seconds
    echo_strength = 0.6  # reduction in strength of the echo
    delay_samples = int(sr * delay_time)

    # Create an empty array for the output
    delayed = np.zeros_like(y)

    # Insert the original signal with an offset and attenuate
    delayed[delay_samples:] = y[:-delay_samples] * echo_strength

    # Mix original and delayed signal
    echoed = y + delayed

    return echoed

def distort(y, sr):
    # Simple hard clipping distortion
    threshold = 0.2
    distorted = np.clip(y, -threshold, threshold)
    
    return distorted

def compress(y, sr):
    # Attack and release times
    attack_ms = 50.0
    release_ms = 50.0
    attack_samples = int((attack_ms / 1000) * sr)
    release_samples = int((release_ms / 1000) * sr)

    # Set threshold and ratio for compression
    threshold = -20  # dB
    ratio = 4.0

    # Convert threshold to linear scale
    threshold_lin = 10**(threshold / 20)

    # Envelope follower - rectify and smooth with a low-pass filter
    envelope = np.abs(y)
    envelope = lfilter([1.0], [1.0, -(1 - (1.0 / attack_samples))], envelope)
    envelope = lfilter([1.0], [1.0, -(1 - (1.0 / release_samples))], envelope)

    # Gain computer
    gain = np.minimum(1, threshold_lin / (envelope + 1e-9))

    # Compression applied
    compressed = y * gain

    return compressed

def chorus(y, sr):
    lfo_rate = 0.5  # LFO rate in Hz
    lfo_depth = 0.002  # depth in seconds

    # LFO waveform
    lfo = 0.5 * lfo_depth * sr * np.sin(2 * np.pi * np.arange(len(y)) * lfo_rate / sr)

    # Chorus effect
    chorus = np.zeros_like(y)
    for i in range(len(y)):
        delay = int(lfo[i])
        if i + delay < len(y) and i + delay >= 0:
            chorus[i] += y[i + delay]
    chorus = 0.7 * y + 0.3 * chorus

    return chorus

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
    #delay_out = os.path.join(OUTPUT_PATH, "delay")
    #os.makedirs(delay_out, exist_ok=True)
    #process_directory(INPUT_PATH, delay_out, delay)
    #distort_out = os.path.join(OUTPUT_PATH, "distort")
    #os.makedirs(distort_out, exist_ok=True)
    #process_directory(INPUT_PATH, distort_out, distort)
    #compress_out = os.path.join(OUTPUT_PATH, "compress")
    #os.makedirs(compress_out, exist_ok=True)
    #process_directory(INPUT_PATH, compress_out, compress)
    #chorus_out = os.path.join(OUTPUT_PATH, "chorus")
    #os.makedirs(chorus_out, exist_ok=True)
    #process_directory(INPUT_PATH, chorus_out, chorus)

if __name__ == "__main__":
    main()
