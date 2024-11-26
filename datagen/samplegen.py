import librosa
import numpy as np
from scipy.signal import convolve
from scipy.signal import lfilter, butter
import soundfile as sf
import os
import random
from pedals import Delay, Distortion, Reverb, Chorus, Noise, Flanger

def generate_sample(audio, sr, fx_chain, fx_params):
    """
    Process a single audio sample with a random chain of effects.
    
    Parameters:
    audio (np.array): Input audio
    sr (int): Sample rate
    fx_chain (list): List of effects to apply in order
    fx_params (dict): Dictionary of parameters for each effect
    
    Returns:
    np.array: Processed audio
    """
    processed = audio.copy()
    
    for fx in fx_chain:
        if fx == 'reverb':
            r = Reverb.Reverb(sr=sr)
            processed = r.reverb(processed, **fx_params['reverb'])
        elif fx == 'delay':
            d = Delay.Delay(sr=sr)
            processed = d.delay(processed, **fx_params['delay'])
        elif fx == 'distortion':
            d = Distortion.Distortion(sr=sr)
            processed = d.distort(processed, **fx_params['distortion'])
        elif fx == 'chorus':
            c = Chorus.Chorus(sr=sr)
            processed = c.process(processed, **fx_params['chorus'])
        elif fx == 'noise':
            n = Noise.NoiseGenerator(sr=sr)
            processed = n.add_noise(processed, **fx_params['noise'])
        #elif fx == 'flanger':
            #f = Flanger.Flanger(sr=sr)
            #processed = f.process(processed, **fx_params['flanger'])
            
    return processed

def generate_random_params(seed=None):
    """
    Generate random effect parameters and chain order.
    
    Parameters:
    seed (int): Random seed for reproducibility
    
    Returns:
    tuple: (fx_chain, fx_params)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize effect processors to get their presets
    r = Reverb.Reverb()
    d = Delay.Delay()
    dist = Distortion.Distortion()
    c = Chorus.Chorus()
    n = Noise.NoiseGenerator()
    #f = Flanger.Flanger()
    
    # Get all available effects and their presets
    fx_presets = {
        'reverb': r.get_presets(),
        'delay': d.get_presets(),
        'distortion': dist.get_presets(),
        'chorus': c.get_presets(),
        'noise': n.get_presets()
        #'flanger': f.get_presets()
    }
    
    # Randomly select which effects to use (85% chance for each)
    available_fx = list(fx_presets.keys())
    fx_chain = [fx for fx in available_fx if random.random() > 0.15]
    
    # If chain is empty, pick at least one effect
    if not fx_chain:
        fx_chain = [random.choice(available_fx)]
    
    # Randomize the order
    random.shuffle(fx_chain)
    
    # Select random presets for each effect
    fx_params = {}
    for fx in fx_presets:
        if fx in fx_chain:
            preset_name = random.choice(list(fx_presets[fx].keys()))
            fx_params[fx] = fx_presets[fx][preset_name]
    
    return fx_chain, fx_params

def generate_samples(input_path, output_path, num_samples, seed=None):
    """
    Generate multiple processed samples from input audio files.
    
    Parameters:
    input_path (str): Path to input audio files
    output_path (str): Path to save processed audio
    num_samples (int): Number of samples to generate
    seed (int): Random seed for reproducibility
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of input files
    input_files = [f for f in os.listdir(input_path) if f.endswith('.wav')]
    if not input_files:
        raise ValueError(f"No WAV files found in {input_path}")
    
    samples_generated = 0
    file_index = 0
    
    while samples_generated < num_samples:
        # Get next input file (loop if necessary)
        input_file = input_files[file_index % len(input_files)]
        file_index += 1
        
        # Load audio
        file_path = os.path.join(input_path, input_file)
        audio, sr = librosa.load(file_path, sr=None)
        
        # Generate random parameters with seed
        current_seed = None if seed is None else seed + samples_generated
        fx_chain, fx_params = generate_random_params(current_seed)
        
        # Process audio
        processed = generate_sample(audio, sr, fx_chain, fx_params)
        
        # Save processed audio
        filename = input_file.replace(".wav", "")
        filechain = str(fx_chain).replace("[", "").replace("]", "").replace("'", "").replace(",", "").replace(" ", "_")
        output_filename = f"{filename}_{filechain}.wav"
        output_file_path = os.path.join(output_path, output_filename)
        sf.write(output_file_path, processed, sr)
        
        # Save parameters for reproducibility
        params_filename = f"{filename}_{filechain}_params.txt"
        params_path = os.path.join(output_path, params_filename)
        with open(params_path, 'w') as f:
            f.write(f"Input file: {input_file}\n")
            f.write(f"Effects chain: {fx_chain}\n")
            f.write("Parameters:\n")
            for fx, params in fx_params.items():
                f.write(f"{fx}: {params}\n")
        
        samples_generated += 1
        
def main():
    INPUT_PATH = "./wav_out/"
    OUTPUT_PATH = "./fx_out/random"
    NUM_SAMPLES = 5
    SEED = 69  # Change this for different random sequences

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    generate_samples(INPUT_PATH, OUTPUT_PATH, NUM_SAMPLES, SEED)

if __name__ == "__main__":
    main()