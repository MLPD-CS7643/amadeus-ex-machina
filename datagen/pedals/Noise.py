import numpy as np
from scipy import signal

class NoiseGenerator:
    def __init__(self, sr=44100):
        """
        Initialize noise generator with sample rate.
        
        Parameters:
        sr (int): Sample rate
        """
        self.sr = sr
        
    def _db_to_linear(self, db):
        """Convert dB to linear gain"""
        return 10 ** (db / 20)
    
    def _linear_to_db(self, linear):
        """Convert linear gain to dB"""
        return 20 * np.log10(max(abs(linear), 1e-6))
    
    def _generate_white_noise(self, length):
        """Generate white noise"""
        return np.random.normal(0, 1, length)
    
    def _generate_pink_noise(self, length):
        """Generate pink noise using the Voss-McCartney algorithm"""
        num_octaves = 16
        octaves = []
        for i in range(num_octaves):
            octave = np.random.normal(0, 1, length)
            octave = signal.resample(octave, length//(2**i))
            octave = signal.resample(octave, length)
            octave = octave * (1/np.sqrt(2**i))
            octaves.append(octave)
        
        pink = np.sum(octaves, axis=0)
        return pink / np.std(pink)
    
    def _generate_brown_noise(self, length):
        """Generate brown noise by integrating white noise"""
        white = np.random.normal(0, 1, length)
        brown = np.cumsum(white)
        brown = brown - np.mean(brown)
        return brown / np.std(brown)
    
    def add_noise(self, audio, noise_type='white', noise_level_db=-50, highpass_freq=20, 
                lowpass_freq=20000, noise_gate_db=-60):
        """
        Add noise to audio signal.
        
        Parameters:
        audio (np.array): Input audio signal
        noise_type (str): Type of noise ('white', 'pink', or 'brown')
        noise_level_db (float): Noise level in dB relative to input signal (e.g., -60 dB)
        highpass_freq (float): High-pass filter frequency in Hz
        lowpass_freq (float): Low-pass filter frequency in Hz
        noise_gate_db (float): Noise gate threshold in dB
        
        Returns:
        np.array: Processed audio with added noise
        """
        # Ensure input is float32 and normalize
        audio = np.array(audio, dtype=np.float32)
        audio = audio / np.max(np.abs(audio))
        
        # Generate and normalize noise
        if noise_type == 'white':
            noise = self._generate_white_noise(len(audio))
        elif noise_type == 'pink':
            noise = self._generate_pink_noise(len(audio))
        elif noise_type == 'brown':
            noise = self._generate_brown_noise(len(audio))
        else:
            raise ValueError("Noise type must be 'white', 'pink', or 'brown'")
        
        # Apply filters to noise
        nyquist = self.sr / 2
        if highpass_freq > 0:
            b, a = signal.butter(4, highpass_freq/nyquist, 'high')
            noise = signal.filtfilt(b, a, noise)
        
        if lowpass_freq < nyquist:
            b, a = signal.butter(4, lowpass_freq/nyquist, 'low')
            noise = signal.filtfilt(b, a, noise)
        
        # Normalize noise
        noise = noise / np.max(np.abs(noise))
        
        # Convert noise level from dB to linear scale
        noise_gain = self._db_to_linear(noise_level_db)
        
        # Apply noise gate to input audio if specified
        if noise_gate_db is not None:
            gate_threshold = self._db_to_linear(noise_gate_db)
            envelope = np.abs(signal.hilbert(audio))
            gate_mask = envelope > gate_threshold
            audio = audio * gate_mask
        
        # Mix noise with audio using dB-scaled noise
        output = audio + (noise * noise_gain)
        
        # Normalize output to prevent clipping
        output = output / np.max(np.abs(output))
        
        return output

    def get_presets(self):
        """
        Get preset noise settings.
        
        Returns:
        dict: Dictionary of preset parameters
        """
        presets = {
            'vinyl_crackle': {
                'noise_type': 'pink',
                'noise_level_db': -50,
                'highpass_freq': 1000,
                'lowpass_freq': 16000,
                'noise_gate_db': -50
            },
            'tape_hiss': {
                'noise_type': 'white',
                'noise_level_db': -60,
                'highpass_freq': 2000,
                'lowpass_freq': 15000,
                'noise_gate_db': -55
            },
            'room_noise': {
                'noise_type': 'brown',
                'noise_level_db': -45,
                'highpass_freq': 30,
                'lowpass_freq': 2000,
                'noise_gate_db': -65
            }
        }
        return presets