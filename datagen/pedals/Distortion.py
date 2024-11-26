import numpy as np
import math

def db_to_linear(db):
        """Convert dB to linear gain"""
        return 10 ** (db / 20)
    
def linear_to_db(linear):
        """Convert linear gain to dB"""
        return 20 * math.log10(max(abs(linear), 1e-6))

class Distortion:
    def __init__(self, sr=44100):
        """
        Initialize distortion effect with sample rate.
        
        Parameters:
        sr (int): Sample rate
        """
        self.sr = sr
        self._prev_gain = 1.0
        self._prev_envelope = 0.0
    
    def distort(self, y, gain=3.0, threshold=0.5, mix=1.0, mode='soft'):
        """
        Apply distortion effect to an audio signal with options for soft or hard clipping.
        
        Parameters:
        y (np.array): Input audio signal (should be normalized between -1 and 1)
        gain (float): Amount of gain to apply before clipping (higher = more distortion)
        threshold (float): Level at which clipping begins (0 to 1)
        mix (float): Mix between dry and wet signal (0 = dry only, 1 = wet only)
        mode (str): Type of clipping ('soft' or 'hard')
        
        Returns:
        np.array: Processed audio signal with distortion effect
        """
        # Make sure input is in float32 format and normalized
        y = np.array(y, dtype=np.float32)
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
        
        # Apply input gain
        gain = linear_to_db(gain)
        amplified = y * gain
        
        # Initialize output array
        processed = np.zeros_like(amplified)
        
        if mode == 'soft':
            # Soft clipping using arctan function
            # Scale arctan to maintain original amplitude range
            processed = np.arctan(amplified) / (np.pi/2)
            
            # Apply threshold by scaling the effect
            processed = processed * threshold
            
        elif mode == 'hard':
            # Hard clipping
            processed = np.clip(amplified, -threshold, threshold)
            
        else:
            raise ValueError("Mode must be either 'soft' or 'hard'")
        
        # Apply wet/dry mix
        output = (processed * mix) + (y * (1 - mix))
        
        # Normalize output to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output
    
    def get_presets(self):
        """
        Get preset distortion settings.
        
        Returns:
        dict: Dictionary of preset parameters
        """
        presets = {
            'subtle_drive': {
                'gain': 2.0,
                'threshold': 0.7,
                'mix': 0.3,
                'mode': 'soft'
            },
            'warm_overdrive': {
                'gain': 4.0,
                'threshold': 0.6,
                'mix': 0.4,
                'mode': 'soft'
            },
            'classic_distortion': {
                'gain': 8.0,
                'threshold': 0.4,
                'mix': 0.5,
                'mode': 'hard'
            },
            'fuzz': {
                'gain': 12.0,
                'threshold': 0.3,
                'mix': 0.5,
                'mode': 'hard'
            }
        }
        return presets