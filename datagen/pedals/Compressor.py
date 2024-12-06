import numpy as np
from scipy.signal import lfilter
import math

class Compressor:
    def __init__(self, sr=44100):
        """
        Initialize compressor with sample rate.
        
        Parameters:
        sr (int): Sample rate
        """
        self.sr = sr
        self._prev_gain = 1.0
        self._prev_envelope = 0.0
        
    def _db_to_linear(self, db):
        """Convert dB to linear gain"""
        return 10 ** (db / 20)
    
    def _linear_to_db(self, linear):
        """Convert linear gain to dB"""
        return 20 * math.log10(max(abs(linear), 1e-6))
    
    def _calculate_attack_release(self, time_ms):
        """Calculate coefficient for attack/release filter"""
        return 1.0 - math.exp(-1.0 / (self.sr * time_ms / 1000.0))
    
    def _soft_knee(self, x_db, threshold, knee_width, ratio):
        """
        Calculate gain reduction with soft knee.
        
        Parameters:
        x_db (float): Input level in dB
        threshold (float): Threshold in dB
        knee_width (float): Knee width in dB
        ratio (float): Compression ratio
        
        Returns:
        float: Gain reduction in dB
        """
        if knee_width > 0:
            # Soft knee
            if x_db < (threshold - knee_width/2):
                # Below knee
                return 0.0
            elif x_db > (threshold + knee_width/2):
                # Above knee
                return -(x_db - threshold) * (1 - 1/ratio)
            else:
                # In knee
                knee_factor = ((x_db - threshold + knee_width/2) / knee_width) ** 2
                return -knee_factor * (x_db - threshold) * (1 - 1/ratio)
        else:
            # Hard knee
            if x_db < threshold:
                return 0.0
            else:
                return -(x_db - threshold) * (1 - 1/ratio)
    
    def process(self, audio, threshold=-20, ratio=4.0, attack_ms=10.0, release_ms=100.0,
                knee_width=10.0, makeup_gain_db=0.0, auto_makeup=True, lookahead_ms=0.0):
        """
        Apply compression to audio signal.
        
        Parameters:
        audio (np.array): Input audio signal
        threshold (float): Threshold level in dB
        ratio (float): Compression ratio (e.g., 4.0 for 4:1)
        attack_ms (float): Attack time in milliseconds
        release_ms (float): Release time in milliseconds
        knee_width (float): Soft knee width in dB
        makeup_gain_db (float): Manual makeup gain in dB
        auto_makeup (bool): Automatically calculate makeup gain
        lookahead_ms (float): Lookahead time in milliseconds
        
        Returns:
        np.array: Compressed audio signal
        """
        # Ensure input is float32
        audio = np.array(audio, dtype=np.float32)
        
        # Apply lookahead if specified
        if lookahead_ms > 0:
            lookahead_samples = int((lookahead_ms / 1000) * self.sr)
            audio_delayed = np.pad(audio, (lookahead_samples, 0))[:-lookahead_samples]
        else:
            audio_delayed = audio
            
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Apply attack/release smoothing
        alpha_attack = self._calculate_attack_release(attack_ms)
        alpha_release = self._calculate_attack_release(release_ms)
        
        # Envelope follower with different attack/release
        env_db = np.zeros_like(envelope)
        env_db[0] = self._linear_to_db(envelope[0])
        
        for i in range(1, len(envelope)):
            env_current_db = self._linear_to_db(envelope[i])
            if env_current_db > env_db[i-1]:
                env_db[i] = env_db[i-1] + alpha_attack * (env_current_db - env_db[i-1])
            else:
                env_db[i] = env_db[i-1] + alpha_release * (env_current_db - env_db[i-1])
        
        # Calculate gain reduction with soft knee
        gain_reduction_db = np.array([self._soft_knee(x, threshold, knee_width, ratio) 
                                    for x in env_db])
        
        # Convert gain reduction to linear
        gain_reduction = self._db_to_linear(gain_reduction_db)
        
        # Calculate auto makeup gain if enabled
        if auto_makeup:
            makeup_gain_db = -(threshold + (threshold / ratio)) / 2
        
        # Apply makeup gain
        makeup_gain = self._db_to_linear(makeup_gain_db)
        
        # Apply compression and makeup gain
        output = audio_delayed * gain_reduction * makeup_gain
        
        # Prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
            
        return output
    
    def get_presets(self):
        """
        Get preset settings for common compression effects.
        
        Returns:
        dict: Preset parameters
        """
        return {
            'vocal_gentle': {
                'threshold': -20,
                'ratio': 2.0,
                'attack_ms': 15,
                'release_ms': 100,
                'knee_width': 10,
                'auto_makeup': True
            },
            'vocal_heavy': {
                'threshold': -24,
                'ratio': 4.0,
                'attack_ms': 10,
                'release_ms': 80,
                'knee_width': 6,
                'auto_makeup': True
            },
            'drums_punch': {
                'threshold': -18,
                'ratio': 4.0,
                'attack_ms': 8,
                'release_ms': 60,
                'knee_width': 3,
                'auto_makeup': True
            },
            'drums_glue': {
                'threshold': -16,
                'ratio': 2.5,
                'attack_ms': 25,
                'release_ms': 150,
                'knee_width': 8,
                'auto_makeup': True
            },
            'bass': {
                'threshold': -22,
                'ratio': 3.0,
                'attack_ms': 20,
                'release_ms': 120,
                'knee_width': 6,
                'auto_makeup': True
            },
            'limiter': {
                'threshold': -2,
                'ratio': 20.0,
                'attack_ms': 1,
                'release_ms': 50,
                'knee_width': 0,
                'auto_makeup': False
            }
        }

# Compression presets
COMPRESSOR_PRESETS = {
    'vocal_gentle': {
        'threshold': -20,
        'ratio': 2.0,
        'attack_ms': 15,
        'release_ms': 100,
        'knee_width': 10,
        'auto_makeup': True
    },
    'vocal_heavy': {
        'threshold': -24,
        'ratio': 4.0,
        'attack_ms': 10,
        'release_ms': 80,
        'knee_width': 6,
        'auto_makeup': True
    },
    'drums_punch': {
        'threshold': -18,
        'ratio': 4.0,
        'attack_ms': 8,
        'release_ms': 60,
        'knee_width': 3,
        'auto_makeup': True
    },
    'drums_glue': {
        'threshold': -16,
        'ratio': 2.5,
        'attack_ms': 25,
        'release_ms': 150,
        'knee_width': 8,
        'auto_makeup': True
    },
    'bass': {
        'threshold': -22,
        'ratio': 3.0,
        'attack_ms': 20,
        'release_ms': 120,
        'knee_width': 6,
        'auto_makeup': True
    },
    'limiter': {
        'threshold': -2,
        'ratio': 20.0,
        'attack_ms': 1,
        'release_ms': 50,
        'knee_width': 0,
        'auto_makeup': False
    }
}

def test_compressor():
    """
    Test function to demonstrate different compression settings
    """
    # Generate test signal (drum-like sound with dynamics)
    duration = 2.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with varying amplitudes
    signal = np.sin(2 * np.pi * 200 * t)
    envelope = np.exp(-5 * np.mod(t, 0.5)) * (1 + np.sin(2 * np.pi * 0.5 * t))
    test_signal = signal * envelope
    
    # Create compressor
    comp = Compressor(sr=sr)
    
    # Test different settings
    results = {
        'dry': test_signal,
        'gentle': comp.process(test_signal, **COMPRESSOR_PRESETS['vocal_gentle']),
        'heavy': comp.process(test_signal, **COMPRESSOR_PRESETS['vocal_heavy']),
        'drums': comp.process(test_signal, **COMPRESSOR_PRESETS['drums_punch']),
        'limiter': comp.process(test_signal, **COMPRESSOR_PRESETS['limiter'])
    }
    
    return results, sr