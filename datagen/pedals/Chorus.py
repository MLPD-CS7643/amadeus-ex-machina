import numpy as np
from scipy.interpolate import interp1d

class Chorus:
    def __init__(self, sr=44100):
        """
        Initialize chorus with sample rate.
        
        Parameters:
        sr (int): Sample rate
        """
        self.sr = sr
        
    def _generate_lfo(self, length, rate, phase=0, shape='sine'):
        """
        Generate LFO waveform for modulation.
        
        Parameters:
        length (int): Length in samples
        rate (float): LFO rate in Hz
        phase (float): Initial phase in radians
        shape (str): LFO shape ('sine', 'triangle', or 'random')
        
        Returns:
        np.array: LFO waveform scaled between 0 and 1
        """
        t = np.arange(length) / self.sr
        
        if shape == 'sine':
            lfo = np.sin(2 * np.pi * rate * t + phase)
        elif shape == 'triangle':
            period = self.sr / rate
            ramp = (2 * np.mod(t * rate + phase/(2*np.pi), 1)) - 1
            lfo = 2 * np.abs(ramp) - 1
        elif shape == 'random':
            # Generate smooth random LFO with linear interpolation
            num_points = max(3, int(length * rate / self.sr) + 2)  # Ensure at least 3 points
            random_points = np.random.uniform(-1, 1, num_points)
            
            # Add endpoints to ensure interpolation covers full range
            x_points = np.linspace(0, length/self.sr, num_points)
            
            # Use linear interpolation instead of cubic
            interpolator = interp1d(x_points, random_points, kind='linear',
                                  bounds_error=False, fill_value="extrapolate")
            lfo = interpolator(t)
        else:
            lfo = np.sin(2 * np.pi * rate * t + phase)  # Default to sine
            
        # Scale to 0-1 range
        return (lfo + 1) / 2
    
    def process(self, audio, rate=1.0, depth=0.3, delay_ms=7.0, voices=3,
                feedback=0.0, mix=0.5, lfo_shape='sine'):
        """
        Apply chorus effect to audio signal.
        
        Parameters:
        audio (np.array): Input audio signal
        rate (float): LFO rate in Hz (0.1 to 5 Hz)
        depth (float): Modulation depth (0 to 1)
        delay_ms (float): Base delay time in milliseconds
        voices (int): Number of chorus voices
        feedback (float): Feedback amount (0 to 1)
        mix (float): Wet/dry mix (0 to 1)
        lfo_shape (str): LFO shape ('sine', 'triangle', or 'random')
        
        Returns:
        np.array: Processed audio with chorus effect
        """
        # Ensure audio is float32
        audio = np.array(audio, dtype=np.float32)
        
        # Calculate delay parameters
        base_delay_samples = int(delay_ms * self.sr / 1000)
        max_delay_samples = base_delay_samples * 2
        
        # Initialize output buffer
        wet = np.zeros_like(audio)
        
        # Process each voice
        for voice in range(voices):
            # Different phase for each voice
            phase = voice * 2 * np.pi / voices
            voice_rate = rate * (1 + 0.1 * (voice - voices/2) / voices)
            
            # Generate LFO
            lfo = self._generate_lfo(len(audio), voice_rate, phase, lfo_shape)
            
            # Calculate delay modulation
            delay_mod = (lfo * depth * base_delay_samples).astype(int)
            
            # Create buffer for this voice
            voice_buffer = np.zeros(len(audio) + max_delay_samples, dtype=np.float32)
            voice_buffer[:len(audio)] = audio
            
            # Process voice
            for i in range(len(audio)):
                delay_idx = i + max_delay_samples - delay_mod[i]
                wet[i] += voice_buffer[delay_idx] / voices
        
        # Apply feedback
        if feedback > 0:
            wet = wet + feedback * np.roll(wet, base_delay_samples)
        
        # Mix dry and wet signals
        output = (1 - mix) * audio + mix * wet
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output
    
    def get_presets(self):
        """
        Get preset chorus settings.
        
        Returns:
        dict: Dictionary of preset parameters
        """
        presets = {
            'subtle': {
                'rate': 0.5,
                'depth': 0.2,
                'delay_ms': 7.0,
                'voices': 2,
                'feedback': 0.1,
                'mix': 0.3,
                'lfo_shape': 'sine'
            },
            'classic': {
                'rate': 0.8,
                'depth': 0.3,
                'delay_ms': 12.0,
                'voices': 3,
                'feedback': 0.2,
                'mix': 0.5,
                'lfo_shape': 'sine'
            },
            'deep': {
                'rate': 0.3,
                'depth': 0.5,
                'delay_ms': 15.0,
                'voices': 4,
                'feedback': 0.3,
                'mix': 0.7,
                'lfo_shape': 'triangle'
            },
            'ensemble': {
                'rate': 0.9,
                'depth': 0.4,
                'delay_ms': 10.0,
                'voices': 6,
                'feedback': 0.2,
                'mix': 0.6,
                'lfo_shape': 'random'
            }
        }
        return presets