import numpy as np

class Flanger:
    def __init__(self, sr=44100, max=0.020):
        """
        Initialize flanger with sample rate.
        
        Parameters:
        sr (int): Sample rate
        """
        self.sr = sr
        self.max_delay_sec = max  # 20ms maximum delay
        
    def _generate_lfo(self, length, rate, phase=0, shape='sine'):
        """
        Generate LFO waveform for modulation.
        
        Parameters:
        length (int): Length in samples
        rate (float): LFO rate in Hz
        phase (float): Initial phase in radians
        shape (str): LFO shape ('sine' or 'triangle')
        
        Returns:
        np.array: LFO waveform scaled between 0 and 1
        """
        t = np.arange(length) / self.sr
        
        if shape == 'sine':
            lfo = np.sin(2 * np.pi * rate * t + phase)
        elif shape == 'triangle':
            # Create triangle wave
            period = self.sr / rate
            ramp = (2 * np.mod(t * rate + phase/(2*np.pi), 1)) - 1
            lfo = 2 * np.abs(ramp) - 1
        else:
            lfo = np.sin(2 * np.pi * rate * t + phase)  # Default to sine
            
        # Scale to 0-1 range
        return (lfo + 1) / 2

    def process(self, audio, rate=0.5, depth=0.9, delay_ms=8.0, feedback=0.8, 
                mix=1.0, lfo_shape='sine', stereo_phase=0):
        """
        Apply flanger effect to audio signal.
        """
        # Ensure audio is float32
        audio = np.array(audio, dtype=np.float32)
        
        # Handle stereo input
        is_stereo = len(audio.shape) > 1 and audio.shape[1] == 2
        
        # Convert delay parameters to samples
        base_delay_samples = int(delay_ms * self.sr / 1000)
        max_delay_samples = int(self.max_delay_sec * self.sr)
        
        # Initialize output buffer with delay buffer padding
        if is_stereo:
            buffer = np.zeros((len(audio) + max_delay_samples, 2), dtype=np.float32)
            output = np.zeros_like(audio)
        else:
            buffer = np.zeros(len(audio) + max_delay_samples, dtype=np.float32)
            output = np.zeros_like(audio)
        
        # Generate LFO for delay modulation
        if is_stereo:
            lfo_left = self._generate_lfo(len(audio), rate, 0, lfo_shape)
            lfo_right = self._generate_lfo(len(audio), rate, stereo_phase, lfo_shape)
            lfo = np.stack([lfo_left, lfo_right], axis=1)
        else:
            lfo = self._generate_lfo(len(audio), rate, 0, lfo_shape)
        
        # Scale LFO by depth and delay amount
        delay_mod = (lfo * depth * base_delay_samples).astype(int)
        
        # Apply flanger effect
        if is_stereo:
            for channel in range(2):
                buffer[:len(audio), channel] = audio[:, channel]
                
                for i in range(len(audio)):
                    # Calculate delayed index
                    delay_idx = delay_idx = i + max_delay_samples - delay_mod[i, channel]
                    
                    # Get delayed sample and add feedback
                    delayed_sample = buffer[delay_idx, channel]
                    
                    # Combine direct signal with delayed and feedback
                    buffer[i + max_delay_samples, channel] = audio[i, channel]
                    output[i, channel] = audio[i, channel] + delayed_sample * feedback
        else:
            buffer[:len(audio)] = audio
            
            for i in range(len(audio)):
                # Calculate delayed index
                delay_idx = i + max_delay_samples - delay_mod[i]
                
                # Get delayed sample and add feedback
                delayed_sample = buffer[delay_idx]
                
                # Combine direct signal with delayed and feedback
                buffer[i + max_delay_samples] = audio[i]
                output[i] = audio[i] + delayed_sample * feedback
        
        # Mix dry and wet signals
        result = (1 - mix) * audio + mix * output
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val
            
        return result

    def get_presets(self):
        """Get preset settings for common flanger sounds."""
        presets = {
            'subtle_sweep': {
                'rate': 0.2,
                'depth': 0.6,
                'delay_ms': 3.0,
                'feedback': 0.6,
                'mix': 0.8,
                'lfo_shape': 'sine'
            },
            'classic_jet': {
                'rate': 0.8,
                'depth': 0.8,
                'delay_ms': 4.0,
                'feedback': 0.7,
                'mix': 1.0,
                'lfo_shape': 'sine'
            },
            'fast_vibrato': {
                'rate': 4.0,
                'depth': 0.8,
                'delay_ms': 2.0,
                'feedback': 0.6,
                'mix': 1.0,
                'lfo_shape': 'triangle'
            },
            'through_zero': {
                'rate': 0.5,
                'depth': 1.0,
                'delay_ms': 1.0,
                'feedback': 0.8,
                'mix': 1.0,
                'lfo_shape': 'sine'
            },
            'stereo_sweep': {
                'rate': 0.3,
                'depth': 0.7,
                'delay_ms': 3.0,
                'feedback': 0.7,
                'mix': 1.0,
                'lfo_shape': 'triangle',
                'stereo_phase': np.pi
            }
        }
        
        return presets