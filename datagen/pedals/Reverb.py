import numpy as np
from scipy.signal import convolve
from scipy.signal import lfilter, butter


class Reverb:
    def __init__(self, sr=44100):
        self.sample_rate = sr
        self.ir = None

    def generate_ir(self, room_size=0.8, decay=0.5, damping=0.5, diffusion=0.5,
                    early_reflections=True, pre_delay=0.02):
        """
        Generate synthetic impulse response with controllable parameters.

        Parameters:
        room_size (float): Size of the room (0.0 to 1.0, higher = bigger)
        decay (float): Decay time multiplier (0.0 to 1.0)
        damping (float): High frequency damping (0.0 to 1.0)
        diffusion (float): Echo density (0.0 to 1.0)
        early_reflections (bool): Include early reflections
        pre_delay (float): Initial delay time in seconds

        Returns:
        np.array: Impulse response
        """
        # Calculate IR duration based on room size and decay
        duration = room_size * 3.0
        num_samples = int(self.sample_rate * duration)

        # Generate base noise
        ir = np.random.normal(0, 1, num_samples)

        # Apply pre-delay
        pre_delay_samples = int(pre_delay * self.sample_rate)
        if pre_delay_samples > 0:
            ir = np.pad(ir, (pre_delay_samples, 0))[:-pre_delay_samples]

        # Generate early reflections
        if early_reflections:
            num_reflections = int(room_size * 20)
            reflection_times = np.random.uniform(0, 0.1, num_reflections)
            reflection_times = np.sort(reflection_times)

            for time in reflection_times:
                idx = int(time * self.sample_rate)
                if idx < len(ir):
                    ir[idx] += np.random.normal(0, 0.5)

        # Apply exponential decay
        decay_curve = np.exp(-decay * np.linspace(0, 5, num_samples))
        ir *= decay_curve

        # Apply diffusion (echo density)
        if diffusion > 0:
            diff_length = int(0.01 * self.sample_rate)
            diff_kernel = np.random.normal(0, diffusion, diff_length)
            diff_kernel = diff_kernel / np.sum(np.abs(diff_kernel))
            ir = convolve(ir, diff_kernel, mode='same')

        # Apply damping (frequency-dependent decay)
        nyquist = self.sample_rate * 0.5
        cutoff = 0.05 + (1.0 - damping) * 0.4
        b, a = butter(2, cutoff, 'low')
        ir = lfilter(b, a, ir)

        # Normalize
        ir = ir / np.max(np.abs(ir))
        self.ir = ir

    def reverb(self, y, mix=0.3, room_size=0.8, decay=0.5, damping=0.5,
                diffusion=0.5, early_reflections=True, pre_delay=0.02):
          """
          Apply reverb effect to audio.
    
          Parameters:
          audio (np.array): Input audio signal
          mix (float): Wet/dry mix (0.0 to 1.0)
          Other parameters: Same as generate_ir()
    
          Returns:
          np.array: Processed audio with reverb
          """
          # Generate new IR if parameters changed or IR doesn't exist
          if self.ir is None:
                self.generate_ir(room_size, decay, damping, diffusion,
                              early_reflections, pre_delay)
    
          # Apply convolution
          wet = convolve(y, self.ir, mode='full')[:len(y)]
    
          # Normalize wet signal
          wet = wet / np.max(np.abs(wet))
    
          # Mix wet and dry signals
          output = (1 - mix) * y + mix * wet
    
          # Final normalization
          output = output / np.max(np.abs(output))
    
          return output
    
    def get_presets(self):
        """
        Get preset reverb settings.
        
        Returns:
        dict: Dictionary of preset parameters
        """
        presets = {
            'small_room': {
                'room_size': 0.3,
                'decay': 0.3,
                'damping': 0.5,
                'diffusion': 0.5,
                'early_reflections': True,
                'pre_delay': 0.01
            },
            'large_hall': {
                'room_size': 0.8,
                'decay': 0.8,
                'damping': 0.5,
                'diffusion': 0.5,
                'early_reflections': True,
                'pre_delay': 0.05
            },
            'plate': {
                'room_size': 0.5,
                'decay': 0.5,
                'damping': 0.7,
                'diffusion': 0.7,
                'early_reflections': False,
                'pre_delay': 0.02
            },
            'cathedral': {
                'room_size': 1.0,
                'decay': 0.9,
                'damping': 0.3,
                'diffusion': 0.8,
                'early_reflections': True,
                'pre_delay': 0.1
            }
        }
        return presets