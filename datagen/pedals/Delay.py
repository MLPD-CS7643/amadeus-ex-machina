import numpy as np

class Delay:
    def __init__(self, sr=44100):
        """
        Initialize delay with sample rate.
        
        Parameters:
        sr (int): Sample rate
        """
        self.sr = sr
        
    def delay(self, y, delay_time=0.4, feedback=0.5, mix=0.5):
        """
        Apply delay effect to audio signal.
        
        Parameters:
        y (np.array): Input audio signal
        delay_time (float): Delay time in seconds
        feedback (float): Amount of feedback (0.0 to 1.0)
        mix (float): Wet/dry mix (0.0 to 1.0)
        
        Returns:
        np.array: Processed audio with delay effect
        """
        # Calculate delay in samples
        delay_samples = int(self.sr * delay_time)
        
        # Create an empty array for the output
        output = np.zeros(len(y), dtype=np.float32)
        delayed = np.zeros(len(y), dtype=np.float32)
        
        # Add delayed copies with feedback
        for i in range(len(y)):
            if i < delay_samples:
                delayed[i] = 0
            else:
                delayed[i] = y[i - delay_samples] + feedback * delayed[i - delay_samples]
        
        # Mix dry and wet signals
        output = (1 - mix) * y + mix * delayed
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
            
        return output
    
    def get_presets(self):
        """
        Get preset delay settings.
        
        Returns:
        dict: Dictionary of preset parameters
        """
        presets = {
            'short_slap': {
                'delay_time': 0.1,
                'feedback': 0.3,
                'mix': 0.4
            },
            'echo': {
                'delay_time': 0.25,
                'feedback': 0.5,
                'mix': 0.5
            },
            'long_delay': {
                'delay_time': 0.5,
                'feedback': 0.6,
                'mix': 0.5
            },
            'ambient': {
                'delay_time': 0.4,
                'feedback': 0.7,
                'mix': 0.6
            }
        }
        return presets