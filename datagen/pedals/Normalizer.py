import numpy as np

class Dynamics:   
    @staticmethod 
    def process_dynamics(audio, threshold_db=-40, ratio=2.0, attack_ms=10, release_ms=100, 
                        knee_db=6.0, makeup_gain_db=0, mode='upward'):
        """
        Process audio dynamics with expansion for quiet signals and optional limiting.
        
        Parameters:
        audio (np.array): Input audio signal
        threshold_db (float): Threshold where expansion begins
        ratio (float): Expansion ratio (1.0 = no expansion, 2.0 = double the range below threshold)
        attack_ms (float): Attack time in milliseconds
        release_ms (float): Release time in milliseconds
        knee_db (float): Knee width in dB for smoother transition
        makeup_gain_db (float): Additional makeup gain in dB
        mode (str): 'upward' or 'downward' expansion
        
        Returns:
        np.array: Processed audio signal
        """
        # Ensure audio is numpy array
        audio = np.array(audio, dtype=np.float32)
        
        # Convert parameters from dB
        threshold = 10 ** (threshold_db / 20)
        knee_width = 10 ** (knee_db / 20)
        makeup_gain = 10 ** (makeup_gain_db / 20)
        
        # Calculate time constants
        sample_rate = 44100  # Assuming 44.1kHz
        attack_samples = int(attack_ms * sample_rate / 1000)
        release_samples = int(release_ms * sample_rate / 1000)
        
        # Initialize gain envelope
        env = np.zeros_like(audio)
        gain = np.ones_like(audio)
        
        # Compute signal envelope
        if len(audio.shape) > 1:
            # Stereo
            env_data = np.maximum(np.abs(audio[:, 0]), np.abs(audio[:, 1]))
        else:
            # Mono
            env_data = np.abs(audio)
        
        # Envelope follower with attack and release
        for i in range(1, len(env)):
            if env_data[i] > env[i-1]:
                # Attack
                env[i] = env[i-1] + (1 - np.exp(-1/attack_samples)) * (env_data[i] - env[i-1])
            else:
                # Release
                env[i] = env[i-1] + (1 - np.exp(-1/release_samples)) * (env_data[i] - env[i-1])
        
        # Compute gain with soft knee
        for i in range(len(env)):
            level_db = 20 * np.log10(max(env[i], 1e-6))
            threshold_db_val = threshold_db
            
            if mode == 'upward':
                # Upward expansion (boost quiet signals)
                if level_db < threshold_db_val - knee_db/2:
                    # Below knee
                    gain_db = (threshold_db_val - level_db) * (ratio - 1)
                elif level_db > threshold_db_val + knee_db/2:
                    # Above knee
                    gain_db = 0
                else:
                    # In knee
                    knee_factor = ((level_db - threshold_db_val + knee_db/2) / knee_db) ** 2
                    gain_db = (threshold_db_val - level_db) * (ratio - 1) * (1 - knee_factor)
            else:
                # Downward expansion (reduce very quiet signals further)
                if level_db < threshold_db_val - knee_db/2:
                    # Below knee
                    gain_db = (threshold_db_val - level_db) * (-ratio)
                elif level_db > threshold_db_val + knee_db/2:
                    # Above knee
                    gain_db = 0
                else:
                    # In knee
                    knee_factor = ((level_db - threshold_db_val + knee_db/2) / knee_db) ** 2
                    gain_db = (threshold_db_val - level_db) * (-ratio) * (1 - knee_factor)
            
            gain[i] = 10 ** (gain_db / 20)
        
        # Apply gain and makeup gain
        if len(audio.shape) > 1:
            # Stereo
            processed = np.zeros_like(audio)
            processed[:, 0] = audio[:, 0] * gain * makeup_gain
            processed[:, 1] = audio[:, 1] * gain * makeup_gain
        else:
            # Mono
            processed = audio * gain * makeup_gain
        
        return processed

    @staticmethod
    def analyze_audio_level(audio):
        """
        Analyze audio levels to determine if expansion is needed.
        
        Parameters:
        audio (np.array): Input audio signal
        
        Returns:
        dict: Audio statistics
        """
        if len(audio.shape) > 1:
            peak_level = np.max(np.abs(audio))
            rms_level = np.sqrt(np.mean(audio**2))
        else:
            peak_level = np.max(np.abs(audio))
            rms_level = np.sqrt(np.mean(audio**2))
        
        peak_db = 20 * np.log10(max(peak_level, 1e-6))
        rms_db = 20 * np.log10(max(rms_level, 1e-6))
        crest_factor = peak_level / max(rms_level, 1e-6)
        
        return {
            'peak_db': peak_db,
            'rms_db': rms_db,
            'crest_factor': crest_factor
        }
    
    @staticmethod
    def limit_audio(audio, threshold_db=-1.0, release_time_ms=50):
        """
        Simple lookahead limiter to prevent peaks above threshold.
        
        Parameters:
        audio (np.array): Input audio signal
        threshold_db (float): Threshold in dB where limiting begins
        release_time_ms (float): Release time in milliseconds
        
        Returns:
        np.array: Limited audio signal
        """
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Ensure audio is numpy array
        audio = np.array(audio, dtype=np.float32)
        
        # Handle mono or stereo
        if len(audio.shape) > 1:
            # For stereo, limit based on max of both channels
            max_peaks = np.maximum(np.abs(audio[:, 0]), np.abs(audio[:, 1]))
        else:
            max_peaks = np.abs(audio)
        
        # Calculate gain reduction needed
        gain_reduction = np.ones_like(max_peaks)
        mask = max_peaks > threshold_linear
        gain_reduction[mask] = threshold_linear / max_peaks[mask]
        
        # Apply release
        release_samples = int(release_time_ms * 44100 / 1000)  # Assuming 44.1kHz
        if release_samples > 0:
            for i in range(1, len(gain_reduction)):
                if gain_reduction[i] < gain_reduction[i-1]:
                    release_factor = np.exp(-1 / release_samples)
                    gain_reduction[i] = max(gain_reduction[i], 
                                        gain_reduction[i-1] * release_factor)
        
        # Apply gain reduction
        if len(audio.shape) > 1:
            # Stereo
            limited_audio = np.zeros_like(audio)
            limited_audio[:, 0] = audio[:, 0] * gain_reduction
            limited_audio[:, 1] = audio[:, 1] * gain_reduction
        else:
            # Mono
            limited_audio = audio * gain_reduction
        
        return limited_audio
    
analyze_audio_level = Dynamics.analyze_audio_level
limit_audio = Dynamics.limit_audio
process_dynamics = Dynamics.process_dynamics