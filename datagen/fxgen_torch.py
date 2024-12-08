import torch
import torchaudio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from datagen.pedals import Delay, Distortion, Reverb, Chorus, Noise
import itertools

BITRATE = 192000

class TorchFXGenerator:
    def __init__(self, audio_dir: Path, metadata_path: Path):
        """
        Initialize FX generator with audio directory and metadata.
        
        Parameters:
        audio_dir (Path): Directory containing input MP3 files
        metadata_path (Path): Path to chord_ref.json
        """
        # Load metadata and validate against audio files
        self.metadata = self._load_data(audio_dir, metadata_path)
        self.fx_metadata = {}
        
        # Initialize effects and their presets
        self.fx_presets = {
            'reverb': Reverb.Reverb().get_presets(),
            'delay': Delay.Delay().get_presets(),
            'distortion': Distortion.Distortion().get_presets(),
            'chorus': Chorus.Chorus().get_presets(),
            'noise': Noise.NoiseGenerator().get_presets()
        }
        
        self.available_fx = list(self.fx_presets.keys())
        self.audio_files = {f.stem: f for f in audio_dir.glob('*.mp3')}
        self._init_preset_maps()

    def _load_data(self, audio_dir: Path, metadata_path: Path) -> Dict:
        """
        Load and validate metadata against available audio files.
        
        Returns:
        Dict: Filtered metadata containing only entries with corresponding audio files
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get available audio files
        audio_files = {f.stem for f in audio_dir.glob('*.mp3')}
        
        # Filter metadata to only include entries with corresponding audio files
        valid_metadata = {k: v for k, v in metadata.items() if k in audio_files}
        
        if not valid_metadata:
            raise ValueError("No matching audio files found for metadata entries")
        
        # Report any mismatches
        missing_audio = set(metadata.keys()) - audio_files
        missing_metadata = audio_files - set(metadata.keys())
        
        if missing_audio:
            print(f"Warning: Missing audio files for metadata entries: {missing_audio}")
        if missing_metadata:
            print(f"Warning: Missing metadata for audio files: {missing_metadata}")
            
        return valid_metadata

    def _calculate_total_combinations(self):
        """Calculate total possible unique combinations of effects and presets."""
        # For each effect, count number of presets + 1 for "none" option
        combinations_per_effect = [(len(presets) + 1) for presets in self.fx_presets.values()]
        
        # Total is product of all possibilities
        total = 1
        for n in combinations_per_effect:
            total *= n
            
        return total

    def _get_unique_random_combination(self, used_combinations):
        """Generate a unique combination not in used_combinations."""
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            fx_chain = [fx for fx in self.available_fx if random.random() > 0.15]
            
            # Generate combination identifier
            combo = {fx: 'none' for fx in self.available_fx}
            
            fx_params = {}
            for fx in fx_chain:
                presets = self.fx_presets[fx]
                preset_name = random.choice(list(presets.keys()))
                fx_params[fx] = presets[preset_name]  # Just store the parameters directly
                combo[fx] = preset_name
            
            # Add preset names separately for metadata
            fx_preset_names = {fx: preset_name for fx, preset_name in combo.items() if preset_name != 'none'}
            
            combo_tuple = tuple(sorted((k, v) for k, v in combo.items()))
            
            if combo_tuple not in used_combinations:
                used_combinations.add(combo_tuple)
                return fx_chain, fx_params, fx_preset_names
                
            attempts += 1
        
        raise ValueError("Could not find unique combination after maximum attempts")

    def _process_audio(self, audio: torch.Tensor, sr: int, fx_chain: List, fx_params: Dict) -> torch.Tensor:
        """Apply FX chain to audio."""
        if not fx_chain:
            return audio

        processed = audio.clone()
        
        for fx in fx_chain:
            audio_np = processed.numpy()
            
            if fx == 'reverb':
                r = Reverb.Reverb(sr=sr)
                processed_np = r.reverb(audio_np, **fx_params[fx])  # Use parameters directly
            elif fx == 'delay':
                d = Delay.Delay(sr=sr)
                processed_np = d.delay(audio_np, **fx_params[fx])
            elif fx == 'distortion':
                d = Distortion.Distortion(sr=sr)
                processed_np = d.distort(audio_np, **fx_params[fx])
            elif fx == 'chorus':
                c = Chorus.Chorus(sr=sr)
                processed_np = c.process(audio_np, **fx_params[fx])
            elif fx == 'noise':
                n = Noise.NoiseGenerator(sr=sr)
                processed_np = n.add_noise(audio_np, **fx_params[fx])
                
            processed = torch.from_numpy(processed_np)
            
        return processed

    def _update_metadata(self, original_key: str, fx_chain: List, fx_params: Dict, fx_preset_names: Dict) -> str:
        """
        Update metadata with FX information while preserving all original metadata.
        Adds new FX-related fields to the copied metadata.
        
        Parameters:
            original_key: Key of the original audio file in metadata
            fx_chain: List of applied effects
            fx_params: Dictionary of effect parameters
            fx_preset_names: Dictionary mapping effects to their preset names
        
        Returns:
            str: New key for the processed audio file
        """
        # Generate numeric code for effects
        fx_code = ""
        for fx in self.available_fx:
            if fx in fx_chain:
                preset_name = fx_preset_names[fx]
                fx_code += str(self.preset_maps[fx][preset_name])
            else:
                fx_code += "0"
        
        # Create new key with fx code
        new_key = f"{original_key}_{fx_code}"
        
        # Create new metadata entry by copying ALL original data
        #self.fx_metadata = {} if not hasattr(self, 'fx_metadata') else self.fx_metadata
        self.fx_metadata[new_key] = self.metadata[original_key].copy()
        
        # Update filename and add FX-related information
        self.fx_metadata[new_key]['original_file'] = self.fx_metadata[new_key]['filename']
        self.fx_metadata[new_key]['filename'] = f"{new_key}.mp3"
        self.fx_metadata[new_key]['fx_code'] = fx_code
        
        # Add effect information
        for fx in self.available_fx:
            if fx in fx_chain:
                self.fx_metadata[new_key][fx] = fx_preset_names[fx]
            else:
                self.fx_metadata[new_key][fx] = "none"
                
        return new_key

    def process_batch(self, output_dir: Path, permutations_per_sample: int = 5, 
                    systematic: bool = False, limit: bool = False):
        """Process all audio files with FX chains."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate maximum possible combinations
        max_combinations = self._calculate_total_combinations()
        print(f"Total possible combinations: {max_combinations}")
        
        if not systematic and permutations_per_sample > max_combinations:
            print(f"Warning: Requested {permutations_per_sample} permutations but only {max_combinations} possible.")
            permutations_per_sample = max_combinations

        for original_key in self.metadata.keys():
            print(f"Processing {original_key}")
            
            audio_path = str(self.audio_files[original_key])
            waveform, sr = torchaudio.load(audio_path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            
            if systematic:
                combinations = self._generate_all_combinations()
            else:
                used_combinations = set()
                combinations = (self._get_unique_random_combination(used_combinations) 
                            for _ in range(permutations_per_sample))
            
            for fx_chain, fx_params, fx_preset_names in combinations:
                processed = self._process_audio(waveform, sr, fx_chain, fx_params)
                new_key = self._update_metadata(original_key, fx_chain, fx_params, fx_preset_names)
                
                output_path = output_dir / f"{new_key}.mp3"
                processed = self._normalize_batch(processed)
                torchaudio.save(
                    str(output_path),
                    processed.unsqueeze(0).to(torch.float32),
                    sr,
                    format="mp3",
                    compression=BITRATE/1000
                )
            
            if limit:
                break
        
        self._save_metadata(output_dir.parent / "fx_chord_ref.json")

    def _normalize_batch(self, batch: torch.Tensor, target_db: float = -1.0) -> torch.Tensor:
        """Normalize a batch of audio to target dB level."""
        max_val = torch.max(torch.abs(batch))
        target_val = 10 ** (target_db / 20)
        
        if max_val > target_val:
            batch = batch * (target_val / max_val)
            
        return batch

    def _init_preset_maps(self):
        """Initialize maps of preset indices for each effect."""
        self.preset_maps = {}
        for fx in self.available_fx:
            # Create map: preset_name -> index (starting from 1)
            # 0 will represent 'none'
            presets = list(self.fx_presets[fx].keys())
            self.preset_maps[fx] = {name: i+1 for i, name in enumerate(presets)}

    def _generate_fx_code(self, fx_chain: List, fx_params: Dict) -> str:
        """Generate numerical code for effects configuration."""
        fx_code = ""
        for fx in self.available_fx:  # Use consistent order
            if fx in fx_chain:
                preset_name = fx_params[fx]['preset_name']
                fx_code += str(self.preset_maps[fx][preset_name])
            else:
                fx_code += "0"
        return fx_code
    
    def _generate_all_combinations(self):
        """Generate all possible combinations of effects and presets systematically."""
        # Create options for each effect (presets + none)
        effect_options = {
            fx: [('none', None)] + [(name, params) 
                for name, params in self.fx_presets[fx].items()]
            for fx in self.available_fx
        }
        
        # Generate all possible combinations using itertools.product
        effect_names = sorted(self.available_fx)  # Sort for consistent ordering
        preset_combinations = itertools.product(*[effect_options[fx] for fx in effect_names])
        
        for combo in preset_combinations:
            # Create fx_chain and fx_params
            fx_chain = []
            fx_params = {}
            fx_preset_names = {}
            
            for fx, (preset_name, params) in zip(effect_names, combo):
                fx_preset_names[fx] = preset_name
                if preset_name != 'none':
                    fx_chain.append(fx)
                    fx_params[fx] = params
                    
            yield fx_chain, fx_params, fx_preset_names

    def _save_metadata(self, path: Path):
        """Save FX metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.fx_metadata, f)
