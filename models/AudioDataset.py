import os
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import Dataset


# Define Dataset Class
class ChordDataset(Dataset):
    def __init__(self, metadata, audio_dir, chord_class_to_idx, transform=None):
        """
        Args:
            metadata (dict): Metadata loaded from JSON.
            audio_dir (str): Directory containing WAV files.
            transform (callable, optional): Transform to apply to waveform.
        """
        self.metadata = metadata
        self.audio_dir = audio_dir
        self.transform = transform or MelSpectrogram(
            sample_rate=44100, n_fft=1024, hop_length=512, n_mels=64
        )
        self.db_transform = AmplitudeToDB()
        self.chord_class_to_idx = chord_class_to_idx

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the filename and corresponding metadata
        filename = list(self.metadata.keys())[idx]
        file_metadata = self.metadata[filename]

        # Load the WAV file
        file_path = os.path.join(self.audio_dir, f"{filename}.wav")
        waveform, sample_rate = torchaudio.load(file_path)

        # Apply transforms to convert to mel-spectrogram
        spectrogram = self.transform(waveform)
        spectrogram = self.db_transform(spectrogram)  # Convert to decibel scale

        # Normalize the spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

        # Get the label
        chord_class = file_metadata["chord_class"]
        label = self.chord_class_to_idx[chord_class]

        return spectrogram, label