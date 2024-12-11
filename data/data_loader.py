from pathlib import Path
import pickle
import pandas as pd
import mir_eval
import mirdata
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchaudio.prototype.transforms as PT
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils.chord_remap import remap_chord_label, CullMode
import json


class MirDataProcessor:
    def __init__(
        self,
        download=False,
        dataset_name="billboard",
        batch_size=64,
        seq_length=16,
        process_sequential=False,
    ):
        """
        Encapsulates utilities for downloading publicly available MIR datasets and preprocessing them to be
        suitable for model training and testing.
        :param download: flag to determine if data should be downloaded upon instantiation
        :param dataset_name: identifier for the MIR dataset to be downloaded
        :param batch_size: batch used utilized by the pytorch dataloaders
        :param seq_length: length of sequences projected in sequential data processing
        :param process_sequential: flag to determine whether to process the data as sequential or tabular data
        """
        self.raw_data_dir = Path(__file__).parent / "raw"
        self.processed_data_dir = Path(__file__).parent / "processed"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.process_sequential = process_sequential

        # Define file paths
        self.combined_csv_path = self.processed_data_dir / "combined_data.csv"
        self.scaler_path = self.processed_data_dir / "scaler.pkl"
        self.label_encoder_path = self.processed_data_dir / "label_encoder.pkl"

        self.scaler = None
        self.label_encoder = None

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = mirdata.initialize(
            dataset_name, data_home=str(self.raw_data_dir)
        )

        if download:
            self.dataset.download(cleanup=True, force_overwrite=True)

    def process_billboard_data(self, cull_mode=CullMode.REMAP):
        """Processes the raw data and creates a combined CSV file for training."""
        combined_csv_path = self.combined_csv_path

        if combined_csv_path.exists():
            combined_csv_path.unlink()

        track_ids = self.dataset.track_ids
        print(f"Found {len(track_ids)} tracks in the dataset.")

        for track_id in track_ids:
            track = self.dataset.track(track_id)

            if track.chroma is None:
                print(f"Chroma data not available for track {track_id}, skipping.")
                continue

            chroma_data = track.chroma
            chroma_array = chroma_data[:, 1:]
            timestamps = chroma_data[:, 0]

            # Get chord annotations using mirdata
            try:
                # Singular library installed song threw an error when accessing this property
                chord_data = track.chords_full
            except ValueError:
                print(f"Invalid chord data for track {track_id}, skipping.")
                continue

            # Ensure chord annotations are available
            if chord_data is None or len(chord_data.intervals) == 0:
                print(f"No chord annotations for track {track_id}, skipping.")
                continue

            chord_intervals = chord_data.intervals
            chord_labels = chord_data.labels
            for i, chord_label in enumerate(chord_labels):
                remapped_root, remapped_chord_class = remap_chord_label(
                    chord_label, cull_mode
                )
                chord_labels[i] = (
                    "N"
                    if remapped_root == "N"
                    else f"{remapped_root}:{remapped_chord_class}"
                )

            # Use mir_eval to get the chord labels at the chroma timestamps
            # This function maps each timestamp to the corresponding chord label
            labels_at_times = np.array(
                mir_eval.util.interpolate_intervals(
                    chord_intervals, chord_labels, timestamps, fill_value="N"
                )
            )

            if self.process_sequential:
                print("Processing dataset as sequential data")
                # Combine song_id, chroma features, and labels
                song_id_column = np.full((chroma_array.shape[0], 1), track_id)
                data_with_labels = np.hstack(
                    (song_id_column, chroma_array, labels_at_times.reshape(-1, 1))
                )
            else:
                print("Processing dataset as tabular data")
                data_with_labels = np.hstack(
                    (chroma_array, labels_at_times.reshape(-1, 1))
                )

            segment_df = pd.DataFrame(data_with_labels)
            segment_df.to_csv(combined_csv_path, mode="a", index=False, header=False)

            print(f"Processed track {track_id} and appended data to combined CSV.")

        print(f"All data processed and saved to {combined_csv_path}")

    def prepare_model_data(self, nrows=None):
        """Prepares the data for training by loading the combined CSV and processing it."""
        print("Loading the combined CSV file...")
        combined_csv_path = self.combined_csv_path

        combined_df = pd.read_csv(combined_csv_path, header=None, nrows=nrows)
        data = combined_df.values

        if self.process_sequential:
            print("Separating song IDs, features, and labels...")
            # The first column is 'song_id', the last column is 'label', and the rest are features
            song_ids = data[:, 0].astype(str)
            features = data[:, 1:-1].astype(float)
        else:
            print("Separating features and labels...")
            features = data[:, :-1].astype(float)

        labels = data[:, -1].astype(str)

        # Fit scaler and label encoder
        print("Scaling features using MinMaxScaler...")
        self.scaler = MinMaxScaler()
        prepped_features = self.scaler.fit_transform(features)

        print("Encoding labels using LabelEncoder...")
        self.label_encoder = LabelEncoder()
        prepped_labels = self.label_encoder.fit_transform(labels)

        # Save scaler and label encoder for future use
        print(f"Saving the scaler to {self.scaler_path}...")
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Saving the label encoder to {self.label_encoder_path}...")
        with open(self.label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        if self.process_sequential:
            X_sequences = []
            y_sequences = []

            print("Creating sequences of chromagram data within song boundaries...")
            # Group data by song_id
            unique_song_ids = np.unique(song_ids)

            for song_id in unique_song_ids:
                # Get indices for this song
                song_indices = np.where(song_ids == song_id)[0]
                song_features = prepped_features[song_indices]
                song_labels = prepped_labels[song_indices]

                num_samples = song_features.shape[0] - self.seq_length + 1

                if num_samples <= 0:
                    print(
                        f"Song {song_id} has insufficient data for the given sequence length, skipping."
                    )
                    continue

                for i in range(num_samples):
                    X_seq = song_features[i : i + self.seq_length, :]
                    y_seq = song_labels[
                        i + self.seq_length // 2
                    ]  # Using the label at the center of the sequence
                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)

            prepped_features = np.array(X_sequences)
            prepped_labels = np.array(y_sequences)

        # Split data into training and testing sets
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            prepped_features, prepped_labels, test_size=0.2, random_state=42
        )

        print("Data preparation complete.")
        return X_train, X_test, y_train, y_test

    def build_data_loaders(self, nrows=None, device="cuda"):
        """Creates data loaders from the preprocessed model data."""
        print("Preparing model data...")
        X_train, X_test, y_train, y_test = self.prepare_model_data(nrows=nrows)

        # Determine the number of classes
        num_classes = len(self.label_encoder.classes_)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

        # Create datasets
        print("Creating TensorDatasets for training and testing data...")
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create data loaders
        print("Creating DataLoaders for training and testing datasets...")
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=0
        )

        print("Data loaders are ready for training and testing.")
        return train_loader, test_loader, num_classes

    @staticmethod
    def _get_song_metadata(fpath: str) -> dict:
        data = pd.read_csv(fpath, usecols=["id", "artist", "title", "actual_rank"])
        # Retrieve only valid records
        valid_data = data[data["actual_rank"].notna()]
        song_metadata = {
            row["id"]: {
                "artist": row["artist"],
                "title": row["title"],
            }
            for _, row in valid_data.iterrows()
        }
        return song_metadata
    

class ChordDataProcessor:
    def __init__(self, device="cpu", process_sequential=False):
        self.process_sequential = process_sequential
        self.device = device
        
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

        self.features = None
        self.labels = None
    
    def load_audio(self, audio_path: str):
        """Load an audio file and convert to mono if necessary."""
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # If stereo, average channels to make mono
            waveform = waveform.mean(dim=0)
        return waveform, sr

    def compute_chromagram_torchaudio(self, waveform: torch.Tensor, sr: int, n_fft: int = 2048, hop_length: int = 512):
        """
        Compute a chromagram using torchaudio.prototype.transforms.ChromaSpectrogram.
        
        Args:
            waveform: The input audio waveform.
            sr: Sampling rate of the audio.
            n_fft: FFT size.
            hop_length: Hop length for STFT.
            
        Returns:
            chromagram: Chromagram tensor of shape (12, time).
        """
        chroma_transform = PT.ChromaSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_chroma=12,  # 12 pitch classes
        )
        chromagram = chroma_transform(waveform)
        return chromagram
    
    def compute_spectrogram_torchaudio(waveform: torch.Tensor, n_fft: int = 2048, hop_length: int = 512):
        """Compute a spectrogram using torchaudio."""
        spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)(waveform)
        return spectrogram

    def load_chord_data(self, chord_json_path, notation="billboard", mode="chroma", jsontype="keyed", audio_path=str(Path.cwd())):
        #mode picks if the audio data is converted to "chroma" (chromagram) or "spectrogram"
        #json controls the json format because I am dumb and there are 2 types (use "keyed" for chordgen and (i think) fxgen, anything else will set it to texture bias format)
        #audio_path parent for the audio file dir, since the jsons have different formatting this will be different for keyed/entry
        """Loads chord data from the JSON file and extracts features/labels."""
        with open(Path(chord_json_path), "r") as f:
            chord_data = json.load(f)

        features = []
        labels = []
        if jsontype == "keyed":
            for key, value in chord_data.items():
                try:
                    audio_path = audio_path + "/"  + value["filename"]
                    waveform, sr = self.load_audio(audio_path)
                    if mode == "chroma":
                        chromagram = self.compute_chromagram_torchaudio(waveform, sr)
                        features.append(chromagram.numpy())
                    if mode == "spectrogram":
                        spectrogram = PT.Spectrogram()(waveform)
                        features.append(spectrogram.numpy())
                    if notation == "billboard":
                        labels.append(value["billboard_notation"])
                    else:
                        labels.append(value["chord_class"])
                except KeyError as e:
                    print(f"Skipping entry {key} due to missing key: {e}")
        else:
            for entry in chord_data:
                try:
                    audio_path = audio_path + "/" + entry["processed_path"]
                    waveform, sr = self.load_audio(audio_path)
                    if mode == "chroma":
                        chromagram = self.compute_chromagram_torchaudio(waveform, sr)
                        features.append(chromagram.numpy())
                    if mode == "spectrogram":
                        spectrogram = PT.Spectrogram()(waveform)
                        features.append(spectrogram.numpy())
                    if notation == "billboard":
                        labels.append(value["billboard_notation"])
                    else:
                        labels.append(value["chord_class"])
                except KeyError as e:
                    print(f"Skipping entry {entry['filename']} due to missing key: {e}")

        self.features = np.array(features)
        self.labels = np.array(labels)

        print(f"Loaded features shape: {self.features.shape}")
        print(f"Loaded labels shape: {self.labels.shape}")

    def preprocess_data(self):
        """Scales features and encodes labels."""
        self.features = self.scaler.fit_transform(self.features)
        self.labels = self.label_encoder.fit_transform(self.labels)

    def prepare_data(self):
        """Prepares features and labels for sequential or tabular processing."""
        if self.process_sequential:
            X_sequences, y_sequences = [], []
            num_samples = len(self.features) - self.seq_length + 1

            if num_samples <= 0:
                raise ValueError("Not enough data for the given sequence length.")

            for i in range(num_samples):
                X_seq = self.features[i : i + self.seq_length]
                y_seq = self.labels[i + self.seq_length // 2]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

            self.features = np.array(X_sequences)
            self.labels = np.array(y_sequences)

        print(f"Prepared features shape: {self.features.shape}")
        print(f"Prepared labels shape: {self.labels.shape}")

    def synchronize_features_and_labels(self):
        """Ensures features and labels have the same number of samples."""
        min_length = min(len(self.features), len(self.labels))
        if len(self.features) != len(self.labels):
            print(
                f"Synchronizing features ({len(self.features)}) and labels ({len(self.labels)}) to length {min_length}"
            )
        self.features = self.features[:min_length]
        self.labels = self.labels[:min_length]

    def build_data_loaders(self, batch_size=64, seq_length=16, test_size=0.2, random_state=42):
        """Splits data into training/testing sets and creates DataLoaders."""
        self.synchronize_features_and_labels()  # Ensure consistency

        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, stratify=self.labels, random_state=random_state)
        num_classes = len(self.label_encoder.classes_)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, test_loader, num_classes

    def process_all_and_build_loaders(self, chord_json_path, notation="billboard", mode="chroma", jsontype="keyed", audio_path=str(Path.cwd()), batch_size=64, seq_length=16, test_size=0.2, random_state=42):
        """Combines all steps into one pipeline."""
        self.load_chord_data(chord_json_path, notation, mode, jsontype, audio_path)
        self.preprocess_data()
        self.prepare_data()
        self.synchronize_features_and_labels()
        return self.build_data_loaders(batch_size=batch_size, seq_length=seq_length, test_size=test_size, random_state=random_state)
