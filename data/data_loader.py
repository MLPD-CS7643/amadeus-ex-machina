from pathlib import Path
import pickle
import pandas as pd
import mir_eval
import mirdata
import torch
from torch.utils.data import TensorDataset, DataLoader
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
    def __init__(self, chord_json_path, batch_size=64, seq_length=16, device="cpu", process_sequential=False):
        """
        Processes the chord data from a JSON file and prepares DataLoaders for training/testing.

        Args:
            chord_json_path (str): Path to the JSON file containing chord data.
            batch_size (int): Batch size for DataLoaders.
            seq_length (int): Sequence length for sequential data processing.
            process_sequential (bool): Whether to process the data as sequential or tabular.
        """
        self.chord_json_path = Path(chord_json_path)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.process_sequential = process_sequential

        self.device = device

        # Scaler and label encoder
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

        # Processed data placeholders
        self.features = None
        self.labels = None

    def load_chord_data(self):
        """Loads chord data from the JSON file and extracts features/labels."""
        with open(self.chord_json_path, "r") as f:
            chord_data = json.load(f)

        features = []
        labels = []

        for key, value in chord_data.items():
            # Extract features (e.g., duration, sample rate, etc.) and labels (e.g., chord class)
            feature_vector = [
                value["octave"],
                value["gm_preset_id"],
                value["duration(s)"],
                value["sample_rate"],
                value["bit_depth"],
            ]
            label = value["chord_class"]  # Use chord class as the label
            features.append(feature_vector)
            labels.append(label)

        self.features = np.array(features)
        self.labels = np.array(labels)

    def preprocess_data(self):
        """Scales features and encodes labels."""
        print("Scaling features using MinMaxScaler...")
        self.features = self.scaler.fit_transform(self.features)

        print("Encoding labels using LabelEncoder...")
        self.labels = self.label_encoder.fit_transform(self.labels)

    def prepare_data(self):
        """Prepares features and labels for sequential or tabular processing."""
        if self.process_sequential:
            print("Processing data as sequential data...")
            X_sequences, y_sequences = [], []

            num_samples = len(self.features) - self.seq_length + 1
            if num_samples <= 0:
                raise ValueError("Not enough data for the given sequence length.")

            for i in range(num_samples):
                X_seq = self.features[i : i + self.seq_length]
                y_seq = self.labels[i + self.seq_length // 2]  # Label at the center of the sequence
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

            self.features = np.array(X_sequences)
            self.labels = np.array(y_sequences)
        else:
            print("Processing data as tabular data...")

    def build_data_loaders(self, test_size=0.2, random_state=42):
        """
        Splits the data into training/testing sets and creates DataLoaders.

        Args:
            test_size (float): Proportion of the data to use as the test set.
            random_state (int): Random seed for reproducibility.
            device (str): Device to load data onto ('cuda' or 'cpu').

        Returns:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            num_classes (int): Number of unique classes.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=random_state
        )

        # Determine the number of classes
        num_classes = len(self.label_encoder.classes_)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=self.device)

        # Create TensorDatasets
        print("Creating TensorDatasets for training and testing data...")
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create DataLoaders
        print("Creating DataLoaders for training and testing datasets...")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        print("DataLoaders are ready for training and testing.")
        return train_loader, test_loader, num_classes

    def process_all_and_build_loaders(self, test_size=0.2, random_state=42):
        """
        Combines loading, preprocessing, and preparing data, and returns DataLoaders.

        Args:
            test_size (float): Proportion of the data to use as the test set.
            random_state (int): Random seed for reproducibility.
            device (str): Device to load data onto ('cuda' or 'cpu').

        Returns:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            num_classes (int): Number of unique classes.
        """
        print("Starting full processing pipeline...")
        self.load_chord_data()
        self.preprocess_data()
        self.prepare_data()
        return self.build_data_loaders(test_size=test_size, random_state=random_state)
