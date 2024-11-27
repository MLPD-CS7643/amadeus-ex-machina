import os
from pathlib import Path
import pickle

import pandas as pd
import jams
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class BillboardDataProcessor:
    def __init__(self, batch_size=64):
        self.raw_data_dir = Path(__file__).parent / "raw"
        self.processed_data_dir = Path(__file__).parent / "processed"
        self.batch_size = batch_size

        # Define file paths
        self.billboard_index_csv_fpath = self.raw_data_dir / "billboard_index.csv"
        self.mapped_data_dir = self.raw_data_dir / "mapped_data"
        self.combined_csv_path = self.processed_data_dir / "combined_data.csv"
        self.scaler_path = self.processed_data_dir / "scaler.pkl"
        self.label_encoder_path = self.processed_data_dir / "label_encoder.pkl"

        self.scaler = None | MinMaxScaler
        self.label_encoder = None | LabelEncoder

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_chroma_csv(fpath: str) -> pd.DataFrame:
        data = pd.read_csv(fpath, header=None, dtype="float32", usecols=range(1, 26))
        headers = ["timestamp"] + [
            f"pitch_class_{i + 1}" for i in range(data.shape[1] - 1)
        ]
        data.columns = headers
        return data

    def process_billboard_data(self):
        """Processes the raw data and creates a combined CSV file for training."""
        combined_csv_path = self.combined_csv_path

        # Remove the existing combined CSV if it exists
        if combined_csv_path.exists():
            combined_csv_path.unlink()

        for subdir in os.listdir(self.mapped_data_dir):
            subdir_id = str(int(subdir))
            lab_path = self.mapped_data_dir / subdir / "annotations" / "full.lab"
            chroma_csv_fpath = (
                self.mapped_data_dir / subdir / "metadata" / "bothchroma.csv"
            )

            # Read chroma features
            chroma_df = self._load_chroma_csv(str(chroma_csv_fpath))
            timestamps = chroma_df["timestamp"].values
            features = chroma_df.iloc[:, 1:].values

            # Read chord annotations
            chords = jams.util.import_lab("chord", str(lab_path))
            chord_data = [(obs.time, obs.value) for obs in chords]

            # Align features and labels
            labels = []
            chord_index = 0
            for time in timestamps:
                while (
                    chord_index < len(chord_data) - 1
                    and chord_data[chord_index + 1][0] <= time
                ):
                    chord_index += 1
                labels.append(chord_data[chord_index][1])

            labels = np.array(labels)

            # TODO look into more robust ways of semantically preserving individual song structure
            song_data = np.hstack((features, labels.reshape(-1, 1)))
            song_df = pd.DataFrame(song_data)

            # Append to the combined CSV
            song_df.to_csv(combined_csv_path, mode="a", index=False, header=False)

            print(f"Processed song {subdir_id} and appended data to combined CSV.")

        print(f"All data processed and saved to {combined_csv_path}")

    def prepare_model_data(self):
        """Prepares the data for training by loading the combined CSV and processing it."""
        print("Loading the combined CSV file...")
        combined_csv_path = self.combined_csv_path

        # Load the combined data
        combined_df = pd.read_csv(combined_csv_path, header=None)
        data = combined_df.values

        # Separate features and labels
        print("Separating features and labels...")
        features = data[:, :-1].astype(float)
        labels = data[:, -1].astype(str)

        # Fit scaler and label encoder
        print("Scaling features using MinMaxScaler...")
        self.scaler = MinMaxScaler()
        features_scaled = self.scaler.fit_transform(features)

        print("Encoding labels using LabelEncoder...")
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Save scaler and label encoder for future use
        print(f"Saving the scaler to {self.scaler_path}...")
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Saving the label encoder to {self.label_encoder_path}...")
        with open(self.label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        # Split data into training and testing sets
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, test_size=0.2, random_state=42
        )

        print("Data preparation complete.")
        return X_train, X_test, y_train, y_test


    def build_data_loaders(self):
        """Creates data loaders from the preprocessed model data."""
        print("Preparing model data...")
        X_train, X_test, y_train, y_test = self.prepare_model_data()

        # Determine the number of classes
        num_classes = len(set(y_train))  # Unique labels in the training set
        print(f"Number of classes determined: {num_classes}")

        # Convert to PyTorch tensors
        print("Converting training and testing data to PyTorch tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create datasets
        print("Creating TensorDatasets for training and testing data...")
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create data loaders
        print("Creating DataLoaders for training and testing datasets...")
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

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