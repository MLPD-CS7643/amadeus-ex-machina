from pathlib import Path
import pickle
import os
import requests
import tarfile
import yt_dlp

import pandas as pd
import mir_eval
import mirdata
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class MirDataProcessor:
    def __init__(self, download=False, dataset_name="billboard", batch_size=64):
        self.raw_data_dir = Path(__file__).parent / "raw"
        self.processed_data_dir = Path(__file__).parent / "processed"
        self.batch_size = batch_size

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

    def process_data(self):
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

            # Use mir_eval to get the chord labels at the chroma timestamps
            # This function maps each timestamp to the corresponding chord label
            labels_at_times = np.array(
                mir_eval.util.interpolate_intervals(
                    chord_intervals, chord_labels, timestamps, fill_value="N"
                )
            )

            data_with_labels = np.hstack((chroma_array, labels_at_times.reshape(-1, 1)))

            segment_df = pd.DataFrame(data_with_labels)
            segment_df.to_csv(combined_csv_path, mode="a", index=False, header=False)

            print(f"Processed track {track_id} and appended data to combined CSV.")

        print(f"All data processed and saved to {combined_csv_path}")

    def prepare_model_data(self):
        """Prepares the data for training by loading the combined CSV and processing it."""
        print("Loading the combined CSV file...")
        combined_csv_path = self.combined_csv_path

        combined_df = pd.read_csv(combined_csv_path, header=None)
        data = combined_df.values

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

    def build_data_loaders(self, device="cuda"):
        """Creates data loaders from the preprocessed model data."""
        print("Preparing model data...")
        X_train, X_test, y_train, y_test = self.prepare_model_data()

        # Determine the number of classes
        num_classes = len(set(y_train))  # Unique labels in the training set
        print(f"Number of classes determined: {num_classes}")

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

def download_and_extract(url, download_path, extract_to):
    """
    Download a tar.xz file from a URL and extract its contents.

    Args:
        url (str): The URL of the file to download.
        download_path (str): The local path to save the downloaded file.
        extract_to (str): The directory where the contents should be extracted.
    """
    try:
        # Download the file
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded file saved to {download_path}.")

        # Extract the tar.xz file
        print(f"Extracting {download_path} to {extract_to}...")
        with tarfile.open(download_path, "r:xz") as tar:
            tar.extractall(path=extract_to)
        print(f"Extraction complete! Files extracted to {extract_to}")

        # Optional: Remove the downloaded tar.xz file to save space
        os.remove(download_path)
        print(f"Removed the downloaded file: {download_path}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")
    except (tarfile.TarError, Exception) as e:
        print(f"An error occurred during extraction: {e}")

def parse_lab_file(lab_file_path):
    """
    Parse a .lab file to extract the title and artist.
    """
    title = artist = None
    with open(lab_file_path, 'r') as file:
        for line in file:
            if line.startswith("# title:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("# artist:"):
                artist = line.split(":", 1)[1].strip()
            if title and artist:
                break
    return title, artist

def download_audio_from_youtube(query, output_path):
    """
    Search for a song on YouTube and download its audio as an MP3 file.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_results = ydl.extract_info(f"ytsearch:{query}", download=False)
            if 'entries' in search_results and search_results['entries']:
                video_url = search_results['entries'][0]['url']
                ydl.download([video_url])
                print(f"Downloaded audio as {output_path}")
                return output_path
            else:
                print("No results found on YouTube.")
                return None
    except Exception as e:
        print(f"An error occurred during YouTube download: {e}")
        return None

def process_lab_files(base_directory):
    """
    Traverse through folders in the base directory, process .lab files, and download audio.
    """
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".txt"):
                lab_file_path = os.path.join(root, file)
                print(f"Processing lab file: {lab_file_path}")

                # Parse the lab file for song title and artist
                title, artist = parse_lab_file(lab_file_path)
                if not title or not artist:
                    print(f"Skipping {lab_file_path}: Missing title or artist")
                    continue

                # Generate a YouTube query and output path for MP3
                query = f"{title} {artist}"
                output_mp3_path = os.path.join(root, f"{title} - {artist}.mp3")

                # Skip if the file already exists
                if os.path.exists(output_mp3_path):
                    print(f"File already exists: {output_mp3_path}")
                    continue

                # Download the audio from YouTube
                download_audio_from_youtube(query, output_mp3_path)