import os
from pathlib import Path
import argparse

import pandas as pd
import jams
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


BILLBOARD_INDEX_CSV_FPATH = str(Path(__file__).parent / "raw" / "billboard_index.csv")
MAPPED_DATA_DIR = str(Path(__file__).parent / "raw" / "mapped_data")
PROCESSED_DATA_DIR = str(Path(__file__).parent / "processed")


def _load_chroma_csv(fpath: str) -> pd.DataFrame:
    data = pd.read_csv(fpath, header=None, dtype="float32", usecols=range(1, 26))
    headers = ["timestamp"] + [f"pitch_class_{i + 1}" for i in range(data.shape[1] - 1)]
    data.columns = headers
    return data


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


def _build_jams_file(
    lab_path: str, chroma_csv_fpath: str, jams_out_path: str, metadata: dict = None
) -> None:
    chroma_df = _load_chroma_csv(chroma_csv_fpath)
    timestamps = chroma_df["timestamp"].values
    features = chroma_df.iloc[:, 1:].values

    chords = jams.util.import_lab("chord", lab_path)
    duration = max([obs.time + obs.duration for obs in chords])

    chords.time = 0
    chords.duration = duration
    jam = jams.JAMS()

    jam.file_metadata.duration = duration
    jam.annotations.append(chords)

    if metadata:
        jam.file_metadata.title = metadata.get("title")
        jam.file_metadata.artist = metadata.get("artist")

    feature_ann = jams.Annotation(namespace="vector")
    feature_ann.annotation_metadata = jams.AnnotationMetadata(
        data_source="Extracted Features"
    )

    for timestamp, feature_vector in zip(timestamps, features):
        feature_ann.append(
            time=timestamp,
            duration=0.0,  # Assuming features are instantaneous measurements
            value=feature_vector.tolist(),
            confidence=None,
        )
    jam.annotations.append(feature_ann)

    jam.save(jams_out_path)
    print(f"Saved JAMS file to {jams_out_path}")


def process_billboard_data():
    """Driver for creating all jam files from the raw billboard lab + csv files"""
    song_metadata = _get_song_metadata(BILLBOARD_INDEX_CSV_FPATH)
    for subdir in os.listdir(MAPPED_DATA_DIR):
        subdir_id = int(subdir)  # Removing leading zeroes
        _build_jams_file(
            lab_path=f"{MAPPED_DATA_DIR}/{subdir}/annotations/full.lab",
            chroma_csv_fpath=f"{MAPPED_DATA_DIR}/{subdir}/metadata/bothchroma.csv",
            jams_out_path=f"{PROCESSED_DATA_DIR}/{subdir}.jams",
            metadata=song_metadata.get(subdir_id),
        )


def prepare_model_data(jams_path: str, save_files: bool = False):
    """Reads the previously generated jams file and creates the train/test split data. Optionally saves CSVs"""
    jam = jams.load(jams_path)

    vector_data = [
        (observation.time, observation.value)
        for observation in jam.search(namespace="vector")[0]["data"]
    ]
    timestamps, features = zip(*vector_data)
    features = np.array(features)

    chord_data = [
        (annotation.time, annotation.value)
        for annotation in jam.search(namespace="chord")[0]["data"]
    ]

    labels = []
    chord_index = 0
    for time in timestamps:
        while (
            chord_index < len(chord_data) - 1 and chord_data[chord_index + 1][0] <= time
        ):
            chord_index += 1
        labels.append(chord_data[chord_index][1])

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_encoded, test_size=0.2, random_state=42
    )

    if save_files:
        jams_id = jams_path.split(".")[0]
        output_path = Path(PROCESSED_DATA_DIR) / jams_id
        output_path.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(X_train).to_csv(
            output_path / "X_train.csv", index=False, header=False
        )
        pd.DataFrame(X_test).to_csv(
            output_path / "X_test.csv", index=False, header=False
        )
        pd.DataFrame(y_train).to_csv(
            output_path / "y_train.csv", index=False, header=False
        )
        pd.DataFrame(y_test).to_csv(
            output_path / "y_test.csv", index=False, header=False
        )
        print(f"Train and test data saved to directory: {output_path}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utilities for prepping model data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--process-billboard",
        action="store_true",
        help="Process billboard data to generate ALL jams files",
    )

    group.add_argument(
        "--prepare-model-data",
        metavar="JAMS_PATH",
        type=str,
        help="Prepare model data from specified jams file",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save files when using --prepare-model (ignored for other operations)",
    )

    args = parser.parse_args()
    if args.process_billboard:
        process_billboard_data()
    elif args.prepare_model_data:
        prepare_model_data(args.prepare_model_data, args.save)
