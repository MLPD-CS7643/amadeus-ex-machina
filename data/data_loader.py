import os

from pathlib import Path
import pandas as pd
import jams

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
            lab_path=f"{MAPPED_DATA_DIR}/{subdir}/annotations/majmin.lab",
            chroma_csv_fpath=f"{MAPPED_DATA_DIR}/{subdir}/metadata/bothchroma.csv",
            jams_out_path=f"{PROCESSED_DATA_DIR}/{subdir}.jams",
            metadata=song_metadata.get(subdir_id)
        )


if __name__ == "__main__":
    # Leaving this here for testing purposes
    # lab_path = str(
    #     Path(__file__).parent
    #     / "raw"
    #     / "mapped_data"
    #     / "0003"
    #     / "annotations"
    #     / "majmin.lab"
    # )
    # chroma_path = str(
    #     Path(__file__).parent
    #     / "raw"
    #     / "mapped_data"
    #     / "0003"
    #     / "metadata"
    #     / "bothchroma.csv"
    # )
    # billboard_path = str(Path(__file__).parent / "raw" / "billboard_index.csv")
    process_billboard_data()
