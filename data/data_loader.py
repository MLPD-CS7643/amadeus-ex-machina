import os

from pathlib import Path
import numpy as np
import pandas as pd
import jams


def load_chroma_csv(fpath: str) -> pd.DataFrame:
    data = pd.read_csv(fpath, header=None).iloc[:, 1:]
    headers = ["timestamp"] + [f"pitch_class_{i + 1}" for i in range(data.shape[1] - 1)]
    data.columns = headers
    return data


def build_jams_file(lab_path: str, chroma_csv_fpath: str, jams_out_path: str, metadata: dict = None) -> None:
    chroma_df = load_chroma_csv(chroma_csv_fpath)
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
        jam.file_metadata.identifiers = {"id": metadata.get("id")}

    feature_ann = jams.Annotation(namespace='vector')
    feature_ann.annotation_metadata = jams.AnnotationMetadata(data_source='Extracted Features')

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


if __name__ == "__main__":
    lab_path = str(
        Path(__file__).parent
        / "raw"
        / "mapped_data"
        / "0003"
        / "annotations"
        / "majmin.lab"
    )
    chroma_path = str(
        Path(__file__).parent
        / "raw"
        / "mapped_data"
        / "0003"
        / "metadata"
        / "bothchroma.csv"
    )
    jams_out_path = str(Path(__file__).parent / "jams_out.jams")
    print(jams_out_path)
    # segments = load_lab_file(str(lab_path))
    # load_chroma_csv(str(chroma_path))
    build_jams_file(lab_path, chroma_path, jams_out_path)
    # print(segments)
