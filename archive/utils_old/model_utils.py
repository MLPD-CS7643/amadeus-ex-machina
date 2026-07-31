import random

import torch
import pandas as pd
import numpy as np
import pickle

RANDOM_SEED = 42


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_annotations(output_lab_path, annotations):
    with open(output_lab_path, "w") as f:
        for annotation in annotations:
            start_time, end_time, chord_label = annotation
            f.write(f"{start_time:.4f} {end_time:.4f} {chord_label}\n")

    print(f"Chord annotations saved to {output_lab_path}")


def run_sequential_chroma_inference(
    chroma_csv_path,
    model,
    scaler,
    label_encoder,
    output_lab_path="output_annotations.lab",
    device="cuda",
):
    # Read CSV and drop the junk column
    chroma_df = pd.read_csv(chroma_csv_path, header=None)
    chroma_df = chroma_df.drop(chroma_df.columns[0], axis=1)

    # Extract timestamps from the first column after dropping the junk column
    timestamps = chroma_df.iloc[:, 0].values

    features = chroma_df.iloc[:, 1:].values

    # Scale the features using the provided scaler
    features_scaled = scaler.transform(features)

    seq_length = model.seq_length  # Assuming the model has this attribute

    num_frames = features_scaled.shape[0]
    num_sequences = num_frames - seq_length + 1

    if num_sequences <= 0:
        print(f"Input data is too short for the given sequence length of {seq_length}.")
        return

    X_sequences = []

    for i in range(num_sequences):
        X_seq = features_scaled[i : i + seq_length, :]
        X_sequences.append(X_seq)

    X_sequences = np.array(X_sequences)  # Shape: (num_sequences, seq_length, input_dim)

    # Convert to tensor and move to device
    X_sequences_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(X_sequences_tensor)
        _, predicted_classes = torch.max(outputs, 1)

    predicted_classes = predicted_classes.cpu().numpy()
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    # Map predictions back to time steps
    # Assign the predicted label to the center time step of each sequence
    time_step_predictions = [[] for _ in range(num_frames)]

    for i in range(num_sequences):
        center_idx = i + seq_length // 2
        if center_idx < num_frames:
            time_step_predictions[center_idx].append(predicted_labels[i])

    # Determine the final prediction for each time step
    final_predictions = []
    for preds in time_step_predictions:
        if preds:
            # Select the most common predicted label
            chord_label = max(set(preds), key=preds.count)
        else:
            # If no prediction, assign a default label (e.g., 'N' for no chord)
            chord_label = "N"
        final_predictions.append(chord_label)

    # Construct annotations using timestamps and final_predictions
    annotations = []
    for i in range(len(final_predictions) - 1):
        start_time = timestamps[i]
        end_time = timestamps[i + 1]
        chord_label = final_predictions[i]
        annotations.append((start_time, end_time, chord_label))

    # Handle the last frame
    start_time = timestamps[-1]
    frame_duration = np.mean(np.diff(timestamps))
    end_time = start_time + frame_duration
    chord_label = final_predictions[-1]
    annotations.append((start_time, end_time, chord_label))

    write_annotations(output_lab_path, annotations)


def run_tabular_chroma_inference(
    chroma_csv_path,
    model,
    scaler,
    label_encoder,
    output_lab_path="output_annotations.lab",
    device="cuda",
):
    # Read CSV and drop the junk column
    chroma_df = pd.read_csv(chroma_csv_path, header=None)
    chroma_df = chroma_df.drop(chroma_df.columns[0], axis=1)

    # Extract timestamps from the first column after dropping the junk column
    timestamps = chroma_df.iloc[:, 0].values

    features = chroma_df.iloc[:, 1:].values

    # Scale the features using the provided scaler
    features_scaled = scaler.transform(features)

    # Convert features to tensor and move to the appropriate device
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted_classes = torch.max(outputs, 1)

    predicted_classes = predicted_classes.cpu().numpy()
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    # Use the timestamps from the data to construct annotations
    annotations = []
    for i in range(len(timestamps) - 1):
        start_time = timestamps[i]
        end_time = timestamps[i + 1]
        chord_label = predicted_labels[i]
        annotations.append((start_time, end_time, chord_label))

    # Averaging out distance between each timestamp to approximate frame timing
    frame_duration = np.mean(np.diff(timestamps))

    final_frame_start = timestamps[-1]
    final_frame_end = final_frame_start + frame_duration
    chord_label = predicted_labels[-1]
    annotations.append((final_frame_start, final_frame_end, chord_label))

    write_annotations(output_lab_path, annotations)


def save_history(solver, filename="history.pkl"):
    # Create a DataFrame from the history lists
    history_df = pd.DataFrame(
        {
            "Train Accuracy": solver.train_accuracy_history,
            "Validation Accuracy": solver.valid_accuracy_history,
            "Train Loss": solver.train_loss_history,
            "Validation Loss": solver.valid_loss_history,
        }
    )

    # Pickle the DataFrame to the specified file
    with open(filename, "wb") as file:
        pickle.dump(history_df, file)

    print(f"History saved to {filename}")


def load_history(solver, filename="history.pkl"):
    # Load the pickled DataFrame from the file
    with open(filename, "rb") as file:
        history_df = pickle.load(file)

    # If the class attributes need to be repopulated from the DataFrame:
    solver.train_accuracy_history = history_df["Train Accuracy"].tolist()
    solver.valid_accuracy_history = history_df["Validation Accuracy"].tolist()
    solver.train_loss_history = history_df["Train Loss"].tolist()
    solver.valid_loss_history = history_df["Validation Loss"].tolist()

    print(f"History loaded from {filename}")
