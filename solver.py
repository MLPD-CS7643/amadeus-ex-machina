from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from models.ChromaTransformer import ChromaTransformerModel 
from models.mlp_chord_classifier import MLPChordClassifier
from models.CNN import CNNModel
from models.CRNN import CRNNModel
from models.RNN import RNNModel
from models.griddy_model import GriddyModel


class Solver:
    def __init__(
        self, model=None, optimizer=None, criterion=None, scheduler=None, **kwargs
    ):
        self.batch_size = kwargs.pop("batch_size", 128)
        self.model_type = kwargs.pop("model_type", "MLPChordClassifier")
        self.model_kwargs = kwargs.pop("model_kwargs", {})
        self.device = kwargs.pop("device", "cpu")
        self.lr = kwargs.pop("learning_rate", 0.001)
        self.epochs = kwargs.pop("epochs", 10)

        if model:
            self.model = model.to(self.device)
        else:
            match self.model_type:
                case "MLPChordClassifier":
                    self.model = MLPChordClassifier(**self.model_kwargs).to(self.device)
                case "CNNModel":
                    self.model = CNNModel(**self.model_kwargs).to(self.device)
                case "CRNNModel":
                    self.model = CRNNModel(**self.model_kwargs).to(self.device)
                case "RNNModel":
                    self.model = RNNModel(**self.model_kwargs).to(self.device)
                case "GriddyModel":
                    self.model = GriddyModel(**self.model_kwargs).to(self.device)
                case "ChromaTransformerModel":
                    self.model = ChromaTransformerModel(**self.model_kwargs).to(self.device)
                case _:
                    # Default to MLPChordClassifier
                    self.model = MLPChordClassifier(**self.model_kwargs)
            self.model.to(self.device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if criterion:
            self.criterion = criterion.to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        self.train_accuracy_history = []
        self.valid_accuracy_history = []
        self.train_loss_history = []
        self.valid_loss_history = []

    @classmethod
    def from_yaml(cls, cfg_path: str, **dynamic_kwargs):
        with open(cfg_path, "r") as fin:
            config = yaml.safe_load(fin)

        kwargs = {}
        for k, v in config.items():
            if k != "description":
                kwargs[k] = v

        if dynamic_kwargs:
            kwargs["model_kwargs"] = dynamic_kwargs

        return cls(**kwargs)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return total_loss, avg_loss, accuracy

    def train_and_evaluate(self, train_loader, valid_loader, plot_results=False):
        best_val_accuracy = 0
        for epoch_idx in range(self.epochs):
            print("-----------------------------------")
            print(f"Epoch {epoch_idx + 1}")
            print("-----------------------------------")

            # Set model to training mode
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Create progress bar for batches
            progress_bar = tqdm(train_loader, desc=f"Training", leave=True)

            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Calculate batch statistics
                _, predicted = torch.max(outputs, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_accuracy = batch_correct / labels.size(0)

                # Update epoch statistics
                total_loss += loss.item() * inputs.size(0)
                total_correct += batch_correct
                total_samples += labels.size(0)

                # Update progress bar description
                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "accuracy": f"{batch_accuracy:.4f}"}
                )

            # Calculate epoch-level training metrics
            avg_train_loss = total_loss / total_samples
            train_accuracy = total_correct / total_samples

            # Evaluate on validation set with progress bar
            val_loss, avg_val_loss, val_accuracy = self.evaluate(valid_loader)

            if self.scheduler:
                self.scheduler.step(avg_val_loss)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(
                    self.model.state_dict(),
                    f"{Path(__file__).parent}/models/checkpoints/{self.model.__class__.__name__}_best_model.pth",
                )

            self.train_accuracy_history.append(train_accuracy)
            self.valid_accuracy_history.append(val_accuracy)

            print(
                f"Training Loss: {avg_train_loss:.4f}. Validation Loss: {avg_val_loss:.4f}."
            )
            print(
                f"Training Accuracy: {train_accuracy:.4f}. Validation Accuracy: {val_accuracy:.4f}."
            )

        if plot_results:
            self.plot_curves(self.model.__class__.__name__)

    def plot_curves(self, file_prefix):
        epochs = [i + 1 for i in range(len(self.train_accuracy_history))]

        # Plot accuracy curves
        plt.figure(figsize=(8, 6))
        plt.plot(
            epochs, self.train_accuracy_history, marker="o", label="Training Accuracy"
        )
        plt.plot(
            epochs, self.valid_accuracy_history, marker="s", label="Validation Accuracy"
        )
        plt.title("Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f"{Path(__file__).parent}/figures/{file_prefix}_accuracy.png")
        plt.show()

        # Plot loss curves
        plt.figure(figsize=(8, 6))
        plt.plot(
            epochs, self.train_loss_history, marker="o", label="Training Loss"
        )
        plt.plot(
            epochs, self.valid_loss_history, marker="s", label="Validation Loss"
        )
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # Usually, no fixed ylim for loss, as it can vary widely
        plt.legend()
        plt.savefig(f"{Path(__file__).parent}/figures/{file_prefix}_loss.png")
        plt.show()

    def run_inference(
        self,
        chroma_csv_path,
        scaler,
        label_encoder,
        output_lab_path="output_annotations.lab",
    ):
        # Read CSV and drop the junk column
        chroma_df = pd.read_csv(chroma_csv_path, header=None)
        chroma_df = chroma_df.drop(chroma_df.columns[0], axis=1)

        # Extract timestamps from the first column after dropping the junk column
        timestamps = chroma_df.iloc[:, 0].values

        features = chroma_df.iloc[:, 1:].values

        # Scale the features using the provided scaler
        features_scaled = scaler.transform(features)

        seq_length = self.model.seq_length  # Assuming the model has this attribute

        num_frames = features_scaled.shape[0]
        num_sequences = num_frames - seq_length + 1

        if num_sequences <= 0:
            print(f"Input data is too short for the given sequence length of {seq_length}.")
            return

        X_sequences = []

        for i in range(num_sequences):
            X_seq = features_scaled[i:i+seq_length, :]
            X_sequences.append(X_seq)

        X_sequences = np.array(X_sequences)  # Shape: (num_sequences, seq_length, input_dim)

        # Convert to tensor and move to device
        X_sequences_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(self.device)

        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_sequences_tensor)
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
                chord_label = 'N'
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

        # Write annotations to the output lab file
        with open(output_lab_path, "w") as f:
            for annotation in annotations:
                start_time, end_time, chord_label = annotation
                f.write(f"{start_time:.4f} {end_time:.4f} {chord_label}\n")

        print(f"Chord annotations saved to {output_lab_path}")

    # def run_inference(
    #     self,
    #     chroma_csv_path,
    #     scaler,
    #     label_encoder,
    #     output_lab_path="output_annotations.lab",
    # ):
    #     # Read CSV and drop the junk column
    #     chroma_df = pd.read_csv(chroma_csv_path, header=None)
    #     chroma_df = chroma_df.drop(chroma_df.columns[0], axis=1)

    #     # Extract timestamps from the first column after dropping the junk column
    #     timestamps = chroma_df.iloc[
    #         :, 0
    #     ].values

    #     features = chroma_df.iloc[:, 1:].values

    #     # Scale the features using the provided scaler
    #     features_scaled = scaler.transform(features)

    #     # Convert features to tensor and move to the appropriate device
    #     features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(
    #         self.device
    #     )

    #     # Run inference
    #     self.model.eval()
    #     with torch.no_grad():
    #         outputs = self.model(features_tensor)
    #         _, predicted_classes = torch.max(outputs, 1)

    #     predicted_classes = predicted_classes.cpu().numpy()
    #     predicted_labels = label_encoder.inverse_transform(predicted_classes)

    #     # Use the timestamps from the data to construct annotations
    #     annotations = []
    #     for i in range(len(timestamps) - 1):
    #         start_time = timestamps[i]
    #         end_time = timestamps[i + 1]
    #         chord_label = predicted_labels[i]
    #         annotations.append((start_time, end_time, chord_label))

    #     # Averaging out distance between each timestamp to approximate frame timing
    #     frame_duration = np.mean(np.diff(timestamps))

    #     final_frame_start = timestamps[-1]
    #     final_frame_end = final_frame_start + frame_duration
    #     chord_label = predicted_labels[-1]
    #     annotations.append((final_frame_start, final_frame_end, chord_label))

    #     # Write annotations to the output lab file
    #     with open(output_lab_path, "w") as f:
    #         for annotation in annotations:
    #             start_time, end_time, chord_label = annotation
    #             f.write(f"{start_time:.4f} {end_time:.4f} {chord_label}\n")

    #     print(f"Chord annotations saved to {output_lab_path}")
