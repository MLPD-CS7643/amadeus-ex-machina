from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

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
                    self.model = MLPChordClassifier(**self.model_kwargs)
                case "CNNModel":
                    self.model = CNNModel(**self.model_kwargs)
                case "CRNNModel":
                    self.model = CRNNModel(**self.model_kwargs)
                case "RNNModel":
                    self.model = RNNModel(**self.model_kwargs)
                case "GriddyModel":
                    self.model = GriddyModel(**self.model_kwargs)
                case _:
                    # Default to MLPChordClassifier
                    self.model = MLPChordClassifier(**self.model_kwargs)
            self.model.to(self.device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        self.train_accuracy_history = []
        self.valid_accuracy_history = []

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

    def train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return total_loss, avg_loss, accuracy

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
            self.plot_curves(f"{self.model.__class__.__name__}_accuracy_curve")

    def plot_curves(self, filename):
        epochs = [i + 1 for i in range(len(self.train_accuracy_history))]

        plt.figure(figsize=(8, 6))

        plt.plot(
            epochs, self.train_accuracy_history, marker="o", label="Training Accuracy"
        )
        plt.plot(
            epochs, self.valid_accuracy_history, marker="s", label="Validation Accuracy"
        )

        plt.title("Accuracy Curve - " + filename)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)

        plt.legend()
        plt.savefig(f"{Path(__file__).parent}/figures/{filename}.png")
        plt.show()

    def run_inference(
        self,
        chroma_csv_path,
        scaler,
        label_encoder,
        output_lab_path="output_annotations.lab",
        hop_length=512,
        sr=44100,
    ):
        new_chromagram_df = pd.read_csv(chroma_csv_path, header=None)
        new_features = new_chromagram_df.values

        new_features_scaled = scaler.transform(new_features)
        new_features_tensor = torch.tensor(new_features_scaled, dtype=torch.float32).to(
            self.device
        )

        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(new_features_tensor)
            _, predicted_classes = torch.max(outputs, 1)

        predicted_classes = predicted_classes.cpu().numpy()
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        hop_duration = hop_length / sr  # Duration of each hop/frame
        num_frames = new_features_scaled.shape[0]
        timestamps = np.arange(num_frames) * hop_duration

        # Pair timestamps with predicted labels
        annotations = list(zip(timestamps, predicted_labels))

        with open(output_lab_path, "w") as f:
            for i in range(len(annotations) - 1):
                start_time = annotations[i][0]
                end_time = annotations[i + 1][0]
                chord_label = annotations[i][1]
                f.write(f"{start_time:.4f} {end_time:.4f} {chord_label}\n")

            start_time = annotations[-1][0]
            end_time = start_time + hop_duration
            chord_label = annotations[-1][1]
            f.write(f"{start_time:.4f} {end_time:.4f} {chord_label}\n")

        print(f"Chord annotations saved to {output_lab_path}")
