import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

class Solver:
    def __init__(
        self, model, optimizer, criterion, scheduler, train_dataloader:DataLoader, valid_dataloader:DataLoader, **kwargs
    ):
        self.device = kwargs.pop("device", "cpu")
        self.dtype = kwargs.pop("dtype", "float16")
        self.batch_size = kwargs.pop("batch_size", 64)
        
        self.epochs = kwargs.pop("epochs", 10)
        self.warmup_epochs = kwargs.pop("warmup_epochs", 0) # 0 = disable
        self.early_stop_epochs = kwargs.pop("early_stop_epochs", 0) # 0 = disable

        self.direction = kwargs.pop("direction", "minimize") # direction to optimize loss function, not used atm but needed for griddy

        self.train_dataloader = DataLoader(train_dataloader.dataset, batch_sampler=train_dataloader.batch_sampler, batch_size=self.batch_size)
        self.valid_dataloader = valid_dataloader

        self.model = model.to(self.device)
        self.optimizer = optimizer.to(self.device)
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler.to(self.device)

        if self.dtype == 'float16':
            self.model = self.model.half()
        elif self.dtype == 'bfloat16':
            self.model = self.model.bfloat16()

        self.base_lr = optimizer.param_groups[0]['lr']
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
    
    def train_and_evaluate(self, plot_results=False):
        train_loader = self.train_dataloader
        valid_loader = self.valid_dataloader

        no_improve = 0
        best_val_accuracy = 0
        for epoch_idx in range(self.epochs):
            print("-----------------------------------")
            print(f"Epoch {epoch_idx + 1}")
            print("-----------------------------------")

            if epoch_idx < self.warmup_epochs:
                self.__lr_warmup(epoch_idx+1)

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
                no_improve = 0
                best_val_accuracy = val_accuracy
                torch.save(
                    self.model.state_dict(),
                    f"{Path(__file__).parent}/models/checkpoints/{self.model.__class__.__name__}_best_model.pth",
                )
            else:
                no_improve += 1

            self.train_accuracy_history.append(train_accuracy)
            self.valid_accuracy_history.append(val_accuracy)

            print(
                f"Training Loss: {avg_train_loss:.4f}. Validation Loss: {avg_val_loss:.4f}."
            )
            print(
                f"Training Accuracy: {train_accuracy:.4f}. Validation Accuracy: {val_accuracy:.4f}."
            )

            if self.early_stop_epochs > 0 and no_improve >= self.early_stop_epochs:
                print("EARLY STOP E:{} L:{:.4f}".format(epoch_idx+1, val_accuracy))
                break

        if plot_results:
            self.plot_curves(f"{self.model.__class__.__name__}_accuracy_curve")

    def __lr_warmup(self, epoch):
        """Adjusts the learning rate according to the epoch during the warmup phase."""
        lr = self.base_lr * (epoch / self.warmup_epochs)  # Linear warm-up
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

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
    ):
        # Read CSV and drop the junk column
        chroma_df = pd.read_csv(chroma_csv_path, header=None)
        chroma_df = chroma_df.drop(chroma_df.columns[0], axis=1)

        # Extract timestamps from the first column after dropping the junk column
        timestamps = chroma_df.iloc[
            :, 0
        ].values

        features = chroma_df.iloc[:, 1:].values

        # Scale the features using the provided scaler
        features_scaled = scaler.transform(features)

        # Convert features to tensor and move to the appropriate device
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(
            self.device
        )

        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
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

        # Write annotations to the output lab file
        with open(output_lab_path, "w") as f:
            for annotation in annotations:
                start_time, end_time, chord_label = annotation
                f.write(f"{start_time:.4f} {end_time:.4f} {chord_label}\n")

        print(f"Chord annotations saved to {output_lab_path}")
