import yaml
import copy
import torch
import optuna
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader


class TrialMetric(Enum):
    LOSS = 0
    ACCURACY = 1
    # add more as needed


class Solver:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        batch_size,
        epochs,
        device="cpu",
        direction="minimize",
        early_stop_epochs=0,
        warmup_epochs=0,
        dtype="float16",
        optuna_prune=False,
        **kwargs,
    ):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        self.epochs = epochs
        self.warmup_epochs = warmup_epochs  # 0 = disable
        self.early_stop_epochs = early_stop_epochs  # 0 = disable

        self.optuna_prune = optuna_prune
        self.direction = direction  # direction to optimize loss function, not used atm but needed for griddy

        self.train_dataloader = DataLoader(
            train_dataloader.dataset, batch_size=self.batch_size, shuffle=True
        )  # workaround to allow griddy of batch_size
        self.valid_dataloader = valid_dataloader

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # if self.dtype == 'float16':
        #     self.model = self.model.half()
        # elif self.dtype == 'bfloat16':
        #     self.model = self.model.bfloat16()

        self.base_lr = optimizer.param_groups[0]["lr"]
        self.train_accuracy_history = []
        self.valid_accuracy_history = []
        self.train_loss_history = []
        self.valid_loss_history = []

        self.best_model = None

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

    def train_and_evaluate(
        self,
        trial=None,
        trial_metric=TrialMetric.LOSS,
        plot_results=False,
        verbose=True,
    ):
        train_loader = self.train_dataloader
        valid_loader = self.valid_dataloader

        no_improve = 0
        best_loss = float("inf")
        best_val_accuracy = 0
        for epoch_idx in range(self.epochs):
            if verbose:
                print("-----------------------------------")
                print(f"Epoch {epoch_idx + 1}")
                print("-----------------------------------")

            if epoch_idx < self.warmup_epochs:
                self.__lr_warmup(epoch_idx + 1)

            # Set model to training mode
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            loader = (
                tqdm(train_loader, desc=f"Training", leave=True)
                if verbose
                else train_loader
            )

            for inputs, labels in loader:
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
                if verbose:
                    loader.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "accuracy": f"{batch_accuracy:.4f}",
                        }
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
                self.best_model = copy.deepcopy(self.model)
                torch.save(
                    self.model.state_dict(),
                    f"{Path(__file__).parent}/models/checkpoints/{self.model.__class__.__name__}_best_model.pth",
                )

            self.train_accuracy_history.append(train_accuracy)
            self.valid_accuracy_history.append(val_accuracy)
            self.train_loss_history.append(avg_train_loss)
            self.valid_loss_history.append(avg_val_loss)

            if verbose:
                print(
                    f"Training Loss: {avg_train_loss:.4f}. Validation Loss: {avg_val_loss:.4f}."
                )
                print(
                    f"Training Accuracy: {train_accuracy:.4f}. Validation Accuracy: {val_accuracy:.4f}."
                )
            # Optuna injection
            if trial:
                match trial_metric:
                    case TrialMetric.LOSS:
                        trial.report(val_loss, epoch_idx + 1)
                    case TrialMetric.ACCURACY:
                        trial.report(val_accuracy, epoch_idx + 1)
                trial.set_user_attr(f"train_loss_epoch_{epoch_idx + 1}", avg_train_loss)
                trial.set_user_attr(f"val_loss_epoch_{epoch_idx + 1}", avg_val_loss)
                trial.set_user_attr(f"train_acc_epoch_{epoch_idx + 1}", train_accuracy)
                trial.set_user_attr(f"val_acc_epoch_{epoch_idx + 1}", val_accuracy)
                if self.optuna_prune and trial.should_prune():
                    print(
                        "OPTUNA PRUNED E:{} L:{:.4f}".format(
                            epoch_idx + 1, avg_val_loss
                        )
                    )
                    raise optuna.exceptions.TrialPruned()
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if self.early_stop_epochs > 0:
                    if no_improve >= self.early_stop_epochs:
                        print(
                            "EARLY STOP E:{} L:{:.4f}".format(
                                epoch_idx + 1, avg_val_loss
                            )
                        )
                        break

        if plot_results:
            self.plot_curves(self.model.__class__.__name__)

        match trial_metric:
            case TrialMetric.LOSS:
                return best_loss
            case TrialMetric.ACCURACY:
                return best_val_accuracy

    def __lr_warmup(self, epoch):
        """Adjusts the learning rate according to the epoch during the warmup phase."""
        lr = self.base_lr * (epoch / self.warmup_epochs)  # Linear warm-up
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

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
        plt.plot(epochs, self.train_loss_history, marker="o", label="Training Loss")
        plt.plot(epochs, self.valid_loss_history, marker="s", label="Validation Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # Usually, no fixed ylim for loss, as it can vary widely
        plt.legend()
        plt.savefig(f"{Path(__file__).parent}/figures/{file_prefix}_loss.png")
        plt.show()
