import random

import torch
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm_notebook

RANDOM_SEED = 42


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def train(model, dataloader, optimizer, criterion, device="cpu"):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return total_loss, avg_loss, accuracy


def evaluate(model, dataloader, criterion, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return total_loss, avg_loss, accuracy


def plot_curves(train_accuracy_history, valid_accuracy_history, filename):
    epochs = [i + 1 for i in range(len(train_accuracy_history))]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_accuracy_history,
            mode="lines+markers",
            name="Training Accuracy",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=valid_accuracy_history,
            mode="lines+markers",
            name="Validation Accuracy",
        )
    )

    fig.update_layout(
        title="Accuracy Curve - " + filename,
        xaxis_title="Epochs",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
        legend_title="Dataset",
    )

    fig.write_image(filename + ".png")
    fig.show()


def train_and_plot(
    model,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    valid_loader,
    epochs=20,
    device="cpu",
    figure_name="",
):
    if not figure_name:
        figure_name = f"{model.__class__.__name__}_learning_curve"
        
    train_accuracy_history = []
    valid_accuracy_history = []
    
    best_val_accuracy = 0

    for epoch_idx in range(epochs):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx + 1))
        print("-----------------------------------")

        train_loss, avg_train_loss, train_accuracy = train(
            model, train_loader, optimizer, criterion, device=device
        )
        scheduler.step(avg_train_loss)

        val_loss, avg_val_loss, val_accuracy = evaluate(
            model, valid_loader, criterion, device=device
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best_model.pth")

        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(val_accuracy)

        print(
            f"Training Loss: {avg_train_loss:.4f}. Validation Loss: {avg_val_loss:.4f}."
        )
        print(
            f"Training Accuracy: {train_accuracy:.4f}. Validation Accuracy: {val_accuracy:.4f}."
        )

    plot_curves(train_accuracy_history, valid_accuracy_history, figure_name)
