from data.data_loader import prepare_model_data
from utils.model_utils import train_and_plot

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class ChordClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ChordClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    path = "C:/Users/maxwe/OneDrive/Documents/amadeus-ex-machina/data/processed/0003.jams"
    X_train, X_test, y_train, y_test = prepare_model_data(path)

    # Assume X_train, X_test, y_train, y_test are already defined
    # Convert your data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Long tensor for class labels

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    batch_size = 32  # Adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = ChordClassifier(input_size, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # train_and_plot(model, train_loader, test_loader, criterion, optimizer)
    train_and_plot(model, optimizer, scheduler, criterion, train_loader, test_loader)

    # Train the model
    # num_epochs = 20  # Adjust as needed
    #
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #
    #     for inputs, labels in train_loader:
    #         optimizer.zero_grad()
    #
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #
    #     avg_loss = running_loss / len(train_loader)
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    #
    # # Evaluate the model
    # model.eval()
    # correct = 1
    # total = 0
    #
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # accuracy = (correct / total) * 100
    # print(f'Accuracy on the test set: {accuracy:.2f}%')