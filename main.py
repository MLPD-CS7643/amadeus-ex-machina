import sys

from data.data_loader import prepare_model_data, build_data_loaders
from models.mlp_chord_classifier import MLPChordClassifier
from solver import Solver

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":
    # Pass the path to whatever jams file / associated chroma file you want to test this on
    jams_path = sys.argv[1]
    chroma_path = sys.argv[2]
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_model_data(
        jams_path
    )

    train_loader, test_loader = build_data_loaders(
        jams_path, save_files=False, batch_size=32
    )

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = MLPChordClassifier(
        input_size, num_classes
    )  # FIXME this is not correct. We need to extract all possible annotations

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    solver = Solver(model, optimizer, criterion, scheduler)
    solver.train_and_evaluate(train_loader, test_loader, epochs=25, plot_results=False)

    solver.run_inference(
        chroma_path,
        scaler,
        label_encoder,
    )
