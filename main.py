import sys

from data.data_loader import prepare_model_data, build_data_loaders
from solver import Solver

import numpy as np


if __name__ == "__main__":
    # Pass the path to whatever jams file / associated chroma file you want to test this on
    jams_path = sys.argv[1]
    chroma_path = sys.argv[2]
    model_config = sys.argv[3]
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_model_data(
        jams_path
    )

    train_loader, test_loader = build_data_loaders(
        jams_path,
        save_files=False,
        batch_size=32,  # FIXME rework the dataloaders to get batch size from the config
    )

    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model_kwargs = {
        "input_size": input_size,
        "num_classes": num_classes,
    }

    solver = Solver.from_yaml(model_config, **model_kwargs)

    solver.train_and_evaluate(train_loader, test_loader, epochs=50, plot_results=True)

    solver.run_inference(
        chroma_path,
        scaler,
        label_encoder,
    )
