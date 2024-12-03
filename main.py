import sys

from data.data_loader import MirDataProcessor
from solver import Solver


if __name__ == "__main__":
    # Pass the path to the model config you are using and an associated chroma file you want to run inference on
    model_config = sys.argv[1]
    chroma_path = sys.argv[2]

    data_processor = MirDataProcessor(batch_size=128)

    print("Processing data and building data loaders...")
    train_loader, test_loader = data_processor.build_data_loaders()

    input_size = train_loader.dataset.tensors[0].shape[1]

    # Get number of classes from the label encoder
    num_classes = len(data_processor.label_encoder.classes_)

    # Define model arguments
    # TODO this needs to be...less ugly
    model_kwargs = {
        "input_size": input_size,
        "num_classes": num_classes,
    }

    # Initialize solver class
    solver = Solver.from_yaml(model_config, **model_kwargs)

    # Train and evaluate the model
    solver.train_and_evaluate(train_loader, test_loader, plot_results=True)

    scaler = data_processor.scaler
    label_encoder = data_processor.label_encoder

    # Run inference using the trained model
    solver.run_inference(
        chroma_path,
        scaler,
        label_encoder,
    )
