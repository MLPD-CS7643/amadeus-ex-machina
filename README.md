# Amadeus Ex Machina


---

## Data Processing

This repository includes utilities to preprocess audio data and prepare it for model training. The provided scripts facilitate processing usable model data from the Billboard McGills Dataset. 

### Scripts and Usage

The [MIR Data Library](https://mirdata.readthedocs.io/en/stable/) contains a number of useable datasets, with the primary one being the Billboard McGills Dataset. To begin using the data processing utilities, you need to initialize and download the desired dataset. Here's how you can do it:

1. **Initialize the Processor**: Create an instance of `MirDataProcessor`, specifying the dataset by name if you wish to use a dataset other than Billboard, and specify `download=True` to download the dataset.

2. **Process the Data**: If you are using the Billboard McGills Dataset, you can process the data by calling the `process_data` method. This will create a combined CSV file in the `data/processed` directory containing all of the processed data.
