# Amadeus Ex Machina


---

## Data Processing

This repository includes utilities to preprocess audio data and prepare it for model training. The provided scripts facilitate generating JAMS files and preparing model data for neural network training. Below are the details of the available commands:

### Scripts and Usage

### 1. **McGill Billboard Data Pull**

Within the `/data` directory, run `./download_data.sh` to pull down the publicly available audio dataset,
which includes chord annotations in the form of LAB files and associated chromagrams stored in CSVs.

### 2. **Generate JAMS Files**

In order to generate the JAMs files from the billboard data, run `python data_loader.py --process-billboard` 

### 3. **Create Usable Model Data**

In order to extract usable train/test data from the JAMs files, run `python .\data_loader.py --prepare-model-data amadeus-ex-machina/data/processed/<song_id>.jams -s`

The `-s` flag is used to designate that you would like to save the train/test data in a CSV.
