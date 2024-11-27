# Amadeus Ex Machina


---

## Data Processing

This repository includes utilities to preprocess audio data and prepare it for model training. The provided scripts facilitate processing usable model data from the Billboard McGills Dataset. 

### Scripts and Usage

### 1. **McGill Billboard Data Pull**

Within the `/data` directory, run `./download_data.sh` to pull down the publicly available audio dataset,
which includes chord annotations in the form of LAB files and associated chromagrams stored in CSVs.

### 2. **Create Usable Model Data**

In order to extract usable train/test data from the raw data, instantiate a `BillboardDataProcessor` and run `process_billboard_data`

This will create a combined CSV of all song chromagrams and associated annotations.
