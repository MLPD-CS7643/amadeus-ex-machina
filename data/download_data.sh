# Index file for song metadata / ID lookup
curl -L -o raw/billboard_index.csv https://www.dropbox.com/sh/o0olz0uwl9z9stb/billboard-2.0-index.csv?dl=1

## Chroma CSVs
curl -L -o raw/archive.zip https://www.kaggle.com/api/v1/datasets/download/jacobvs/mcgill-billboard

# Full LAB files
curl -L -o raw/billboard_annotations.tar.gz https://www.dropbox.com/s/ep41gwy28vo3wxy/billboard-2.0.1-lab.tar.gz?dl=1

unzip raw/archive.zip -d raw
rm raw/archive.zip

tar -xzf raw/billboard_annotations.tar.gz -C raw
rm raw/billboard_annotations.tar.gz

rm -r raw/annotations

mv raw/metadata/metadata/* raw/metadata
rm -r raw/metadata/metadata

# Map associated metadata/annotations together
for annotation_dir in raw/McGill-Billboard/*; do
    if [ -d "$annotation_dir" ]; then
        numeric_label=$(basename "$annotation_dir")
        # Check if the corresponding metadata subdirectory exists
        metadata_dir="raw/metadata/$numeric_label"
        if [ -d "$metadata_dir" ]; then
            # Create a subdirectory in final for this numeric label
            final_subdir="raw/mapped_data/$numeric_label"
            mkdir -p "$final_subdir"

            # Copy the corresponding annotation and metadata directories
            cp -r "$annotation_dir" "$final_subdir/annotations"
            cp -r "$metadata_dir" "$final_subdir/metadata"
        fi
    fi
done

rm -r raw/McGill-Billboard
rm -r raw/metadata