#!/usr/bin/env bash

# download URL for fasttext 300-d english word embeddings
EMBEDDINGS_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
OUTPUT_DIR="./fasttext_embeddings"
OUTPUT_FILE="$OUTPUT_DIR/cc.en.300.bin.gz"

# create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

if [ -f "$OUTPUT_FILE" ]; then
    echo "The embeddings file already exists at: $OUTPUT_FILE"
else
    # download
    echo "Downloading FastText word embeddings..."
    curl -o "$OUTPUT_FILE" "$EMBEDDINGS_URL"

    # verify if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download completed successfully. File saved to: $OUTPUT_FILE"
    else
        echo "Download not succesfull."
        exit 1
    fi
fi

# extract file
echo "Extracting..."
gunzip -k "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Extraction completed successfully."
else
    echo "Extraction failed."
    exit 1
fi

echo "Complete!"