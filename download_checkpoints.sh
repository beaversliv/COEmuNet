#!/bin/bash

# Set the Google Drive file ID and output directory
FILE_ID="1v4NUBV9Uc3WPUP3usCMnJ98Piwe1F33D"
OUTPUT_DIR="./checkpoint"
FILE_NAME="pretrained.pth"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Install gdown if not available
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi

# Download the file using gdown
echo "Downloading file $FILE_NAME from Google Drive..."
gdown --id $FILE_ID -O "$OUTPUT_DIR/$FILE_NAME"

echo "Download complete. File saved at $OUTPUT_DIR/$FILE_NAME."
