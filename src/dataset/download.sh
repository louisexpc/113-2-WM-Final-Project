#!/bin/bash

# Check for gdown installation
if ! command -v gdown &>/dev/null; then
    echo "gdown not found. Installing..."
    pip install --upgrade gdown || {
        echo "Installation failed. Aborting."
        exit 1
    }
fi

# Read each line from files.txt
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines or malformed ones
    [[ -z "$line" || "$line" != *:* ]] && continue

    # Split path and file ID
    IFS=':' read -r file_path file_id <<<"$line"

    # Extract directory from file path
    dir_path=$(dirname "$file_path")

    # Create target directory if it doesn't exist
    mkdir -p "$dir_path"

    echo "Downloading $file_id -> $file_path"
    gdown --fuzzy "https://drive.google.com/uc?id=$file_id" -O "$file_path"
done <files.txt

echo "Download completed."
