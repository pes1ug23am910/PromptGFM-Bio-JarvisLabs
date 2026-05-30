#!/bin/bash

# Data download script for PromptGFM-Bio
# This script will download all required biomedical datasets

echo "Starting data download for PromptGFM-Bio..."

# Create data directories if they don't exist
mkdir -p data/raw/biogrid
mkdir -p data/raw/string
mkdir -p data/raw/disgenet
mkdir -p data/raw/hpo

# Run Python download script
python src/data/download.py

echo "Data download complete!"
