#!/bin/sh
set -e

# Clean the cache directory before starting
echo "Clearing DFP cache..."
rm -rf /tmp/dfp_cache
echo "Cache cleared."

# Run the training pipeline
echo "Starting Morpheus DFP Training Pipeline..."
python /app/dfp_training.py
