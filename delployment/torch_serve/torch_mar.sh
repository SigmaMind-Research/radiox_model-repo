#!/bin/bash

# Define the model name, version, and serialized model file path
MODEL_NAME="radiox"
MODEL_VERSION="5.0"
SERIALIZED_MODEL="path/to/serialized/model.pt"

# Check if the MAR file already exists and remove it
if [ -f "${MODEL_NAME}.mar" ]; then
  rm "${MODEL_NAME}.mar"
  echo "Removed existing model archive."
fi

# Create the MAR file
torch-model-archiver \
  --model-name "${MODEL_NAME}" \
  --version "${MODEL_VERSION}" \
  --model-file "radiox_model.py"
  --serialized-file "${SERIALIZED_MODEL}" \
  --handler "custum_handler.py"  # Replace with the actual handler script \
  --extra-files "vocab.model" 

# Check if the MAR file was created successfully
if [ -f "${MODEL_NAME}.mar" ]; then
  echo "Model archive (${MODEL_NAME}.mar) created successfully."
else
  echo "Failed to create the model archive (${MODEL_NAME}.mar)."
fi
