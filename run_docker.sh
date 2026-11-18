#!/bin/bash

set -e

echo "Building Docker image..."
docker build -t mh-handbook-rag .

echo "Starting Streamlit application..."
docker run \
  --rm \
  -p 8501:8501 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  mh-handbook-rag
