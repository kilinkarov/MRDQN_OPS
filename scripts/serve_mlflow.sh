#!/bin/bash
# Serve model using MLflow
# Usage: bash scripts/serve_mlflow.sh <model_path> <port>

set -e

MODEL_PATH=${1:-"trained_models/BTC_777/model_final.pth"}
PORT=${2:-5000}

echo "======================================"
echo "MLflow Model Serving"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Starting MLflow model server..."
echo "Server will be available at: http://127.0.0.1:$PORT"
echo ""
echo "Test with:"
echo "  curl -X POST http://127.0.0.1:$PORT/invocations -H 'Content-Type: application/json' -d '{\"inputs\": [[0.5, 0.8, ...32 values...]]}'"
echo ""

mlflow models serve \
  -m "file://$MODEL_PATH" \
  -p $PORT \
  --no-conda \
  --env-manager local

echo ""
echo "Server stopped."
