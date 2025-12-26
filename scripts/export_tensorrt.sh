#!/bin/bash
# Export ONNX model to TensorRT format
# Usage: bash scripts/export_tensorrt.sh <onnx_model_path> <output_trt_path>

set -e

ONNX_MODEL=${1:-"trained_models/model.onnx"}
TRT_MODEL=${2:-"trained_models/model.trt"}

echo "Converting ONNX model to TensorRT..."
echo "Input: $ONNX_MODEL"
echo "Output: $TRT_MODEL"

# Check if trtexec is available
if ! command -v trtexec &> /dev/null
then
    echo "Error: trtexec not found. Please install TensorRT."
    echo "Visit: https://developer.nvidia.com/tensorrt"
    exit 1
fi

# Convert to TensorRT with FP16 precision
trtexec \
  --onnx="$ONNX_MODEL" \
  --saveEngine="$TRT_MODEL" \
  --explicitBatch \
  --fp16 \
  --workspace=4096 \
  --verbose

echo "Conversion completed successfully!"
echo "TensorRT model saved to: $TRT_MODEL"
