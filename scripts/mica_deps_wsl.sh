#!/usr/bin/env bash
# Use MICA's dependency versions without replacing existing SMIRK Torch/NumPy.
# Pin ONNX/protobuf together to retain MediaPipe's existing protobuf API.
set -euo pipefail
"$HOME/.local/bin/uv" pip install --python "$HOME/envs/smirk/bin/python" \
  insightface==0.7 onnxruntime==1.13.1 onnx==1.13.0 protobuf==3.20.3 \
  loguru==0.6.0 face-alignment==1.3.5 albumentations==1.3.0 \
  numba==0.56.4 llvmlite==0.39.1 numpy==1.22.4
