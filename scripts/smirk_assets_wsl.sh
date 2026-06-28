#!/usr/bin/env bash
set -e
SM="$HOME/envs/smirk/bin/python"
SMIRK=/mnt/c/Users/13666/Workspace/smirk
mkdir -p "$SMIRK/assets" "$SMIRK/pretrained_models"

echo "=== mediapipe face_landmarker.task ==="
URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
if command -v wget >/dev/null; then wget -q "$URL" -O "$SMIRK/assets/face_landmarker.task";
else curl -sL "$URL" -o "$SMIRK/assets/face_landmarker.task"; fi
ls -la "$SMIRK/assets/face_landmarker.task"

echo "=== SMIRK pretrained model (Google Drive via gdown) ==="
"$SM" -m gdown "https://drive.google.com/uc?id=1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE" -O "$SMIRK/pretrained_models/SMIRK_em1.pt"
ls -la "$SMIRK/pretrained_models/"
