#!/usr/bin/env bash
# Install the remaining SMIRK deps into the WSL py3.10 env (torch/torchvision/pytorch3d already in).
set -e
UV="$HOME/.local/bin/uv"
SM="$HOME/envs/smirk/bin/python"
REQ=/mnt/c/Users/13666/Workspace/smirk/requirements.txt

# strip CR, drop lines we already installed (torch/torchvision) and chumpy (handled separately)
tr -d '\r' < "$REQ" | grep -viE '^(torch==|torchvision==|chumpy)' > /tmp/smirk_req.txt
echo "=== installing SMIRK deps ==="
"$UV" pip install --python "$SM" -r /tmp/smirk_req.txt 2>&1 | tail -5
echo "=== chumpy (setup.py imports pip -> may need --no-build-isolation) ==="
"$UV" pip install --python "$SM" chumpy==0.70 2>&1 | tail -3 \
  || "$UV" pip install --python "$SM" --no-build-isolation chumpy==0.70 2>&1 | tail -3
echo "=== verify imports ==="
"$SM" -c "import numpy,mediapipe,chumpy,pytorch_lightning,timm,omegaconf,sklearn,skimage,cv2,albumentations; print('numpy',numpy.__version__,'| all SMIRK deps import OK')"
