#!/usr/bin/env bash
set -e
UV="$HOME/.local/bin/uv"
SM="$HOME/envs/smirk/bin/python"
"$UV" pip install --python "$SM" pip setuptools wheel 2>&1 | tail -1
echo "=== chumpy via --no-build-isolation (pip now in env) ==="
"$UV" pip install --python "$SM" --no-build-isolation chumpy==0.70 2>&1 | tail -5
echo "=== verify all SMIRK deps ==="
"$SM" -c "import numpy,mediapipe,chumpy,pytorch_lightning,timm,omegaconf,sklearn,skimage,cv2,albumentations; print('numpy', numpy.__version__, '| all SMIRK deps OK')"
