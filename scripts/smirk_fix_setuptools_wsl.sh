#!/usr/bin/env bash
set -e
UV="$HOME/.local/bin/uv"
SM="$HOME/envs/smirk/bin/python"
# pytorch_lightning 2.2.1 uses pkg_resources.declare_namespace, removed in setuptools>=80
"$UV" pip install --python "$SM" "setuptools<70" 2>&1 | tail -2
echo "=== verify all SMIRK deps ==="
"$SM" -c "import numpy,mediapipe,chumpy,pytorch_lightning,timm,omegaconf,sklearn,skimage,cv2,albumentations,pkg_resources; print('numpy', numpy.__version__, '| all SMIRK deps OK')"
