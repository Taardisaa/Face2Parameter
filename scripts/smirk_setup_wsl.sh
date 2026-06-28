#!/usr/bin/env bash
# Build the isolated SMIRK env inside WSL2 (Linux), where pytorch3d has prebuilt wheels.
# Run:  wsl -d Ubuntu -- bash /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/smirk_setup_wsl.sh
set -e
UV="$HOME/.local/bin/uv"
SM="$HOME/envs/smirk/bin/python"

cd "$HOME"
[ -x "$SM" ] || "$UV" venv --python 3.10 "$HOME/envs/smirk"

echo "=== torch 2.0.1 cu118 ==="
"$UV" pip install --python "$SM" torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -2
echo "=== fvcore/iopath ==="
"$UV" pip install --python "$SM" fvcore iopath 2>&1 | tail -1
echo "=== pytorch3d (linux prebuilt wheel) ==="
"$UV" pip install --python "$SM" --no-deps pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html 2>&1 | tail -3
echo "=== verify ==="
"$SM" -c "import torch, pytorch3d; from pytorch3d import _C; print('pytorch3d', pytorch3d.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
