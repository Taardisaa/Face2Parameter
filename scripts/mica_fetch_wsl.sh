#!/usr/bin/env bash
# Official MICA README checkpoint; retain locally, never add weights to Git.
# Run: wsl -d Ubuntu -- bash scripts/mica_fetch_wsl.sh /mnt/c/.../outputs/mica_assets_20261007
set -euo pipefail
dest="${1:?Pass an ignored asset directory}"
mkdir -p "$dest"
fetch() {
  local file="$1" id="$2"
  if [ -e "$dest/$file" ]; then
    echo "Existing asset retained: $dest/$file"
    sha256sum "$dest/$file"
    return
  fi
  if [ -e "$dest/$file.part" ]; then
    echo "Partial prior download retained; choose a new directory or inspect it first." >&2
    exit 1
  fi
  "$HOME/envs/smirk/bin/python" -m gdown "https://drive.google.com/uc?id=$id" -O "$dest/$file.part"
  mv -n "$dest/$file.part" "$dest/$file"
  sha256sum "$dest/$file"
}
fetch mica.tar 1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt
fetch antelopev2.zip 16PWKI_RjjbE4_kqpElG-YFqe8FpXjads
