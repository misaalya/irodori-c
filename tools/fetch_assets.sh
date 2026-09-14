#!/usr/bin/env bash
# Populate weights/ without Python: tokenizer + codec exports from the GitHub
# release assets tarball, and the FP32 checkpoint from Hugging Face.
#   tools/fetch_assets.sh                                   # base model
#   tools/fetch_assets.sh phasefield-audio/Irodori-TTS-v4.1-Anime
#   RELEASE=v0.2.0 tools/fetch_assets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE=${RELEASE:-v0.2.0}
MODEL_REPO=${1:-Aratako/Irodori-TTS-v4.1-Small}
BASE="https://github.com/misaalya/irodori-c/releases/download/$RELEASE"
mkdir -p downloads weights
for f in "irodori-c-assets-$RELEASE.tar.gz" SHA256SUMS; do
  [ -f "downloads/$f" ] || curl -L --fail --progress-bar -o "downloads/$f" "$BASE/$f"
done
(cd downloads && sha256sum -c --ignore-missing SHA256SUMS)
tar -C weights --strip-components=2 -xzf "downloads/irodori-c-assets-$RELEASE.tar.gz" --wildcards '*/weights/*'
if [ ! -f weights/model.safetensors ]; then
  echo "Downloading $MODEL_REPO model.safetensors (~3 GB)"
  curl -L --fail --progress-bar -C - -o weights/model.safetensors \
    "https://huggingface.co/$MODEL_REPO/resolve/main/model.safetensors"
fi
ls -la weights
