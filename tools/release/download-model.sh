#!/usr/bin/env bash
# Download the FP32 checkpoint (about 3 GB) into weights/. Only curl is needed.
#   ./download-model.sh                      # Aratako/Irodori-TTS-v4.1-Small
#   ./download-model.sh phasefield-audio/Irodori-TTS-v4.1-Anime
set -euo pipefail
cd "$(dirname "$0")"
REPO=${1:-Aratako/Irodori-TTS-v4.1-Small}
mkdir -p weights
URL="https://huggingface.co/$REPO/resolve/main/model.safetensors"
echo "Downloading $URL -> weights/model.safetensors"
curl -L --fail --progress-bar -C - -o weights/model.safetensors "$URL"
echo "done. Run: ./run-demo.sh   or   bin/irodori-onemkl --text 'こんにちは。' --model weights/model.safetensors --tokenizer weights/tokenizer.bin --decoder weights/dacvae_decoder.safetensors --dit-precision int8 --codec-precision int8 --out out.wav"
