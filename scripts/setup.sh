#!/bin/bash

set -euo pipefail

pip install transformers==4.41.0
pip install peft==0.10.0
pip install gradio==5.23.0
pip install diffusers==0.31.0
pip install torchvision

pip install huggingface-hub==0.28.1
pip install torchmetrics==0.6.0
pip install gradio_image_annotation
pip install carvekit
pip install opencv-python==4.8.0.74
pip install numpy==1.26.4
pip install sentencepiece
# huggingface-cli login

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${PROJECT_ROOT}/weights"
DEMO_EMBED_DIR="${PROJECT_ROOT}/demo_assets/embeddings"

mkdir -p "${WEIGHTS_DIR}" "${DEMO_EMBED_DIR}"

download_if_missing() {
    local url="$1"
    local destination="$2"
    if [ -f "${destination}" ]; then
        echo "✓ $(basename "${destination}") already exists, skipping download"
        return
    fi
    echo "↓ Downloading $(basename "${destination}")"
    wget -O "${destination}" "${url}"
}

download_if_missing "https://huggingface.co/yeates/OmniPaint/resolve/main/weights/omnipaint_remove.safetensors" "${WEIGHTS_DIR}/omnipaint_remove.safetensors"
download_if_missing "https://huggingface.co/yeates/OmniPaint/resolve/main/weights/omnipaint_insert.safetensors" "${WEIGHTS_DIR}/omnipaint_insert.safetensors"

download_if_missing "https://huggingface.co/yeates/OmniPaint/resolve/main/embeddings/remove.npz" "${DEMO_EMBED_DIR}/remove.npz"
download_if_missing "https://huggingface.co/yeates/OmniPaint/resolve/main/embeddings/insert.npz" "${DEMO_EMBED_DIR}/insert.npz"

echo "✓ Setup complete. You can now use the OmniPaint CLI tools."
