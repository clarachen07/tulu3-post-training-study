#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "README.md" || ! -d "scripts" ]]; then
  echo "Error: please run this script from the project root."
  exit 1
fi

echo "Python executable:"
which python
python --version

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing PyTorch 2.6.0 with CUDA 12.4..."
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing LLM inference dependencies..."
python -m pip install transformers accelerate safetensors sentencepiece protobuf huggingface_hub

echo
echo "GPU environment setup finished."
echo "Next: python scripts/check_gpu_env.py"
