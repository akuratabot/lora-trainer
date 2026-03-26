# Qwen-Image-2512 LoRA Training Container
# Base: NGC PyTorch 25.11 (CUDA 13.0.2, PyTorch 2.10, ARM64 SBSA + Blackwell)
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Blackwell sm_100 kernel compilation target
ENV TORCH_CUDA_ARCH_LIST="10.0"
ENV DEBIAN_FRONTEND=noninteractive

# Install SimpleTuner with CUDA 13 support
RUN pip install --no-cache-dir 'simpletuner[cuda13]' \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Copy default config files (used by initContainer to seed PVC)
COPY config/ /app/config/

WORKDIR /workspace
