# Qwen-Image-2512 LoRA Training Container
# Base: NGC PyTorch 25.11 (CUDA 13.0.2, PyTorch 2.10, ARM64 SBSA + Blackwell)
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Blackwell sm_100 kernel compilation target
ENV TORCH_CUDA_ARCH_LIST="10.0"
ENV DEBIAN_FRONTEND=noninteractive

# Install SimpleTuner with CUDA 13 support
RUN pip install --no-cache-dir 'simpletuner[cuda13]' \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Patch SimpleTuner safety_check to handle nvidia-smi returning [N/A] for
# unified memory GPUs (e.g. NVIDIA GB10 Grace Blackwell). The check is only
# used to warn about the SOAP optimizer; we default to 128 (GB) so no warning
# fires and training proceeds normally.
RUN sed -i \
    's/total_memory = int(output.decode().strip()) \/ 1024/raw = output.decode().strip(); total_memory = (int(raw) if raw.lstrip("-").isdigit() else 131072) \/ 1024/' \
    /usr/local/lib/python3.12/dist-packages/simpletuner/helpers/training/default_settings/safety_check.py

# Copy default config files (used by initContainer to seed PVC)
COPY config/ /app/config/

# Create non-root user for restricted namespace compatibility.
# All writes go to the PVC mounted at /data, so /workspace is just a fallback.
RUN chown -R 1000:1000 /app

USER 1000

WORKDIR /workspace
