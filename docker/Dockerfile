# CUDA training image.
# The base tag was pulled and this image was built locally as myna-ai:local.
# pytorch 2.5.1 ships torch 2.5.1, and transformers 5 refuses that runtime.
# The upgrade below stays on the CUDA 12.4 wheel instead of a default PyPI wheel.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/myna-ai

RUN useradd --create-home --uid 1000 myna
COPY --chown=myna:myna . /opt/myna-ai

RUN pip install --no-cache-dir --upgrade "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -e ".[train]" \
    && chown -R myna:myna /opt/myna-ai

USER myna

# QLoRA additionally needs bitsandbytes, which is not installed here because
# the quant extra is CUDA-specific. Install it on the GPU host:
# pip install -e ".[quant]"
CMD ["python", "scripts/train.py", "--config", "configs/training/olmo3_7b_qlora.yaml"]
