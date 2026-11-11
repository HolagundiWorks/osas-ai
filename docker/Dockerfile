# CUDA training image. This file was not built in the development environment
# that created the repository, so the base tag still needs to be confirmed on
# the host before a production build.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/myna-ai

RUN useradd --create-home --uid 1000 myna
COPY --chown=myna:myna . /opt/myna-ai

RUN pip install --no-cache-dir -e ".[train]" \
    && chown -R myna:myna /opt/myna-ai

USER myna

# QLoRA additionally needs bitsandbytes, which is not installed here because
# the quant extra is CUDA-specific. Install it in the image build once the
# base tag is confirmed: pip install -e ".[quant]"
CMD ["python", "scripts/train.py", "--config", "configs/training/olmo3_7b_qlora.yaml"]
