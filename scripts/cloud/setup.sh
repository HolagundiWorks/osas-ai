#!/usr/bin/env bash
# Prepare a GPU host. This script does not create a cloud account or start billing.
set -euo pipefail

if [[ -z "${MYNA_OUTPUT_DIR:-}" ]]; then
  echo "Set MYNA_OUTPUT_DIR to a persistent disk path before training." >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -e ".[train]"
python - <<'PY'
import torch
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"free_gb {free/1024**3:.2f} total_gb {total/1024**3:.2f}")
else:
    raise SystemExit("No CUDA device is visible. Do not start 7B training.")
PY

mkdir -p "${MYNA_OUTPUT_DIR}"
echo "Install bitsandbytes separately if QLoRA import fails: pip install -e '.[quant]'"
echo "Download weights with: python scripts/download_model.py --execute --destination \"\${MYNA_MODEL_DIR}\""
echo "Then stop the instance if you are not ready to train. See docs/cloud-training.md."
