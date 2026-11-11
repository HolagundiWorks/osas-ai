# Cloud training

This repository does not create RunPod or Lambda instances. The commands below are what an operator runs after a GPU machine exists.

## What was actually run

The development environment had no CUDA device. Pytest passed 24 tests. The CPU smoke model trained for 6 steps (loss 4.856 to 4.645), resumed through step 8, and reloaded for generation. Gradient checkpointing was enabled for a separate 1-step run on the same tiny model. QLoRA, the 7B download, GGUF conversion and any cloud instance were not run. `assert_training_device` rejects the 7B QLoRA config when CUDA is absent.

## Hardware

| Run | Device |
| --- | --- |
| `smoke-tiny` LoRA | CPU is allowed only when `allow_cpu_smoke` is true |
| OLMo 3 7B QLoRA | NVIDIA GPU, at least 16 GB free. 24 GB is the practical default at sequence length 2048, batch size 1, gradient accumulation 16 |
| OLMo 3 7B LoRA in bf16 | At least 28 GB free |
| Full fine-tune | Not configured. Do not attempt it on one 24 GB card |

If CUDA is missing, or free memory is below the registry threshold, training raises `GpuRequiredError` before weights load. It does not continue on CPU.

## One-time setup on the GPU host

```bash
export MYNA_OUTPUT_DIR=/persistent/myna/outputs
export MYNA_MODEL_DIR=/persistent/myna/models/olmo3-7b-instruct
export MYNA_TRAIN_PATH=/persistent/myna/train.jsonl
export MYNA_VALIDATION_PATH=/persistent/myna/validation.jsonl
# optional, for higher Hub rate limits; never commit this
export HF_TOKEN=...

bash scripts/cloud/setup.sh
python scripts/download_model.py --model-key olmo3-7b-instruct --destination "$MYNA_MODEL_DIR"
python scripts/download_model.py --execute --model-key olmo3-7b-instruct --destination "$MYNA_MODEL_DIR"
```

QLoRA also needs bitsandbytes:

```bash
python -m pip install -e ".[quant]"
```

Edit the training YAML so `model_path` points at `$MYNA_MODEL_DIR`, or pass a local snapshot. `local_files_only` stays true so a missing download fails closed.

## Launch

```bash
python scripts/train.py --config configs/training/olmo3_7b_qlora.yaml
```

Resume, using the absolute step count:

```bash
python scripts/train.py \
  --config configs/training/olmo3_7b_qlora.yaml \
  --resume-from-checkpoint "$MYNA_OUTPUT_DIR/checkpoint-50"
```

Checkpoints land in `MYNA_OUTPUT_DIR`. Copy the adapter directory off the GPU disk before you terminate the machine. The original base weights stay in the model directory for rollback.

TRL, if you want that backend instead of the native loop:

```bash
# set backend: trl in a copy of the YAML, then
python scripts/train.py --config configs/training/olmo3_7b_qlora.yaml
```

The native loop is the one with an explicit assistant-token mask. Use it unless you have a reason to compare TRL.

## Stop the bill

Closing SSH does not stop the charge.

```bash
bash scripts/cloud/remind_stop.sh
```

- RunPod: stop or terminate the pod on the Pods page.
- Lambda Cloud: terminate the instance on the Instances page.

`scripts/cloud/remind_stop.sh` only prints those steps. It does not call either API.

## GitHub Actions

`.github/workflows/ci.yml` runs unit tests and, in a second job, the CPU smoke tests. It does not train the 7B model. `validate-training-config.yml` is manual and only checks that the QLoRA config is rejected without a GPU.

## Docker

`docker/Dockerfile` starts from `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. That image was not pulled here, so confirm the tag before a production build. The image does not contain Hub tokens or model weights. Install the quant extra in the image only after the CUDA base is confirmed.
