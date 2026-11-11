# Myna AI

Training, evaluation, compression and offline inference for a Myna architecture and engineering assistant.

MynaCAD and CoreOps stay in their own applications. This repository trains and serves a model that can propose tool calls. It does not ship a launcher, and it does not write drawings or estimate databases.

The default model key is `allenai/Olmo-3-7B-Instruct` (`Olmo3ForCausalLM`, Apache 2.0). A `smoke-tiny` GPT-2 is included so the pipeline can be tested without downloading a 7B checkpoint. A passing smoke run is not evidence that the 7B model was trained.

## Layout

```text
configs/          model registry, training, data, tools, evaluation
src/myna_ai/      package
scripts/          inspect, prepare, train, evaluate, export, download
tests/            unit tests and a CPU vertical slice
docker/           CUDA image definition (not built in the initial environment)
docs/             architecture, data, training, evaluation, deployment, security
```

## Verified external facts

Checked while this repository was created:

- OLMo 3 7B Instruct config, tokenizer and `chat_template.jinja` on the Hub. The Jinja template does not emit the `<functions></functions>` marker shown in the model-card illustration.
- Transformers support starts at 4.57.0. This package requires `>=4.57.1`.
- LoRA target modules match the linear names used by public OLMo 3 QLoRA recipes.
- llama.cpp registers `Olmo3ForCausalLM` on the `olmo2` GGUF architecture from build b7120. No conversion was run here.
- `markov-ai/cad-environments` revision `78aea4f2cdde9d1ef2fc1780c12689e6ec06b1e9`: 51 workflows, CC BY-NC 4.0, not gated, index parquet 40867 bytes, card total about 152 GB. Event files are desktop UI traces, not Myna API calls.

Details are in `docs/architecture.md` and `docs/dataset-preparation.md`.

## Install

```bash
python -m pip install -e ".[dev]"          # unit tests
python -m pip install -e ".[dev,train]"    # CPU smoke test, adds PyTorch
python -m pip install -e ".[quant]"        # bitsandbytes, on the CUDA training host only
```

Development tests do not download the 7B weights.

## Tests

```bash
pytest -m "not train"
pytest -m train
```

## Dataset

```bash
python scripts/prepare_dataset.py --adapter synthetic --output data/processed/synthetic --commercial-use
python scripts/inspect_dataset.py --huggingface
```

CAD Environments cannot be prepared for commercial use. See `docs/dataset-preparation.md`.

## CPU smoke training

```bash
python scripts/prepare_dataset.py --adapter synthetic --output data/processed/synthetic --commercial-use
python scripts/train.py --config configs/training/smoke_tiny_lora.yaml
```

The trainer writes `outputs/smoke-tiny/train_result.json` with `olmo_7b_trained: false`.

On the machine used to build this repository (CPU only, torch 2.15.0, transformers 5.22.2) a 6-step smoke run moved the loss from 4.856 to 4.645. Pytest also resumed that run from step 6 through step 8, exported the adapter, reloaded it, and executed a mock `cad.create_geometry` call. One extra step with gradient checkpointing enabled also completed. None of that loaded or trained OLMo 3 7B. `Olmo3ForCausalLM` does import in transformers 5.22.2.

## OLMo 3 7B QLoRA

Run this only on a CUDA machine with at least 16 GB free, preferably 24 GB, after downloading the weights. The process exits if there is no suitable GPU.

```bash
export MYNA_OUTPUT_DIR=/persistent/myna/outputs
export MYNA_MODEL_DIR=/persistent/myna/models/olmo3-7b-instruct
export MYNA_TRAIN_PATH=/persistent/myna/train.jsonl
export MYNA_VALIDATION_PATH=/persistent/myna/validation.jsonl
python scripts/download_model.py --execute --destination "$MYNA_MODEL_DIR"
python scripts/train.py --config configs/training/olmo3_7b_qlora.yaml
```

Set `model_path` in the YAML to `$MYNA_MODEL_DIR` before an offline run. Resume with `--resume-from-checkpoint`. Stop the cloud instance afterwards:

```bash
bash scripts/cloud/remind_stop.sh
```

Full instructions: `docs/cloud-training.md`.

## Evaluation and export

```bash
python scripts/evaluate.py --reference --label reference --output outputs/eval-reference
python scripts/export.py --checkpoint outputs/smoke-tiny/checkpoint-12 --output outputs/export --model-key smoke-tiny --gguf-check
python scripts/export.py --checkpoint outputs/smoke-tiny/checkpoint-12 --output outputs/merged --model-key smoke-tiny --merge
```

`--merge` writes a full-precision smoke model. It does not download OLMo, and it refuses a QLoRA checkpoint. `--full-precision` copies safetensors and skips `optimizer.pt`. A GGUF file is written only with `--gguf-convert`, and that file is not marked as loaded by llama.cpp.

Recovery from a pruned checkpoint prints a plan unless `--execute-recovery` is set. The OLMo config still exits on CPU.

```bash
python scripts/train.py --config configs/training/smoke_tiny_lora.yaml --recovery --pruned-checkpoint outputs/smoke-tiny/checkpoint-12
```

A model is not promoted when a critical quantity, revision, geometry or engineering-safety case fails. The reference command scores scripted gold answers. It does not load OLMo.

## Inference boundary

Application code should validate tool calls and authorise anything that changes a project. Numerical quantities, code checks and approval transitions belong to deterministic Myna services. The tool catalog is an example schema (`verified: false`), not a confirmed MynaCAD or CoreOps API.

## Licence

Package code is Apache 2.0. OLMo 3 weights are Apache 2.0 under Ai2's responsible-use guidance. CAD Environments is CC BY-NC 4.0 and is not cleared here for commercial training.
