# Deployment

## What can be shipped

- A PEFT adapter directory plus `manifest.json` (checksums and metadata).
- A local base-model directory produced by `scripts/download_model.py --execute`.
- The Python client in `myna_ai.inference`, which loads local files and does not need the training machine.

The CUDA training image can be built with `docker build -t myna-ai:local -f docker/Dockerfile .`. `.dockerignore` keeps `models/` out of that image. The image tag built here is `myna-ai:local`. It is not a GGUF runtime.

GGUF is a documented target, not an artifact this repository has produced. llama.cpp has registered `Olmo3ForCausalLM` since build b7120 and stores it under the `olmo2` architecture label, including the sliding-window pattern. Public quantisations of `allenai/Olmo-3-7B-Instruct` exist. `assess_gguf` reports `conversion_executed: false` until `convert_hf_to_gguf.py` is actually run. Training-time NF4 QLoRA is not that file.

Until a GGUF conversion is tested on the exact checkpoint you trained, the offline runtime is Hugging Face Transformers from a local directory:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/models/olmo3-7b-instruct", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("/models/olmo3-7b-instruct", local_files_only=True)
```

Set `use_cache=True` on the config before `generate`. The published config leaves it false.

## Export

```bash
python scripts/export.py \
  --checkpoint outputs/smoke-tiny/checkpoint-8 \
  --output outputs/export \
  --model-key smoke-tiny \
  --gguf-check
```

For a real run, point `--checkpoint` at the 7B adapter directory and `--model-key` at `olmo3-7b-instruct`.

`--full-precision` copies `model.safetensors` and tokenizer sidecars. It leaves `optimizer.pt` behind. A LoRA checkpoint still contains adapter tensors in that state dict.

`--merge` runs PEFT `merge_and_unload` and writes a full-precision directory plus a checksum manifest. The smoke checkpoint can be merged on CPU because its base weights are already in `model.safetensors`. An OLMo merge requires `model_path` in `training_config.json` to be a local directory that already contains `config.json`. The exporter does not download weights. QLoRA checkpoints are refused. `promotion_allowed` stays false, and `olmo_7b_trained` stays false.

```bash
python scripts/export.py --checkpoint outputs/smoke-tiny/checkpoint-12 --output outputs/merged --model-key smoke-tiny --merge
python scripts/export.py --checkpoint outputs/smoke-tiny/checkpoint-12 --output outputs/weights --model-key smoke-tiny --full-precision
```

`--gguf-convert` runs a supplied `convert_hf_to_gguf.py` and writes a sibling `.gguf` file. The report sets `llama_cpp_load_verified` to false. A converter that exits 0 is not a claim that llama.cpp loaded OLMo 3. Qwen conversion is refused while `licence_verified` is false. The smoke GPT-2 is not a GGUF target.

## Application integration

MynaCAD and CoreOps should talk to `LocalInferenceClient` or an equivalent boundary:

1. The application builds the user message and the project context.
2. The client returns tool calls.
3. The application validates them against its real API, not only against `configs/inference/tools.yaml`.
4. A person or an application policy authorises state-changing calls.
5. Deterministic services compute quantities, costs, code checks and revision numbers.
6. The application writes the drawing or the estimate.

The tool names `cad.*`, `coreops.*`, `structural.*` and `bim.*` are proposed contracts. `verified` is false on every entry. Do not bind them to a live database until the product APIs are checked and the catalog is updated.

The mock executor is for tests. `persisted` is false on its results.

## Offline operation

After the base weights and adapter are on disk, inference uses `local_files_only`. Do not set a Hub token in the deployed application unless you still need to download. The smoke runtime reads `model.safetensors` from the checkpoint and does not open a network connection.
