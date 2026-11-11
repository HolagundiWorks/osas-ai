# Deployment

## What can be shipped

- A PEFT adapter directory plus `manifest.json` (checksums and metadata).
- A local base-model directory produced by `scripts/download_model.py --execute`.
- The Python client in `myna_ai.inference`, which loads local files and does not need the training machine.

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

For a real run, point `--checkpoint` at the 7B adapter directory and `--model-key` at `olmo3-7b-instruct`. Merging the adapter into the base weights is left to PEFT's `merge_and_unload` on the GPU machine after you have decided the adapter is worth merging. This package does not merge inside the exporter, because a merge needs the base weights.

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
