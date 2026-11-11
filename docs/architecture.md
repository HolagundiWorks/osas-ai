# Architecture

Myna AI is a Python package for dataset preparation, LoRA and QLoRA fine-tuning, evaluation, compression experiments, export and offline inference. It does not contain MynaCAD or CoreOps. Myna Launcher is not part of this repository.

## Model selection

`configs/models/registry.yaml` is the only model list. Code asks for a key such as `olmo3-7b-instruct`. The default key is OLMo 3 7B Instruct.

Verified from the Hugging Face files for `allenai/Olmo-3-7B-Instruct`:

| Item | Value |
| --- | --- |
| Class | `Olmo3ForCausalLM` |
| Licence | Apache 2.0 |
| Layers / hidden / intermediate | 32 / 4096 / 11008 |
| Heads | 32 query, 32 key-value |
| Context | 65536, with YaRN factor 8 from 8192 |
| Sliding window | 4096 on three of every four layers |
| Vocabulary | 100278 |
| EOS / pad | `<|endoftext|>` / `<|pad|>` |
| Transformers | 4.57.1 or newer |
| Chat | ChatML plus `<function_calls>` |

The published `config.json` sets `use_cache` to false. Generation code must enable the cache or it will be very slow. LoRA targets are the attention and MLP projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

`smoke-tiny` is a randomly initialised GPT-2 used to exercise the pipeline on CPU. It is not an OLMo checkpoint. `qwen2.5-7b-instruct` is a registered alternative whose licence flag stays unverified until that card is checked at download time.

## Data flow

```text
source index or synthetic builder
    -> licence check
    -> adapter
    -> exact-duplicate removal
    -> conversation validation
    -> workflow-level split
    -> JSONL
    -> masked supervised training
    -> adapter checkpoint
    -> evaluation gate
    -> manifest
    -> local inference client
    -> schema check
    -> authorised mock tool
```

CAD Environments workflows become semantic notes. Mouse and keyboard events stay non-executable. Example tool names live in `configs/inference/tools.yaml` and are marked unverified.

## Training

The native loop is the tested path for span-level loss masking. It renders the OLMo 3 chat template, masks every token outside assistant spans, and saves a full smoke state dict plus a PEFT adapter directory. QLoRA uses bitsandbytes NF4 only after the CUDA check passes. There is no CPU fallback for the 7B model.

TRL can be selected with `backend: trl`. A smoke run uses the local character tokenizer and does not download weights. TRL masks the prompt before the first assistant span. The native loop is still the path that masks every assistant span when later user turns are present.

## Inference boundary

`LocalInferenceClient` returns text, parsed tool calls and validation problems. It can run the in-memory mock executor. It has no API that writes a drawing, an estimate database or a project file. State-changing example tools require an `Authorisation` decision. Quantities are read from `ProjectState`, not from numbers the model invents.

## Compression

Structured MLP pruning builds a new module and leaves the original weights in place. `recover_pruned_mlp` then trains a copy of that module against the original outputs. A recovery plan for the language-model trainer does not launch a job unless `execute` is true, and OLMo 7B still refuses CPU. The OLMo 3 assessment records that every layer shares one `intermediate_size`, attention is not pruned, and a pruned shape is not a tested GGUF export. Dynamic int8 is a separate experiment from QLoRA and from GGUF.
