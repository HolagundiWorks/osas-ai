# Security

## Secrets

- Hub tokens, cloud keys and database passwords stay in environment variables or the host secret store.
- `.env` is gitignored. `.env.example` has empty names only.
- Logs pass through a redaction filter for token-shaped strings and for secret environment values.
- `scripts/download_model.py` prints a plan. It does not print `HF_TOKEN`.

## Data

- Do not commit model weights, GGUF files or customer drawings.
- CAD Environments is CC BY-NC 4.0. The preparer blocks commercial use.
- Clipboard text, selected text, key characters and click coordinates are not copied into training prompts.
- Do not upload confidential project files to a GPU host to debug a dataset converter.

## Model behaviour

- The system prompt tells the model not to invent quantities, structural results or approval transitions.
- Evaluation blocks promotion when a critical case fails.
- The inference client rejects unauthorised state changes.
- The mock executor ignores the idea of a model-supplied quantity: the read tool returns `ProjectState.quantities`.
- Raw completions are not applied to files.

## Training hosts

- Terminate the GPU instance after the adapter is copied out. See `docs/cloud-training.md`.
- Checkpoints should sit on a disk you control.
- The Docker image runs as a non-root user and does not bake in credentials.

## Supply chain

- Training dependencies are optional extras so unit tests do not need PyTorch.
- `bitsandbytes` is imported only when QLoRA starts, and only after the GPU check.
- `trust_remote_code` is false.
