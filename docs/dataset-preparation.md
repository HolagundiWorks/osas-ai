# Dataset preparation

## CAD Environments

Dataset: `markov-ai/cad-environments`.

Checked against the Hub API and `data/workflows.parquet` at revision `78aea4f2cdde9d1ef2fc1780c12689e6ec06b1e9`:

- Public, not gated.
- Licence: CC BY-NC 4.0.
- One viewer split, `train`, backed by the parquet index.
- 51 rows. The card reports 99.03 hours and about 152 GB because the repository also holds screen recordings and CAD files.
- The index parquet itself is 40867 bytes.
- Columns: `workflow_id`, `software`, `software_slug`, `domain`, `task`, `description`, `deliverables`, `duration_hours`, `platform`, `event_count`, `fps`, `total_frames`, and the URL columns listed in `configs/datasets/cad_environments.yaml`.
- Software slugs in the index: `autocad` (4), `sketchup` (7), `revit-architecture` (6), `revit-mep` (1), `revit-structure` (3), `staad-pro` (5), `solidworks` (19), `nx-cad` (6).

Default ingestion keeps the first six slugs. SOLIDWORKS and NX stay available if you pass them explicitly.

One small STAAD.Pro workflow was opened to confirm file shapes, not to commit its content:

- `task_desc.json` has `schema_version`, `dimensions` and `deliverables`.
- `narration.json` has `annotations` (`start_s`, `end_s`, `action`, `intent`, `app`, `details`) and `summary`.
- `events.json` is JSONL. Types include `mouse_move`, `click`, `key`, `clipboard` and window events. Those are desktop traces.

The adapter does not download video, `input_files` or `output_files`. It counts event types and drops coordinates, key characters, selected text and clipboard payloads. Narration `details` are withheld from the prompt because they describe clicks. `myna_api` on every derived operation is null.

CC BY-NC 4.0 is not a licence to train a commercial Myna product. `prepare_dataset` refuses `commercial_use=true` for this source. Research use must set `commercial_use=false` and `acknowledge_noncommercial_licence=true`.

## Synthetic corpus

`myna_ai.datasets.synthetic` holds original conversations for tests and for the CPU smoke run. Held-out prompts live in `myna_ai.evaluation.cases` and are checked for leakage after a split.

## Parametric instructions

`parametric_instructions` reads a local JSONL file with `instruction` and `response`. No public parametric corpus is wired in. Put a licence on each row when you add one.

## Commands

```bash
python scripts/inspect_dataset.py --source tests/fixtures/cad
python scripts/inspect_dataset.py --huggingface
python scripts/prepare_dataset.py --adapter synthetic --output data/processed/synthetic --commercial-use
python scripts/prepare_dataset.py \
  --adapter cad_environments \
  --source /path/to/workflows.parquet \
  --output data/processed/cad-research \
  --acknowledge-noncommercial-licence
```

The Hugging Face command reads the dataset API. It does not download the 152 GB tree. Downloading only the index is a separate, explicit step with `huggingface_hub` or a browser. Do not point the training job at customer drawings.
