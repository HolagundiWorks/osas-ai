# Evaluation

The held-out cases in `myna_ai.evaluation.cases` cover selection, geometry creation, quantity reads, approved-estimate revision, unsupported requests, design assumptions, invalid geometry and approval bypass. Their user prompts are not in the synthetic training corpus. The leakage check is `tests/unit/test_dataset_pipeline.py`.

A case is critical when a wrong answer would invent a quantity, skip approval, accept invalid geometry, or state an engineering conclusion the model is not allowed to make. Any critical failure blocks promotion, even if the averages look acceptable.

Thresholds are in `configs/evaluation/acceptance.yaml`.

| Measurement | How it is produced |
| --- | --- |
| Tool selection and arguments | Parsed `<function_calls>` or JSON, compared with the case |
| Schema validity | Proposed JSON Schema in `configs/inference/tools.yaml` |
| Revision compliance | The approved-revision and approval-bypass cases |
| Hallucination | A number in the completion that is not in the user text |
| Latency | Wall time around the generator |
| Tokens per second | Not invented. Character rate is labelled as characters |
| RAM | Process peak RSS when the caller supplies it |
| GPU memory | Only when CUDA reports a peak |
| Artifact size | Sum of files in an export directory |

Compare labels by passing several reports to `compare_reports`. The A/B/C/D matrix uses the same roles: A original, B fine-tuned, C quantised, D pruned. A slot with no report is stored as `measured: false` and has no metrics. `olmo_7b_baseline_measured` is true only when slot A was supplied and that report sets `olmo_7b_evaluated`. The scripted reference does not set that flag. The original 7B checkpoint has not been downloaded, so there is no measured baseline for it.

```bash
python scripts/evaluate.py \
  --reference --label reference --output outputs/eval-reference --slot B \
  --matrix-output outputs/matrix
```

```bash
python scripts/evaluate.py --reference --label reference --output outputs/eval-reference
python scripts/benchmark.py --artifact outputs/export --output outputs/benchmark.json
```

`--reference` scores the gold completions. It does not load OLMo.

Promotion of a real adapter is a separate decision after that adapter has been scored. The exporter writes `promotion_allowed: false` until that happens.
