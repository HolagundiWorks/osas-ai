"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from myna_ai.common.logging import configure_logging


def inspect_dataset() -> None:
    parser = argparse.ArgumentParser(description="Inspect a CAD dataset index or the Hugging Face dataset card.")
    parser.add_argument("--source", type=Path, help="Local index or directory.")
    parser.add_argument("--huggingface", action="store_true", help="Read Hub metadata only. Does not download media.")
    parser.add_argument("--dataset-id", default="markov-ai/cad-environments")
    args = parser.parse_args()
    configure_logging()
    if args.huggingface:
        from myna_ai.datasets.inspect import inspect_huggingface

        report = inspect_huggingface(args.dataset_id)
    elif args.source:
        from myna_ai.datasets.inspect import inspect_local

        report = inspect_local(args.source)
    else:
        parser.error("Pass --source or --huggingface.")
    print(json.dumps(report, indent=2))


def prepare_dataset() -> None:
    parser = argparse.ArgumentParser(description="Convert a dataset into validated Myna JSONL splits.")
    parser.add_argument("--adapter", required=True, choices=["synthetic", "cad_environments", "parametric_instructions"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--commercial-use", action="store_true")
    parser.add_argument("--acknowledge-noncommercial-licence", action="store_true")
    parser.add_argument("--software", nargs="*", default=None)
    args = parser.parse_args()
    configure_logging()
    from myna_ai.datasets.prepare import prepare_dataset as prepare

    options = {}
    if args.software:
        options["software_slugs"] = args.software
    report = prepare(
        adapter_name=args.adapter,
        source=args.source,
        output_dir=args.output,
        commercial_use=args.commercial_use,
        acknowledge_noncommercial_licence=args.acknowledge_noncommercial_licence,
        seed=args.seed,
        adapter_options=options,
    )
    print(json.dumps(report, indent=2))


def train() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a registered model. Refuses CPU for OLMo 7B.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--validation-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--recovery", action="store_true", help="Plan or launch a recovery fine-tune from a pruned checkpoint.")
    parser.add_argument("--pruned-checkpoint", type=Path)
    parser.add_argument("--execute-recovery", action="store_true", help="Start the recovery job. Without this flag only a plan is written.")
    args = parser.parse_args()
    configure_logging()
    if args.recovery:
        if args.pruned_checkpoint is None:
            parser.error("--pruned-checkpoint is required with --recovery.")
        from myna_ai.compression.recovery import launch_recovery_finetune

        plan = launch_recovery_finetune(
            pruned_checkpoint=args.pruned_checkpoint,
            train_config=args.config,
            output_dir=args.output_dir,
            execute=args.execute_recovery,
            max_steps=args.max_steps,
        )
        print(json.dumps(plan, indent=2, default=str))
        return
    from myna_ai.training.config import load_training_config
    from myna_ai.training.loop import run_training

    config = load_training_config(
        args.config,
        {
            "train_path": args.train_path,
            "validation_path": args.validation_path,
            "output_dir": args.output_dir,
            "max_steps": args.max_steps,
            "resume_from_checkpoint": args.resume_from_checkpoint,
        },
    )
    result = run_training(config)
    printable = {key: value for key, value in result.items() if key != "losses"}
    printable["loss_first"] = result["losses"][0] if result.get("losses") else None
    printable["loss_last"] = result["losses"][-1] if result.get("losses") else None
    print(json.dumps(printable, indent=2, default=str))


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Score the held-out suite. Use --reference for the scripted gold behaviour.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--label", default="reference")
    parser.add_argument("--matrix-output", type=Path, help="Write the A/B/C/D comparison. Unsupplied slots stay unmeasured.")
    parser.add_argument("--slot", choices=["A", "B", "C", "D"], help="Place the --reference report in this matrix slot.")
    parser.add_argument("--report", action="append", default=[], help="Existing report as SLOT=directory, for example B=outputs/eval.")
    args = parser.parse_args()
    configure_logging()
    from myna_ai.evaluation.matrix import build_comparison_matrix, load_report_slot, write_matrix
    from myna_ai.evaluation.runner import evaluate_generator, reference_generator, write_report

    if not args.reference and not args.matrix_output:
        parser.error("Pass --reference, or --matrix-output with --report. A 7B model is not downloaded by this command.")
    if args.reference and args.output is None:
        parser.error("--output is required with --reference.")
    reports: dict = {}
    payload: dict = {}
    if args.reference:
        report = evaluate_generator(reference_generator(), model_label=args.label)
        json_path, markdown_path = write_report(report, args.output)
        payload = {"json": str(json_path), "markdown": str(markdown_path), "promote": report["promotion"]["promote"]}
        if args.slot:
            reports[args.slot] = report
    for item in args.report:
        slot, report = load_report_slot(item)
        reports[slot] = report
    if args.matrix_output:
        matrix = build_comparison_matrix(reports)
        json_path, markdown_path = write_matrix(matrix, args.matrix_output)
        payload["matrix_json"] = str(json_path)
        payload["matrix_markdown"] = str(markdown_path)
        payload["olmo_7b_baseline_measured"] = matrix["olmo_7b_baseline_measured"]
    print(json.dumps(payload, indent=2))


def export() -> None:
    parser = argparse.ArgumentParser(description="Export an adapter, a merged model, or a full-precision copy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--merge", action="store_true", help="Merge the LoRA adapter into a local full-precision model.")
    parser.add_argument("--full-precision", action="store_true", help="Copy safetensors and sidecars, excluding optimizer state.")
    parser.add_argument("--gguf-check", action="store_true")
    parser.add_argument("--gguf-convert", action="store_true", help="Run a converter. A written file is not a verified llama.cpp load.")
    parser.add_argument("--gguf-model-dir", type=Path)
    parser.add_argument("--gguf-converter", type=Path)
    args = parser.parse_args()
    configure_logging()
    if args.merge and args.full_precision:
        parser.error("Pass only one of --merge and --full-precision.")
    if args.gguf_convert and args.gguf_model_dir is None:
        parser.error("--gguf-model-dir is required with --gguf-convert.")
    from myna_ai.export.gguf import assess_gguf, convert_to_gguf
    from myna_ai.models.registry import require_model

    spec = require_model(args.model_key)
    metadata = {
        "model_key": spec.key,
        "base_model": spec.hf_id,
        "promotion_allowed": False,
        "notes": "Promotion stays false until the evaluation gate is applied to this artifact.",
    }
    if args.merge:
        from myna_ai.export.merge import merge_adapter_checkpoint

        manifest = merge_adapter_checkpoint(args.checkpoint, args.output, metadata)
    elif args.full_precision:
        from myna_ai.export.full_precision import export_full_precision

        manifest = export_full_precision(args.checkpoint, args.output, {**metadata, "artifact_type": "full_precision"})
    else:
        from myna_ai.export.manifest import export_adapter_directory

        manifest = export_adapter_directory(
            args.checkpoint,
            args.output,
            {**metadata, "artifact_type": "lora_adapter"},
        )
    payload: dict = {"manifest_files": len(manifest["files"]), "artifact_type": manifest.get("artifact_type")}
    if args.gguf_convert:
        gguf_output = args.output.parent / f"{args.output.name}.gguf"
        payload["gguf"] = convert_to_gguf(
            spec,
            model_dir=args.gguf_model_dir,
            output=gguf_output,
            converter=args.gguf_converter,
            execute=True,
        )
    elif args.gguf_check:
        payload["gguf"] = assess_gguf(spec, converter=args.gguf_converter)
    print(json.dumps(payload, indent=2))


def benchmark() -> None:
    parser = argparse.ArgumentParser(description="Write a resource note for a completed artifact directory.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_logging()
    from myna_ai.common.io import write_json

    files = [path for path in args.artifact.rglob("*") if path.is_file()]
    report = {
        "artifact": str(args.artifact),
        "bytes": sum(path.stat().st_size for path in files),
        "file_count": len(files),
        "latency_measured": False,
        "note": "Pass a loaded runtime to evaluate_generator for latency. This command does not download a model.",
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
