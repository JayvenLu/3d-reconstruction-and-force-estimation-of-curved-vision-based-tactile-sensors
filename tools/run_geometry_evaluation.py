"""Evaluate an existing six-model batch, without training or changing its artifacts."""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_geometry_experiments import (read_json, atomic_json, checked_receipt,
    experiment_entries, run_logged, CODE_FILES, MATRIX)
from geometry_processing import file_hash, METHODS


def latest_trained_batch():
    for path in sorted((ROOT / "reports/geometry/batches").glob("*/run.json"), reverse=True):
        state = read_json(path)
        if len(state.get("jobs", [])) == 6 and all(j.get("status") == "complete" for j in state["jobs"]):
            return path.parent
    raise FileNotFoundError("No completed six-model training batch; supply --batch-dir")


def prepare(args, receipt_reader=checked_receipt):
    batch_dir = Path(args.batch_dir).resolve() if args.batch_dir else latest_trained_batch()
    source = batch_dir / "run.json"
    batch = read_json(source)
    jobs = batch["jobs"]
    actual = [(j["config"]["route"], j["config"]["interpolation"]["method"]) for j in jobs]
    if sorted(actual) != sorted(MATRIX) or any(j["status"] != "complete" for j in jobs):
        raise ValueError("Expected five completed image models and one completed gradient model")
    for job in jobs:
        job["result"] = receipt_reader(job)
    data = jobs[0]["config"]["data"]
    if any((j["config"]["data"]["record"], j["config"]["data"]["val_split"]) !=
           (data["record"], data["val_split"]) for j in jobs):
        raise ValueError("Training models use different evaluation datasets")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output).resolve() if args.output else batch_dir / "re_evaluations" / stamp
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    entries = experiment_entries(jobs, getattr(args, "gradient_methods", None))
    command = [sys.executable, "-B", str(ROOT / "infer_geo_offline.py"),
               "--experiments", str(output / "experiments.json"),
               "--record", data["record"], "--split", data["val_split"],
               "--device", args.device or batch["options"]["device"],
               "--mask-source", args.mask_source, "--detector-source", args.detector_source,
               "--warmup", str(args.warmup), "--repeats", str(args.repeats),
               "--cpu-threads", str(args.cpu_threads), "--output", str(output / "evaluation")]
    if args.solver_backend:
        command += ["--solver-backend", args.solver_backend]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    state = {"schema_version": 1, "status": "pending", "source_batch": str(batch_dir),
             "source_run_sha256": file_hash(source), "command": command,
             "experiments": entries, "training_receipts": [j["result"] for j in jobs],
             "smoke_only": args.limit is not None,
             "code_sha256": {name: file_hash(ROOT / name) for name in (*CODE_FILES, "tools/run_geometry_evaluation.py")}}
    return output, state


def run_evaluation(output, state, runner=run_logged):
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "experiments.json", {"experiments": state["experiments"]})
    state["status"] = "running"
    atomic_json(output / "evaluation_run.json", state)
    try:
        runner(state["command"], output / "evaluation.log")
        report = read_json(output / "evaluation/summary.json")
        expected = [(int(e["version"]), e["method"]) for e in state["experiments"]]
        actual = [(int(e["version"]), e["effective_config"]["interpolation"]["method"]) for e in report["experiments"]]
        if actual != expected or any(not e["comparable"] or e["failed_frames"] for e in report["experiments"]):
            raise ValueError("Evaluation is incomplete, failed, or tested different models")
        if not (output / "evaluation/report.md").exists():
            raise ValueError("Missing evaluation report")
        source = Path(state["source_batch"]) / "run.json"
        if file_hash(source) != state["source_run_sha256"]:
            raise ValueError("Source batch changed during evaluation")
        state["status"] = "complete"
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        atomic_json(output / "evaluation_run.json", state)
    print(f"[Evaluation complete] {output / 'evaluation/report.md'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", help="Existing training batch; default: latest with six completed trainings")
    p.add_argument("--output", help="New directory; never reuse a previous result directory")
    p.add_argument("--device", choices=("cpu", "cuda"), help="Default: training batch device")
    p.add_argument("--mask-source", choices=("auto", "saved", "prefer_saved"), default="auto")
    p.add_argument("--detector-source", choices=("training", "model"), default="training")
    p.add_argument("--gradient-methods", nargs="+", choices=METHODS["gradient"],
                   help="Default: nearest linear cubic ns telea; image models always included")
    p.add_argument("--solver-backend", choices=("cpu", "gpu"), help="Default: model configuration")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--limit", type=int, help="Smoke verification only; omit for full evaluation")
    p.add_argument("--dry-run", action="store_true", help="Verify receipts and print command without creating files")
    args = p.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
        p.error("Invalid timing or sample limit")
    output, state = prepare(args)
    print(f"[Source batch] {state['source_batch']}")
    print(f"[Models] {[e['version'] for e in state['experiments']]}")
    print(subprocess.list2cmdline(state["command"]))
    if not args.dry_run:
        run_evaluation(output, state)


if __name__ == "__main__":
    main()
