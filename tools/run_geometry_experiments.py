"""Run six geometry trainings sequentially, then evaluate ten processing pipelines."""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry_processing import load_config, validate_config, file_hash, save_json, METHODS

MATRIX = [("image", m) for m in ("nearest", "linear", "cubic", "telea", "ns")] + [("gradient", "nearest")]
CODE_FILES = ("3_training_pix2normal.py", "model.py", "geometry_processing.py", "geometry_dataset.py",
              "infer_geo_offline.py", "geometry_pipeline.py", "geometry_evaluation.py", "depth_reconstruction_core.py",
              "geometry_evaluation_policy.py",
              "tools/run_geometry_experiments.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    save_json(temp, value)
    temp.replace(path)


def make_jobs(base, run_dir):
    """All shared settings come from one template, preventing accidental confounding."""
    jobs = []
    for route, method in MATRIX:
        cfg = copy.deepcopy(base)
        cfg["route"] = route
        cfg["interpolation"]["method"] = method
        cfg["training"]["pixel_policy"] = "gt_valid" if route == "image" else "exclude_current_background_union"
        validate_config(cfg)
        name = f"{route}_{method}"
        jobs.append({"name": name, "config": cfg, "status": "pending",
                     "config_path": str(run_dir / "configs" / f"{name}.json"),
                     "receipt_path": str(run_dir / "receipts" / f"{name}.json")})
    return jobs


def train_command(job, options):
    return [sys.executable, "-B", str(ROOT / "3_training_pix2normal.py"),
            "--config", job["config_path"], "--workers", str(options["workers"]),
            "--device", options["device"], "--result-json", job["receipt_path"]]


def checked_receipt(job):
    receipt = read_json(job["receipt_path"])
    version_dir = ROOT / "logs" / "gel_experiment" / f"version_{int(receipt['version'])}"
    if Path(receipt["version_dir"]).resolve() != version_dir.resolve():
        raise ValueError("Receipt points to an unexpected model directory")
    checkpoint = Path(receipt["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(version_dir.resolve()):
        raise ValueError("Checkpoint is outside the recorded version")
    cfg_path = version_dir / "geometry_config.json"
    if file_hash(checkpoint) != receipt["checkpoint_sha256"] or file_hash(cfg_path) != receipt["config_sha256"]:
        raise ValueError(f"Model artifacts changed: {job['name']}")
    if Path(receipt["requested_config"]).resolve() != Path(job["config_path"]).resolve():
        raise ValueError("Training receipt does not match the requested configuration file")
    actual = load_config(cfg_path)
    expected = job["config"]
    for key in ("route", "interpolation", "detector", "input", "network"):
        if actual[key] != expected[key]:
            raise ValueError(f"Saved model configuration mismatch: {key}")
    for group in ("training", "data"):
        if any(actual[group].get(k) != v for k, v in expected[group].items()):
            raise ValueError(f"Saved model configuration mismatch: {group}")
    if not (version_dir / "training_result.json").exists():
        raise ValueError("Training has not exported its final result")
    return receipt


def experiment_entries(jobs, gradient_methods=None):
    gradient_methods = METHODS["gradient"] if gradient_methods is None else tuple(gradient_methods)
    if not gradient_methods or len(set(gradient_methods)) != len(gradient_methods) or any(m not in METHODS["gradient"] for m in gradient_methods):
        raise ValueError("Select distinct supported gradient methods")
    entries = []
    for job in jobs:
        methods = [job["config"]["interpolation"]["method"]] if job["config"]["route"] == "image" else gradient_methods
        entries.extend({"version": job["result"]["version"], "method": method, "source": "raw"} for method in methods)
    return entries


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(subprocess.list2cmdline(command) + "\n")
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            code = child.wait()
            if code:
                raise RuntimeError(f"Child exited with {code}; see {log_path}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()


@contextmanager
def exclusive_run(run_dir):
    """An OS lock is released even if the launcher crashes; no stale PID cleanup."""
    with (run_dir / "runner.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run_suite(run_dir, state, runner=run_logged, receipt_reader=checked_receipt):
    state_path = run_dir / "run.json"
    def save():
        atomic_json(state_path, state)
    try:
        for index, job in enumerate(state["jobs"], 1):
            if Path(job["receipt_path"]).exists():
                job["result"] = receipt_reader(job)
                job["status"] = "complete"
                save()
                print(f"[Batch] Skip completed {job['name']}: version {job['result']['version']}", flush=True)
                continue
            if job["status"] == "complete":
                raise ValueError(f"Completed job lost its receipt: {job['name']}")
            job["status"] = "running"
            job["attempts"] = job.get("attempts", 0) + 1
            save()
            print(f"[Batch] Training {index}/6: {job['name']}", flush=True)
            runner(train_command(job, state["options"]), run_dir / "logs" / f"{job['name']}_{job['attempts']}.log")
            job["result"] = receipt_reader(job)
            job["status"] = "complete"
            save()
        entries = experiment_entries(state["jobs"])
        experiments = run_dir / "experiments.json"
        atomic_json(experiments, {"experiments": entries})
        evaluation = state.setdefault("evaluation", {"status": "pending", "attempts": 0})
        if evaluation["status"] != "complete":
            evaluation["attempts"] += 1
            evaluation["status"] = "running"
            output = run_dir / f"evaluation_{evaluation['attempts']:02d}"
            evaluation["output"] = str(output)
            save()
            opts = state["options"]
            data = state["jobs"][0]["config"]["data"]
            cmd = [sys.executable, "-B", str(ROOT / "infer_geo_offline.py"), "--experiments", str(experiments),
                   "--record", data["record"], "--split", data["val_split"], "--device", opts["device"],
                   "--mask-source", opts["eval_mask_source"], "--solver-backend", opts["solver_backend"],
                   "--detector-source", "training",
                   "--warmup", str(opts["warmup"]), "--repeats", str(opts["repeats"]), "--output", str(output)]
            print(f"[Batch] All six trainings complete; evaluating {len(entries)} pipelines", flush=True)
            runner(cmd, run_dir / "logs" / f"evaluation_{evaluation['attempts']}.log")
        output = Path(evaluation["output"])
        report = read_json(output / "summary.json")
        if len(report["experiments"]) != len(entries) or any(not e["comparable"] or e["failed_frames"] for e in report["experiments"]):
            raise ValueError("Evaluation is incomplete or contains failed frames")
        if not (output / "report.md").exists():
            raise ValueError("Evaluation report was not written")
        evaluation["status"] = "complete"
        state["status"] = "complete"
        state.pop("error", None)
        save()
        print(f"[Batch] Complete: {Path(evaluation['output']) / 'report.md'}", flush=True)
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        for job in state["jobs"]:
            if job["status"] == "running": job["status"] = state["status"]
        if state.get("evaluation", {}).get("status") == "running":
            state["evaluation"]["status"] = state["status"]
        state["error"] = str(exc) or type(exc).__name__
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", help="Shared template; defaults to configs/geometry/image_telea.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--train-mask-source", choices=("auto", "saved", "prefer_saved"))
    parser.add_argument("--eval-mask-source", choices=("auto", "saved", "prefer_saved"))
    parser.add_argument("--solver-backend", choices=("cpu", "gpu"))
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--run-dir", help="New output directory; must not already exist")
    parser.add_argument("--resume", help="Resume an existing run directory, skipping completed trainings")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without creating files or starting jobs")
    args = parser.parse_args()
    if args.resume:
        if any(getattr(args, name) is not None for name in ("base_config", "epochs", "batch_size", "workers", "device", "train_mask_source", "eval_mask_source", "solver_backend", "warmup", "repeats", "run_dir")):
            parser.error("Resume uses the saved settings; do not supply overrides")
        run_dir = Path(args.resume).resolve()
        state = read_json(run_dir / "run.json")
        for name, sha in state["code_sha256"].items():
            if file_hash(ROOT / name) != sha:
                raise ValueError(f"Code changed since run creation: {name}; start a new run to avoid mixing implementations")
        for job in state["jobs"]:
            if read_json(job["config_path"]) != job["config"]:
                raise ValueError(f"Run configuration changed: {job['config_path']}")
    else:
        base = load_config(Path(args.base_config).resolve() if args.base_config else ROOT / "configs/geometry/image_telea.json")
        if "sample_ids" in base["data"]:
            parser.error("Use a training preset, not a saved model's runtime configuration")
        if args.epochs is not None: base["training"]["max_epochs"] = args.epochs
        if args.batch_size is not None: base["training"]["batch_size"] = args.batch_size
        if args.train_mask_source: base["training"]["mask_source"] = args.train_mask_source
        options = {"workers": 0 if args.workers is None else args.workers, "device": args.device or "cuda",
                   "eval_mask_source": args.eval_mask_source or "auto", "solver_backend": args.solver_backend or "cpu",
                   "warmup": 10 if args.warmup is None else args.warmup, "repeats": 3 if args.repeats is None else args.repeats}
        if min(base["training"]["max_epochs"], base["training"]["batch_size"], options["repeats"]) < 1 or min(options["workers"], options["warmup"]) < 0:
            parser.error("Invalid epochs/batch size/workers/warmup/repeats")
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path(args.run_dir).resolve() if args.run_dir else ROOT / "reports/geometry/batches" / stamp
        state = {"schema_version": 1, "status": "pending", "options": options,
                 "jobs": make_jobs(base, run_dir), "code_sha256": {name: file_hash(ROOT / name) for name in CODE_FILES}}
    print(f"[Batch] Run directory: {run_dir}", flush=True)
    for job in state["jobs"]:
        print(subprocess.list2cmdline(train_command(job, state["options"])))
    print("[Batch] Then evaluate five image models + five methods of the shared gradient model.")
    if args.dry_run:
        return
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        for job in state["jobs"]:
            atomic_json(job["config_path"], job["config"])
        atomic_json(run_dir / "run.json", state)
    with exclusive_run(run_dir):
        # Read state only after acquiring the lock so concurrent resume cannot use stale state.
        run_suite(run_dir, read_json(run_dir / "run.json"))


if __name__ == "__main__":
    main()
