"""Standard-library-only resource checks; never open a camera or load a checkpoint."""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Resources:
    # Historical examples, not resources supplied with this source release.
    geo_version: int = 118
    force_version: int = 76
    prior: str = "depth_prior/dep_p_20260407_181422.npy"
    preset: str = "cam/20260827-145539/grid_preset_rgb_perspective.npy"


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Resource must be inside this project: {relative}")
    return path


def literals(path):
    """Read saved configuration without executing a training script."""
    values = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values


def _required(values, keys, source):
    missing = sorted(set(keys) - values.keys())
    if missing:
        raise ValueError(f"Incomplete model configuration {source}: missing {', '.join(missing)}")
    return {key: values[key] for key in keys}


def geometry_settings(version, root=ROOT):
    directory = local_path(root, f"logs/gel_experiment/version_{version}")
    config = directory / "geometry_config.json"
    if config.is_file():
        saved = json.loads(config.read_text(encoding="utf-8"))
        inp = _required(saved["input"], ("use_pos", "use_nir"), config)
        # The real-time path is the existing image-domain Modelv2_ori pipeline.
        # Other architectures/routes remain available through offline GeometryPipeline.
        if saved.get("route") != "image" or saved["network"]["class"] != "Modelv2_ori":
            raise ValueError(f"{config}: real-time inference supports image-route Modelv2_ori only; use offline evaluation for other routes")
        kwargs = saved["network"].get("kwargs", {})
        if kwargs.get("output_dims", [256, 128, 64]) != [256, 128, 64] or kwargs.get("dropout", .1) != .1:
            raise ValueError(f"{config}: architecture differs from the real-time Modelv2_ori defaults")
        if (saved["input"].get("width"), saved["input"].get("height")) != (640, 480):
            raise ValueError(f"{config}: the supplied real-time pipeline expects 640x480 calibration")
        result = {"record": saved.get("data", {}).get("record"), "use_pos": inp["use_pos"], "nir_training": inp["use_nir"]}
    else:
        candidates = [directory / "3_training_pix2normal.py", directory / "code/3_training_pix2normal.py"]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Missing geometry configuration: {config} or matching training source snapshot")
        values = literals(source)
        result = _required(values, ("use_pos", "nir_training"), source)
        result["record"] = values.get("record")
    if any(type(result[key]) is not bool for key in ("use_pos", "nir_training")):
        raise ValueError(f"Geometry version {version}: input flags must be booleans")
    if result["nir_training"]:
        raise ValueError("The supplied live camera path provides BGR only; NIR models require a synchronized NIR input implementation")
    return result


FORCE_KEYS = ("record", "pixmm", "use_nir", "lag_dz", "pos_dim", "depth_source",
              "normalize_disp", "convert_dz_to_pixel", "convert_disp_to_sensor", "use_bg_sub")


def force_settings(version, root=ROOT):
    directory = local_path(root, f"force_logs/HyperNet_Run/version_{version}")
    config = directory / "force_config.json"
    if config.is_file():
        values = json.loads(config.read_text(encoding="utf-8"))
    else:
        candidates = [directory / "code/3_training_dep2force_2.py", directory / "3_training_dep2force_2.py"]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Missing force configuration: {config} or matching training source snapshot")
        values = literals(source)
        # Older training snapshots omit this keyword and use the saved dataset default.
        dataset = directory / "code/dataset.py"
        if "use_bg_sub" not in values and dataset.is_file():
            tree = ast.parse(dataset.read_text(encoding="utf-8-sig"))
            for cls in tree.body:
                if isinstance(cls, ast.ClassDef) and cls.name == "Dataset_gel_force_dynamic_weight":
                    for method in cls.body:
                        if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                            for arg, default in zip(method.args.args[-len(method.args.defaults):], method.args.defaults):
                                if arg.arg == "use_bg_sub":
                                    values[arg.arg] = ast.literal_eval(default)
        config = source
    result = _required(values, FORCE_KEYS, config)
    for key in ("use_nir", "lag_dz", "normalize_disp", "convert_dz_to_pixel", "convert_disp_to_sensor", "use_bg_sub"):
        if type(result[key]) is not bool:
            raise ValueError(f"{config}: {key} must be a boolean")
    if result["depth_source"] not in ("pred", "gt") or not isinstance(result["pixmm"], (int, float)) or result["pixmm"] <= 0:
        raise ValueError(f"{config}: invalid depth_source or pixmm")
    if type(result["pos_dim"]) is not int or result["pos_dim"] not in (2, 3):
        raise ValueError(f"{config}: pos_dim must be 2 or 3")
    return result


def resource_errors(resources=Resources(), root=ROOT):
    """Check presence/configuration, not numerical validity or device compatibility."""
    root = Path(root).resolve()
    errors = []
    files = [(resources.prior, "boundary depth prior; generate with get_depth_prior.py"),
             (resources.preset, "marker grid; generate with calibrate_grid.py"),
             (f"logs/gel_experiment/version_{resources.geo_version}/model.py", "geometry model source saved during training"),
             (f"force_logs/HyperNet_Run/version_{resources.force_version}/code/model.py", "force model source saved during training"),
             (f"force_logs/HyperNet_Run/version_{resources.force_version}/code/dataset.py", "force coordinate/preprocessing source saved during training")]
    for relative, purpose in files:
        try:
            path = local_path(root, relative)
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing {relative} ({purpose})")
        except ValueError as exc:
            errors.append(str(exc))
    for base, version in (("logs/gel_experiment", resources.geo_version), ("force_logs/HyperNet_Run", resources.force_version)):
        directory = local_path(root, f"{base}/version_{version}/checkpoints")
        if not any(p.is_file() and p.stat().st_size for p in directory.rglob("*.ckpt")):
            errors.append(f"Missing {directory.relative_to(root)}/*.ckpt (train the corresponding model)")
    for loader, version in ((geometry_settings, resources.geo_version), (force_settings, resources.force_version)):
        try:
            settings = loader(version, root)
            if loader is force_settings and settings["normalize_disp"]:
                tag = f"{float(settings['pixmm']):.6f}".replace(".", "p").replace("-", "m")
                filename = (f"disp_norm_stats_depth_{settings['depth_source']}_pixmm_{tag}"
                            f"_lag_{int(settings['lag_dz'])}_bgsub_{int(settings['use_bg_sub'])}"
                            f"_dz2pixel_{int(settings['convert_dz_to_pixel'])}"
                            f"_disp2sensor_{int(settings['convert_disp_to_sensor'])}.npz")
                stats = local_path(root, f"data/{settings['record']}/train/{filename}")
                if not stats.is_file() or not stats.stat().st_size:
                    errors.append(f"Missing {stats.relative_to(root)} (normalization statistics from this model's training data)")
        except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
            errors.append(str(exc))
    # These live models import root sources. Reject mismatched archived definitions
    # instead of silently combining a checkpoint with another model/coordinate system.
    for relative, _ in files[2:]:
        saved, current = local_path(root, relative), root / Path(relative).name
        if saved.is_file() and current.is_file():
            try:
                if ast.dump(ast.parse(saved.read_text(encoding="utf-8-sig"))) != ast.dump(ast.parse(current.read_text(encoding="utf-8-sig"))):
                    errors.append(f"Source mismatch: {relative} and {current.name}; use sources from the same training run")
            except (ValueError, SyntaxError) as exc:
                errors.append(f"Invalid source snapshot {relative}: {exc}")
    return errors


def require_resources(resources=Resources(), root=ROOT):
    errors = resource_errors(resources, root)
    if errors:
        raise FileNotFoundError("Runtime resources are not ready:\n- " + "\n- ".join(errors) + "\nSee README.md: Required resources and Workflow. No camera has been opened.")
