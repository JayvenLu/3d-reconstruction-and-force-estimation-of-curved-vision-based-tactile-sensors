"""Versioned marker detection and image/gradient completion shared by training and inference."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.spatial import cKDTree, Delaunay, QhullError

ROOT = Path(__file__).resolve().parent
METHODS = {"image": ("telea", "ns", "nearest", "linear", "cubic"),
           "gradient": ("nearest", "linear", "cubic", "ns", "telea")}


def default_config(route="image", method=None):
    return {
        "schema_version": 1,
        "route": route,
        "interpolation": {"method": method or ("telea" if route == "image" else "nearest"),
                          "radius": 3, "dilate_iterations": 5, "donor_ring": 5},
        "detector": {"block_size": 151, "c": 26, "blur_size": 35,
                     "close_size": 11, "dilate_size": 5, "erode_size": 3},
        "input": {"use_pos": True, "use_nir": False, "nir_scale": 1.0,
                  "color_order": "BGR", "intensity_divisor": 255.0,
                  "nir_read": "opencv_u8", "background": "difference",
                  "position": "width_normalized_y_up", "height": 480, "width": 640},
        "network": {"class": "Modelv2_ori", "kwargs": {"dropout": 0.1, "output_dims": [256, 128, 64]}},
        "training": {"seed": 240, "pixels_per_sample": 2048, "background_probability": 0.5,
                     "learning_rate": 0.001, "weight_decay": 0.0, "max_epochs": 300,
                     "loss": "sum_component_l1_unit_normals", "optimizer": "adam", "precision": 32,
                     "scheduler": "cosine", "scheduler_eta_min": 1e-5,
                     "early_stopping_patience": 40, "early_stopping_min_delta": 1e-5,
                     "batch_size": 32, "mask_source": "prefer_saved",
                     "pixel_policy": "gt_valid" if route == "image" else "exclude_current_background_union"},
        "data": {"record": "20260404_103156", "train_split": "train", "val_split": "val"},
        "depth": {"prior": "depth_prior/dep_p_20260407_181422.npy", "prior_sha256": None,
                  "backend": "cpu", "eps": 1e-6, "unit": "mm", "gradient_unit": "mm/pixel",
                  "convention": "p=-nx/nz;q=ny/nz", "numeric_chain": "v1_rect_boundary_dirichlet_fp32"},
    }


def merge_config(base, patch):
    result = copy.deepcopy(base)
    for key, value in patch.items():
        result[key] = merge_config(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported geometry configuration schema")
    route = config["route"]
    if route not in METHODS or config["interpolation"]["method"] not in METHODS[route]:
        raise ValueError("Unsupported completion route/method")
    for key in ("block_size", "blur_size", "close_size", "dilate_size", "erode_size"):
        v = config["detector"][key]
        if not isinstance(v, int) or v < (3 if key == "block_size" else 1) or v % 2 != 1:
            raise ValueError(f"detector.{key} must be a positive odd integer")
    ip = config["interpolation"]
    if ip["dilate_iterations"] < 0 or ip["donor_ring"] < 1 or ip["radius"] <= 0:
        raise ValueError("Invalid interpolation radius/dilation/donor ring")
    expected = "gt_valid" if route == "image" else "exclude_current_background_union"
    if config["training"]["pixel_policy"] != expected:
        raise ValueError(f"{route} requires training.pixel_policy={expected}")
    inp = config["input"]
    fixed = {"color_order": "BGR", "intensity_divisor": 255.0, "nir_read": "opencv_u8",
             "background": "difference", "position": "width_normalized_y_up"}
    if any(inp[k] != v for k, v in fixed.items()):
        raise ValueError("Unsupported input/coordinate convention")
    if (config["depth"]["convention"] != "p=-nx/nz;q=ny/nz" or config["depth"]["unit"] != "mm"
            or config["depth"]["gradient_unit"] != "mm/pixel"
            or config["depth"]["numeric_chain"] != "v1_rect_boundary_dirichlet_fp32"):
        raise ValueError("Unsupported depth convention/unit")
    if config["depth"]["backend"] not in ("cpu", "gpu") or config["depth"]["eps"] <= 0:
        raise ValueError("Invalid depth solver settings")
    if config["training"]["pixels_per_sample"] < 1 or not 0 <= config["training"]["background_probability"] <= 1:
        raise ValueError("Invalid pixel sampling settings")
    if config["training"]["mask_source"] not in ("auto", "saved", "prefer_saved"):
        raise ValueError("Unknown training mask source")
    fixed_training = {"loss": "sum_component_l1_unit_normals", "optimizer": "adam", "scheduler": "cosine", "precision": 32}
    if any(config["training"].get(k, v) != v for k, v in fixed_training.items()):
        raise ValueError("This pipeline preserves the original loss, optimizer, scheduler and precision")
    if min(inp["height"], inp["width"], inp["nir_scale"]) <= 0:
        raise ValueError("Input dimensions and NIR scale must be positive")
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def detect_markers(frame, detector=None):
    d = detector or default_config()["detector"]
    if frame is None:
        raise ValueError("Marker detector received no image")
    if frame.ndim == 3:
        v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
    else:
        v = frame
        if v.dtype == np.uint16:
            v = (v >> 8).astype(np.uint8)
    if v.dtype != np.uint8:
        raise ValueError("Detector expects uint8 RGB/gray or uint16 gray")
    v = cv2.GaussianBlur(v, (d["blur_size"],) * 2, 0)
    mask = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, d["block_size"], d["c"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((d["close_size"],) * 2, np.uint8))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d["dilate_size"],) * 2))
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d["erode_size"],) * 2))


def expand_mask(mask, iterations):
    mask = (np.asarray(mask) != 0).astype(np.uint8) * 255
    if iterations:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=int(iterations))
    return mask != 0


def legacy_inpaint(frame, mask, method="telea", radius=3, dilate_iter=1):
    """Preserve the historical OpenCV behavior, return type and defaults."""
    if method not in ("telea", "ns"):
        raise ValueError(f"unknown inpaint method: {method}")
    used = expand_mask(mask, max(0, dilate_iter)).astype(np.uint8) * 255
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(frame, used, radius, flag), used


def complete_field(field, mask, method="nearest", donor_ring=5, radius=3):
    """Fill masked values only; one interpolation geometry shared by all channels.

    Donors are finite pixels in a ring around each connected missing component.
    Degenerate/out-of-hull linear/cubic targets use nearest donors and are counted.
    """
    src = np.asarray(field)
    missing = np.asarray(mask, dtype=bool)
    if src.shape[:2] != missing.shape or src.ndim not in (2, 3):
        raise ValueError("Field and mask dimensions differ")
    if method not in ("nearest", "linear", "cubic", "telea", "ns"):
        raise ValueError(f"Unknown completion method: {method}")
    count = int(missing.sum())
    info = {"masked_pixels": count, "fallback_pixels": 0, "fallback_fraction": 0.0}
    result = src.copy()
    if not count:
        return result, info
    data = src[..., None] if src.ndim == 2 else src
    finite = np.isfinite(data).all(axis=-1)
    donors_valid = ~missing & finite
    if not donors_valid.any():
        raise ValueError("No finite unmasked interpolation donors")
    if method in ("telea", "ns"):
        if not finite[~missing].all():
            raise ValueError("Non-finite image donors")
        flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        m = missing.astype(np.uint8) * 255
        if src.dtype == np.uint8 and (src.ndim == 2 or src.shape[2] == 3):
            repaired = cv2.inpaint(src, m, radius, flags)
        else:
            repaired = np.stack([cv2.inpaint(data[..., c].astype(np.float32), m, radius, flags)
                                 for c in range(data.shape[-1])], axis=-1)
            if src.ndim == 2:
                repaired = repaired[..., 0]
        result[missing] = repaired[missing]
        return result, info
    out = result[..., None] if result.ndim == 2 else result
    n, labels, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
    for label in range(1, n):
        x, y, w, h, _ = stats[label]
        pad = int(donor_ring)
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(src.shape[1], x+w+pad), min(src.shape[0], y+h+pad)
        component = labels[y0:y1, x0:x1] == label
        ring = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=pad) != 0
        donor = ring & donors_valid[y0:y1, x0:x1]
        points = np.argwhere(donor) + [y0, x0]
        targets = np.argwhere(component) + [y0, x0]
        if not len(points):
            # Boundary/large masks can have no donors in the configured ring.
            points = np.argwhere(donors_valid)
        values = data[points[:, 0], points[:, 1]].astype(np.float64)
        nearest_indices = cKDTree(points).query(targets)[1]
        nearest = values[nearest_indices]
        if method == "nearest":
            filled = nearest
        else:
            try:
                triangulation = Delaunay(points)
                cls = LinearNDInterpolator if method == "linear" else CloughTocher2DInterpolator
                filled = cls(triangulation, values)(targets)
                bad = ~np.isfinite(filled).all(axis=-1)
            except (QhullError, ValueError):
                filled = nearest.copy()
                bad = np.ones(len(targets), dtype=bool)
            filled[bad] = nearest[bad]
            info["fallback_pixels"] += int(bad.sum())
        if np.issubdtype(src.dtype, np.integer):
            bounds = np.iinfo(src.dtype)
            filled = np.clip(np.rint(filled), bounds.min, bounds.max)
        out[targets[:, 0], targets[:, 1]] = filled.astype(src.dtype)
    info["fallback_fraction"] = info["fallback_pixels"] / count
    return result, info


def complete_image(image, mask, config):
    p = config["interpolation"]
    return complete_field(image, mask, p["method"], p["donor_ring"], p["radius"])


def resolve_mask(frame, split_dir, sample_id, config, source="prefer_saved"):
    if source not in ("auto", "saved", "prefer_saved"):
        raise ValueError(f"Unknown mask source: {source}")
    path = Path(split_dir) / "marker_masks" / f"{sample_id}.png"
    if source != "auto" and path.exists():
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != frame.shape[:2]:
            raise ValueError(f"Invalid saved marker mask: {path}")
        return m, "saved"
    if source == "saved":
        raise FileNotFoundError(path)
    return detect_markers(frame, config["detector"]), "auto"


def image_path(split_dir, sample_id, channel="rgb", folder="images"):
    folder = Path(split_dir) / folder
    direct = folder / f"{sample_id}_{channel}.png"
    if direct.exists():
        return direct
    matches = [p for p in folder.glob(f"*_{channel}.png") if p.stem.rsplit('_', 1)[0].isdigit()
               and int(p.stem.rsplit('_', 1)[0]) == sample_id]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one image for {sample_id}/{channel}: {folder}")
    return matches[0]


def read_image(path, nir=False, scale=1.0):
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if nir else cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError(path)
    if nir and scale != 1.0:
        a = np.clip(a.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return a


def discover_ids(split_dir, required=("normal_gt",), use_nir=False):
    split_dir = Path(split_dir)
    ids = sorted({int(p.stem[:-4]) for p in (split_dir / "images").glob("*_rgb.png")
                  if p.stem[:-4].isdigit() and int(p.stem[:-4]) != 0})
    templates = {"normal_gt": "normal{}.npy", "normal_complete_gt": "normal_complete{}.npy",
                 "depth_gt": "depth{}.npy", "mask": "mask{}.npy"}
    accepted, missing = [], []
    for sid in ids:
        absent = [name for name in required if not (split_dir / name / templates[name].format(sid)).exists()]
        if use_nir:
            try:
                image_path(split_dir, sid, "nir")
            except FileNotFoundError:
                absent.append("nir")
        if absent:
            missing.append({"id": sid, "missing": absent})
        else:
            accepted.append(sid)
    return accepted, missing


def resolve_prior(config, version_dir=None):
    p = Path(config["depth"]["prior"])
    if not p.is_absolute():
        p = (Path(version_dir) / p) if version_dir and (Path(version_dir) / p).exists() else ROOT / p
    expected = config["depth"].get("prior_sha256")
    if expected and file_hash(p) != expected:
        raise ValueError(f"Depth prior checksum mismatch: {p}")
    return p


def normals_to_gradients(normal, eps=1e-6):
    from depth_reconstruction_core import _normal_to_pq
    p, q = _normal_to_pq(np.asarray(normal, dtype=np.float32), eps=eps)
    return np.stack((p, q), axis=-1)


def gradients_to_normals(gradient):
    n = np.stack((-gradient[..., 0], gradient[..., 1], np.ones(gradient.shape[:2])), axis=-1).astype(np.float32)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def audit_split_duplicates(train_dir, evaluation_dir):
    """Compare content, not filenames; repeated geometry alone is not leakage proof."""
    train_files = sorted((Path(train_dir) / "images").glob("*_rgb.png"))
    eval_files = sorted((Path(evaluation_dir) / "images").glob("*_rgb.png"))
    train_hashes = {file_hash(p) for p in train_files if int(p.stem[:-4]) != 0}
    evaluated = [p for p in eval_files if int(p.stem[:-4]) != 0]
    duplicate_ids = [int(p.stem[:-4]) for p in evaluated if file_hash(p) in train_hashes]
    return {"evaluated_images": len(evaluated), "duplicate_images": len(duplicate_ids),
            "duplicate_rate": len(duplicate_ids) / len(evaluated) if evaluated else 0.0,
            "duplicate_ids": duplicate_ids,
            "interpretation": "replay_validation_not_generalization" if duplicate_ids else "no_exact_image_duplicates_detected"}
