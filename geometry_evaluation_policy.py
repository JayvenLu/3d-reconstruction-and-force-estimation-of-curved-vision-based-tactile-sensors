"""Recover actual training mask parameters without rewriting historical models."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from geometry_processing import file_hash, image_path, read_image, resolve_mask


def training_mask_config(config, source="training", mask_source="auto"):
    effective = copy.deepcopy(config)
    audit = {"detector_source": source, "saved_detector": config["detector"],
             "effective_detector": config["detector"], "manifest_records": 0}
    if source == "model":
        audit["interpretation"] = "explicit model-default detector; training-mask equality not required"
        return effective, audit
    if source != "training":
        raise ValueError("Unknown detector source")
    manifests = config["data"].get("cache_manifest", {})
    records = [item["preparation"] for split in manifests.values() for item in split.values()]
    audit["manifest_records"] = len(records)
    if not records:
        audit["interpretation"] = "no archived preparation records; model configuration fallback"
        return effective, audit
    detectors, unresolved, edited = {}, 0, 0
    for prep in records:
        saved = prep.get("resolved_mask_source") == "saved" or bool(prep["sources"].get("saved_mask"))
        edit = prep.get("mask_edit_record", {})
        detector = edit.get("detector") if saved else prep.get("detector")
        if detector is None:
            unresolved += 1
        else:
            detectors[json.dumps(detector, sort_keys=True)] = detector
        edited += bool(edit.get("deleted_centers") or edit.get("extra_dilation_applied"))
    audit.update(unresolved_records=unresolved, manually_edited_records=edited,
                 distinct_detectors=list(detectors.values()))
    if mask_source != "saved" and (unresolved or edited or len(detectors) != 1):
        raise ValueError("Cannot reproduce training masks with one automatic detector; "
                         "use --mask-source saved, or explicitly --detector-source model for an unmatched experiment")
    if len(detectors) == 1 and not unresolved:
        effective["detector"] = copy.deepcopy(next(iter(detectors.values())))
    audit["effective_detector"] = effective["detector"]
    audit["interpretation"] = "actual archived preparation detector; exact masks checked on recorded samples"
    return effective, audit


def verify_training_masks(config, effective, split_dir, sample_ids, mask_source):
    """Preflight only: hash validation and mask comparison never enter timed inference."""
    split_dir = Path(split_dir)
    manifests = config["data"].get("cache_manifest", {})
    preparations = {}
    for split in manifests.values():
        for sid, item in split.items():
            prep = item["preparation"]
            if Path(prep["split"]).resolve() == split_dir.resolve():
                preparations[int(sid)] = prep
    checked, unrecorded = [], []
    for sid in sample_ids:
        if sid not in preparations:
            unrecorded.append(sid)
            continue
        prep = preparations[sid]
        rgb_path = image_path(split_dir, sid)
        if file_hash(rgb_path) != prep["sources"]["rgb"]:
            raise ValueError(f"Training source image changed: {rgb_path}")
        if prep["dilate_iterations"] != effective["interpolation"]["dilate_iterations"]:
            raise ValueError(f"Training dilation mismatch for sample {sid}")
        raw = read_image(rgb_path)
        training_config = copy.deepcopy(config)
        training_config["detector"] = prep["detector"]
        if prep["sources"].get("saved_mask"):
            mask_path = split_dir / "marker_masks" / f"{sid}.png"
            if not mask_path.exists() or file_hash(mask_path) != prep["sources"]["saved_mask"]:
                raise ValueError(f"Training marker mask missing or changed: {mask_path}")
            expected, _ = resolve_mask(raw, split_dir, sid, training_config, "saved")
        else:
            expected, _ = resolve_mask(raw, split_dir, sid, training_config, "auto")
        actual, _ = resolve_mask(raw, split_dir, sid, effective, mask_source)
        if not np.array_equal(actual != 0, expected != 0):
            raise ValueError(f"Evaluation mask differs from training for sample {sid}; "
                             "use --mask-source saved for edited masks")
        checked.append(sid)
    return {"verified_sample_ids_including_background": checked, "unrecorded_sample_ids": unrecorded,
            "exact_replay_verified": bool(checked) and not unrecorded,
            "timing": "preflight excluded; automatic inference re-detects each frame"}
