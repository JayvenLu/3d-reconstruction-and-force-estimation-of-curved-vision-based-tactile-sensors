"""Deterministic preparation/cache; stochastic pixel sampling is training-only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from geometry_processing import (ROOT, validate_config, discover_ids, image_path, read_image,
    resolve_mask, expand_mask, complete_image, fingerprint, file_hash, save_json)


class PreparedGeometryDataset(Dataset):
    def __init__(self, split_dir, config, mode="train", sample_ids=None, cache_dir=None):
        self.split_dir = Path(split_dir).resolve()
        self.config = validate_config(copy.deepcopy(config))
        self.mode = mode
        if mode not in ("train", "eval"):
            raise ValueError("mode must be train or eval")
        self.N = config["training"]["pixels_per_sample"]
        self.cache_root = Path(cache_dir or ROOT / ".cache" / "geometry")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.prepared = {}
        self.audit = {"excluded": [], "missing": [], "cache_hits": 0, "cache_misses": 0}
        self._prepare(0)
        self.background = self._load(0)
        valid_bg = self._valid(self.background, background=True)
        if not valid_bg.any():
            self.audit["excluded"].append({"id": 0, "reason": "zero valid background pixels; background sampling disabled"})
        self.bg_indices = np.flatnonzero(valid_bg)
        h, w = self.background["rgb"].shape[:2]
        if (h, w) != (config["input"]["height"], config["input"]["width"]):
            raise ValueError("Image shape does not match saved input height/width")
        yy, xx = torch.meshgrid(-torch.linspace(-h+1, h-1, h)/w,
                                torch.linspace(-w+1, w-1, w)/w, indexing="ij")
        self.position = torch.stack((xx, yy), dim=-1).numpy()
        discovered, self.audit["missing"] = discover_ids(self.split_dir, use_nir=config["input"]["use_nir"])
        ids = discovered if sample_ids is None else list(sample_ids)
        if set(ids) - set(discovered):
            raise ValueError("Requested sample IDs have missing images/GT")
        self.sample_ids = []
        for sid in ids:
            self._prepare(sid)
            item = self._load(sid)
            if not self._valid(item).any():
                self.audit["excluded"].append({"id": sid, "reason": "zero valid pixels"})
            else:
                self.sample_ids.append(sid)
        if not self.sample_ids:
            raise ValueError(f"No usable samples: {self.split_dir}; {self.audit}")

    def _prepare(self, sid):
        cfg = self.config
        rgb_path = image_path(self.split_dir, sid)
        normal_path = self.split_dir / "normal_gt" / f"normal{sid}.npy"
        if not normal_path.exists():
            raise FileNotFoundError(normal_path)
        saved_mask = self.split_dir / "marker_masks" / f"{sid}.png"
        mask_record = saved_mask.with_suffix(".json")
        sources = {"rgb": file_hash(rgb_path), "gt": file_hash(normal_path),
                   "saved_mask": file_hash(saved_mask) if saved_mask.exists() and cfg["training"]["mask_source"] != "auto" else None}
        if sources["saved_mask"] and mask_record.exists():
            sources["saved_mask_record"] = file_hash(mask_record)
        nir_path = None
        if cfg["input"]["use_nir"]:
            nir_path = image_path(self.split_dir, sid, "nir")
            sources["nir"] = file_hash(nir_path)
        # Gradient interpolation methods do not affect the training dataset.
        prep = {"cache_schema": 1, "split": str(self.split_dir), "id": sid, "sources": sources,
                "input": cfg["input"], "route": cfg["route"], "detector": cfg["detector"],
                "mask_source": cfg["training"]["mask_source"],
                "dilate_iterations": cfg["interpolation"]["dilate_iterations"]}
        if cfg["route"] == "image":
            prep["interpolation"] = cfg["interpolation"]
        if sources.get("saved_mask_record"):
            prep["mask_edit_record"] = json.loads(mask_record.read_text(encoding="utf-8"))
        key = fingerprint(prep)
        out = self.cache_root / f"{key}.npz"
        if out.exists():
            self.audit["cache_hits"] += 1
        else:
            rgb = read_image(rgb_path)
            raw_mask, source = resolve_mask(rgb, self.split_dir, sid, cfg, cfg["training"]["mask_source"])
            mask = expand_mask(raw_mask, cfg["interpolation"]["dilate_iterations"])
            gt = np.load(normal_path).astype(np.float32)
            if gt.shape != (*rgb.shape[:2], 3) or not np.isfinite(gt).all():
                raise ValueError(f"Invalid GT: {normal_path}")
            arrays = {"rgb": rgb, "mask": mask, "gt": gt}
            if nir_path:
                arrays["nir"] = read_image(nir_path, True, cfg["input"]["nir_scale"])
                if arrays["nir"].shape != rgb.shape[:2]:
                    raise ValueError(f"RGB/NIR shape mismatch: {nir_path}")
            if cfg["route"] == "image":
                arrays["rgb"], info = complete_image(rgb, mask, cfg)
                prep["rgb_completion"] = info
                if nir_path:
                    arrays["nir"], prep["nir_completion"] = complete_image(arrays["nir"], mask, cfg)
            # Preparation is single-process, completed before DataLoader workers start.
            np.savez_compressed(out, **arrays)
            prep["resolved_mask_source"] = source
            save_json(out.with_suffix(".json"), prep)
            self.audit["cache_misses"] += 1
        self.prepared[sid] = out

    def preparation_manifest(self):
        """Archive source hashes and actual mask/edit provenance alongside model configuration."""
        return {str(sid): {"cache_fingerprint": path.stem,
                          "preparation": json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))}
                for sid, path in self.prepared.items()}

    def _load(self, sid):
        with np.load(self.prepared[sid]) as z:
            return {k: z[k] for k in z.files}

    def _valid(self, item, background=False):
        valid = np.isfinite(item["gt"]).all(axis=-1) & np.any(item["gt"] != 0, axis=-1)
        if self.config["route"] == "gradient":
            mask = item["mask"] if background else item["mask"] | self.background["mask"]
            valid &= ~mask
        return valid

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sid = self.sample_ids[index]
        rng = np.random if self.mode == "train" else np.random.RandomState(self.config["training"]["seed"] + sid)
        background = (self.mode == "train" and self.bg_indices.size > 0
                      and rng.rand() < self.config["training"]["background_probability"])
        item = self.background if background else self._load(sid)
        valid = self.bg_indices if background else np.flatnonzero(self._valid(item))
        selected = rng.choice(valid, self.N, replace=len(valid) < self.N)

        def tensor(a, divisor=1.0):
            if a.ndim == 2:
                a = a[..., None]
            x = a.reshape(-1, a.shape[-1])[selected].astype(np.float32) / divisor
            return torch.from_numpy(x.T.copy()).unsqueeze(1)

        sample = {"rgb": tensor(item["rgb"], 255), "bg_rgb": tensor(self.background["rgb"], 255),
                  "gt": tensor(item["gt"]), "position": tensor(self.position)}
        if self.config["input"]["use_nir"]:
            sample.update(nir=tensor(item["nir"], 255), bg_nir=tensor(self.background["nir"], 255))
        return sample
