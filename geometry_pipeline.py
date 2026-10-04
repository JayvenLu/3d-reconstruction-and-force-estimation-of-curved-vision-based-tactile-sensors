"""High-level raw-image API. Legacy Geo_Model.predict still accepts prepared tensors."""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

from geometry_processing import (ROOT, default_config, validate_config, resolve_prior,
    resolve_mask, expand_mask, complete_image, complete_field, normals_to_gradients,
    gradients_to_normals)
from depth_reconstruction_core import reconstruct_from_gradients


class GeometryPipeline:
    def __init__(self, version=118, device="cuda", config=None, method=None, mask_source="auto",
                 prior=None, solver_backend=None, detector_source="model"):
        from infer_geo_offline import Geo_Model
        self.model = Geo_Model(version=version, device=device)
        self.version_dir = ROOT / "logs" / "gel_experiment" / f"version_{version}"
        saved = self.model.geometry_config
        self.is_legacy = saved is None
        self.config = copy.deepcopy(saved or config or default_config())
        self.overrides = {}
        if saved is not None and config is not None and config != saved:
            raise ValueError("Use version configuration; only gradient method may be overridden")
        from geometry_evaluation_policy import training_mask_config
        self.config, self.mask_parameter_audit = training_mask_config(self.config, detector_source, mask_source)
        if self.config["detector"] != (saved or config or default_config())["detector"]:
            self.overrides["detector"] = self.mask_parameter_audit
        if saved is None:
            if self.config["route"] != "image":
                raise ValueError("Legacy weights have no evidence of marker-excluded training; gradient route requires a new model")
            self.config["input"]["use_pos"] = self.model.use_pos
            self.config["input"]["use_nir"] = self.model.use_nir
        if method is not None:
            if self.config["route"] == "image" and method != self.config["interpolation"]["method"]:
                raise ValueError("Image method differs from training; train a matching image model")
            if method != self.config["interpolation"]["method"]:
                self.overrides["interpolation.method"] = {"saved": self.config["interpolation"]["method"], "effective": method}
            self.config["interpolation"]["method"] = method
        validate_config(self.config)
        if solver_backend is not None:
            if solver_backend not in ("cpu", "gpu"):
                raise ValueError("Unknown solver backend")
            if solver_backend != self.config["depth"]["backend"]:
                self.overrides["depth.backend"] = {"saved": self.config["depth"]["backend"], "effective": solver_backend}
            self.config["depth"]["backend"] = solver_backend
        self.mask_source = mask_source
        if mask_source not in ("auto", "saved", "prefer_saved"):
            raise ValueError("Invalid mask source")
        prior_path = prior or resolve_prior(self.config, self.version_dir if saved else None)
        if prior is not None:
            self.overrides["depth.prior"] = {"saved": self.config["depth"]["prior"], "effective": str(prior)}
        self.prior = np.load(prior_path).astype(np.float32)
        if not np.isnan(self.prior).any() or np.isinf(self.prior).any():
            raise ValueError("Invalid boundary prior")
        self.prior_path = str(prior_path)
        self.device = self.model.device
        self.background_mask = None
        self.background_setup_ms = None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _check_images(self, rgb, nir):
        cfg = self.config["input"]
        if rgb is None or rgb.dtype != np.uint8 or rgb.shape != (cfg["height"], cfg["width"], 3):
            raise ValueError("Expected native-resolution uint8 BGR image matching model config")
        if self.prior.shape != rgb.shape[:2]:
            raise ValueError("Prior and image dimensions differ; no implicit rescaling allowed")
        if cfg["use_nir"] and (nir is None or nir.dtype != np.uint8 or nir.shape != rgb.shape[:2]):
            raise ValueError("Expected aligned uint8 NIR (read through read_image with saved nir_scale)")

    @staticmethod
    def _tensor(image, device):
        if image.ndim == 2:
            image = image[..., None]
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(device=device, dtype=torch.float32).div_(255).unsqueeze(0)

    def set_background(self, rgb, nir=None, split_dir=None, preprocessed=False, marker_mask=None):
        self._check_images(rgb, nir)
        if preprocessed and (not self.is_legacy or self.config["route"] != "image"):
            raise ValueError("Preprocessed input is reserved for legacy image-route references")
        t0 = time.perf_counter()
        mask, source = resolve_mask(rgb, split_dir or self.version_dir, 0, self.config, self.mask_source) if marker_mask is None else (marker_mask, "explicit")
        self.background_mask = expand_mask(mask, self.config["interpolation"]["dilate_iterations"])
        if self.config["route"] == "image" and not preprocessed:
            rgb, _ = complete_image(rgb, self.background_mask, self.config)
            if self.model.use_nir:
                nir, _ = complete_image(nir, self.background_mask, self.config)
        self.model.load_bg(rgb, nir)
        self.synchronize()
        self.background_setup_ms = (time.perf_counter()-t0)*1000
        self.background_mask_source = source

    def predict_raw(self, rgb, nir=None, *, sample_id=1, split_dir=None,
                    profile=False, preprocessed=False, marker_mask=None):
        """Return normals, gradients, depth, effective mask and optional stage times.

        NIR input must use the same read_image(..., nir=True, scale=...) as training.
        Passing preprocessed=True is reserved for the old images_removed reference.
        """
        if self.background_mask is None:
            raise RuntimeError("Call set_background first")
        self._check_images(rgb, nir)
        if preprocessed and (not self.is_legacy or self.config["route"] != "image"):
            raise ValueError("Preprocessed input is reserved for legacy image-route references")
        timings = {}
        self.synchronize()
        start = last = time.perf_counter()

        def stage(name):
            nonlocal last
            if profile:
                self.synchronize()
                now = time.perf_counter()
                timings[name] = (now-last)*1000
                last = now

        if marker_mask is None:
            base_mask, source = resolve_mask(rgb, split_dir or self.version_dir, sample_id, self.config, self.mask_source)
        else:
            base_mask, source = marker_mask, "explicit"
        mask = expand_mask(base_mask, self.config["interpolation"]["dilate_iterations"])
        effective = mask | self.background_mask
        stage("detection_ms")
        info = {"masked_pixels": 0, "fallback_pixels": 0, "fallback_fraction": 0.0}
        if self.config["route"] == "image" and not preprocessed:
            rgb, info = complete_image(rgb, mask, self.config)
            if self.model.use_nir:
                nir, _ = complete_image(nir, mask, self.config)
        stage("image_completion_ms")
        self.model.sample["rgb"] = self._tensor(rgb, self.device)
        if self.model.use_nir:
            self.model.sample["nir"] = self._tensor(nir, self.device)
        stage("input_transfer_ms")
        with torch.inference_mode():
            prediction = self.model._forward_once()
            prediction = F.normalize(prediction, p=2, dim=1).squeeze(0).permute(1, 2, 0)
        stage("network_ms")
        normal = prediction.detach().cpu().numpy()
        stage("output_transfer_ms")
        if not np.isfinite(normal).all() or np.any(np.linalg.norm(normal, axis=-1) < 1e-6):
            raise ValueError("Model produced non-finite/zero normals")
        gradient = normals_to_gradients(normal, self.config["depth"]["eps"])
        clamped_normal_pixels = int((normal[..., 2] <= self.config["depth"]["eps"]).sum())
        stage("gradient_conversion_ms")
        if self.config["route"] == "gradient":
            ip = self.config["interpolation"]
            gradient, info = complete_field(gradient, effective, ip["method"], ip["donor_ring"], ip["radius"])
        stage("gradient_completion_ms")
        depth = reconstruct_from_gradients(gradient, self.prior, self.config["depth"]["backend"], str(self.device))
        stage("poisson_ms")
        if self.config["route"] == "gradient":
            normal = gradients_to_normals(gradient)
        stage("normal_output_ms")
        self.synchronize()
        timings["profile_total_ms" if profile else "end_to_end_ms"] = (time.perf_counter()-start)*1000
        if not np.isfinite(depth).all():
            raise ValueError("Depth reconstruction contains non-finite values")
        return {"normal": normal, "gradient": gradient, "depth": depth,
                "marker_mask": effective, "timings": timings, "completion": info,
                "clamped_normal_pixels": clamped_normal_pixels,
                "mask_source": source, "input_source": "legacy_images_removed" if preprocessed else "raw"}
