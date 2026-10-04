"""Full-frame accuracy and staged/whole-pipeline timing; no training-time sampling."""
from __future__ import annotations

import argparse
import csv
import datetime
import gc
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from geometry_processing import (ROOT, default_config, load_config, save_json, file_hash,
    discover_ids, image_path, read_image, resolve_mask, expand_mask, normals_to_gradients,
    audit_split_duplicates, METHODS)
from geometry_pipeline import GeometryPipeline
from geometry_evaluation_policy import training_mask_config, verify_training_masks
from depth_reconstruction_core import reconstruct_from_gradients


def region_metrics(result, gt_gradient, gt_depth, gt_normal, region):
    count = int(region.sum())
    if not count:
        return {"pixels": 0}
    pred_g, pred_d, pred_n = result["gradient"][region], result["depth"][region], result["normal"][region]
    if not all(np.isfinite(a).all() for a in (pred_g, pred_d, pred_n)):
        raise ValueError("Non-finite predictions in fixed evaluation region")
    dg = (pred_g - gt_gradient[region]).astype(np.float64)
    dz = (pred_d - gt_depth[region]).astype(np.float64)
    dot = np.sum(pred_n.astype(np.float64) * gt_normal[region], axis=-1)
    angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    return {"pixels": count, "p_mae": float(np.abs(dg[:, 0]).mean()),
            "q_mae": float(np.abs(dg[:, 1]).mean()),
            "gradient_mae": float(np.abs(dg).mean()),
            "gradient_mae_sum": float(np.abs(dg).mean(axis=0).sum()),
            "p_rmse": float(np.sqrt(np.square(dg[:, 0]).mean())),
            "q_rmse": float(np.sqrt(np.square(dg[:, 1]).mean())),
            "gradient_rmse": float(np.sqrt(np.square(dg).mean())),
            "depth_mae": float(np.abs(dz).mean()), "depth_rmse": float(np.sqrt(np.square(dz).mean())),
            "depth_bias": float(dz.mean()), "bias_aligned_depth_mae": float(np.abs(dz-dz.mean()).mean()),
            "bias_aligned_depth_rmse": float(np.sqrt(np.square(dz-dz.mean()).mean())),
            "normal_angle_deg": float(angle.mean())}


def aggregate_regions(rows):
    output = {}
    for name in ("contact", "marker", "ring", "other", "full"):
        metrics = [r["regions"][name] for r in rows if r.get("status") == "ok" and r["regions"][name]["pixels"]]
        total = sum(m["pixels"] for m in metrics)
        output[name] = {"pixels": total, "frames": len(metrics)}
        if not total:
            continue
        for key in metrics[0]:
            if key == "pixels": continue
            if key.endswith("rmse"):
                value = np.sqrt(sum(m[key]**2 * m["pixels"] for m in metrics)/total)
            else:
                value = sum(m[key] * m["pixels"] for m in metrics)/total
            output[name][key] = float(value)
    return output


def timing_summary(rows):
    result = {}
    keys = sorted({k for row in rows for k in row})
    for key in keys:
        values = np.array([r[key] for r in rows if key in r], dtype=np.float64)
        result[key] = {"count": len(values), "mean": float(values.mean()),
                       "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    return result


def write_csv(path, rows):
    keys = sorted({k for row in rows for k in row})
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_example(path, rgb, result, gt_depth, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    axs[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)); axs[0].set_title("Raw image")
    yy, xx = np.where(mask)
    y0, y1 = max(0, yy.min()-15), min(mask.shape[0], yy.max()+16)
    x0, x1 = max(0, xx.min()-15), min(mask.shape[1], xx.max()+16)
    axs[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="yellow", linewidth=1))
    crop = np.s_[y0:y1, x0:x1]
    low, high = float(gt_depth[mask].min()), float(gt_depth[mask].max())
    for ax, a, title in zip(axs[1:3], (gt_depth, result["depth"]), ("GT depth (mm)", "Predicted depth (mm)")):
        im = ax.imshow(np.where(mask, a, np.nan)[crop], vmin=low, vmax=high, cmap="viridis")
        ax.set_title(title + "\nContact crop"); fig.colorbar(im, ax=ax, shrink=0.7)
    err = np.where(mask, np.abs(result["depth"]-gt_depth), np.nan)
    im = axs[3].imshow(err[crop], cmap="magma"); axs[3].set_title("Absolute error (mm)\nContact crop")
    fig.colorbar(im, ax=axs[3], shrink=0.7)
    for ax in axs: ax.axis("off")
    fig.savefig(path, dpi=130); plt.close(fig)


def _load_frame(split_dir, sid, use_nir, nir_scale, folder="images"):
    rgb = read_image(image_path(split_dir, sid, folder=folder))
    nir = read_image(image_path(split_dir, sid, "nir", folder), True, nir_scale) if use_nir else None
    return rgb, nir


def evaluate(args):
    cv2.setNumThreads(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    output = Path(args.output) if args.output else ROOT / "reports" / "geometry" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite a completed evaluation: {output}")
    legacy_config = load_config(args.legacy_config) if args.legacy_config else default_config()
    entries = []
    if args.experiments:
        entries = json.loads(Path(args.experiments).read_text(encoding="utf-8"))["experiments"]
    else:
        for version in args.versions:
            cfg_path = ROOT / "logs" / "gel_experiment" / f"version_{version}" / "geometry_config.json"
            cfg = load_config(cfg_path) if cfg_path.exists() else legacy_config
            methods = args.methods if cfg["route"] == "gradient" and args.methods else [cfg["interpolation"]["method"]]
            entries.extend({"version": version, "method": method, "source": "raw"} for method in methods)
            if args.legacy_reference and not cfg_path.exists():
                entries.append({"version": version, "source": "images_removed"})
    if not entries:
        raise ValueError("No experiments selected")
    configs, mask_audits = {}, {}
    for entry in entries:
        version = int(entry["version"])
        if version in configs:
            continue
        path = ROOT / "logs" / "gel_experiment" / f"version_{version}" / "geometry_config.json"
        saved = load_config(path) if path.exists() else legacy_config
        effective, provenance = training_mask_config(saved, args.detector_source, args.mask_source)
        configs[version] = (saved, effective)
        mask_audits[version] = provenance
    first = configs[int(entries[0]["version"])][1]
    args.record = args.record or first["data"]["record"]
    args.split = args.split or first["data"]["val_split"]
    args.solver_backend = args.solver_backend or first["depth"]["backend"]
    split_dir = ROOT / "data" / args.record / args.split
    candidates, missing = discover_ids(split_dir, ("normal_gt", "normal_complete_gt", "depth_gt", "mask"))
    if args.limit is not None:
        candidates = candidates[:args.limit]
    if not candidates:
        raise ValueError("No contact frames with complete GT")
    for version, (saved, effective) in configs.items():
        for key in ("height", "width"):
            if effective["input"][key] != first["input"][key]:
                raise ValueError("Comparison requires the same image resolution")
        if (effective["detector"] != first["detector"] and args.mask_source != "saved") or effective["interpolation"]["dilate_iterations"] != first["interpolation"]["dilate_iterations"]:
            raise ValueError("Comparison requires a common marker detector and dilation")
        if args.detector_source == "training":
            mask_audits[version]["mask_equality"] = verify_training_masks(
                saved, effective, split_dir, [0, *candidates], args.mask_source)
        print(f"[Mask parameters] v{version}: {effective['detector']}", flush=True)
    audit = audit_split_duplicates(ROOT / "data" / args.record / first["data"]["train_split"], split_dir)
    print(f"[Audit] exact train/evaluation duplicate rate: {audit['duplicate_rate']:.1%}", flush=True)
    report = {"dataset": str(split_dir), "split": args.split, "sample_ids": candidates,
              "missing_gt_frames": missing, "duplicate_audit": audit,
              "interpretation": "回放验证，不代表泛化能力" if audit["duplicate_images"] else "指定数据集评测",
              "hardware": {"platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                  "opencv": cv2.__version__, "device": args.device,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "cpu_threads": args.cpu_threads},
              "protocol": {"warmup": args.warmup, "timing_repeats": args.repeats,
                  "sample_limit": args.limit, "smoke_only": args.limit is not None,
                  "mask_source": args.mask_source, "detector_source": args.detector_source,
                  "mask_parameter_audit": mask_audits, "solver_backend": args.solver_backend,
                  "aggregation": "pixel-weighted across fixed regions; bias removed per frame per region only for aligned metric",
                  "normal_angle": "degrees in existing depth-per-pixel label coordinates",
                  "timing": "synchronized per-stage profiling is separate from uninterrupted end-to-end pass; I/O and metrics excluded",
                  "background_cost": "one-off initialization reported separately"}, "experiments": []}
    # Fixed reference regions are created once and shared by all methods, never from predicted validity.
    reference_cfg = first
    report["region_reference"] = {"detector": reference_cfg["detector"], "dilation": reference_cfg["interpolation"]["dilate_iterations"], "ring_pixels": 5}
    reference_bg = read_image(image_path(split_dir, 0))
    base_bg_mask, _ = resolve_mask(reference_bg, split_dir, 0, reference_cfg, args.mask_source)
    expanded_bg = expand_mask(base_bg_mask, reference_cfg["interpolation"]["dilate_iterations"])
    frame_data, invalid = {}, []
    for sid in candidates:
        t0 = time.perf_counter()
        try:
            raw = read_image(image_path(split_dir, sid))
            complete = np.load(split_dir / "normal_complete_gt" / f"normal_complete{sid}.npy").astype(np.float32)
            contact_normal = np.load(split_dir / "normal_gt" / f"normal{sid}.npy").astype(np.float32)
            depth = np.load(split_dir / "depth_gt" / f"depth{sid}.npy").astype(np.float32)
            contact = np.load(split_dir / "mask" / f"mask{sid}.npy") != 0
            if complete.shape != raw.shape or contact_normal.shape != raw.shape or depth.shape != raw.shape[:2] or contact.shape != depth.shape:
                raise ValueError("Image/GT shape mismatch")
            if not np.isfinite(complete).all() or not np.isfinite(depth).all() or not np.isfinite(contact_normal).all():
                raise ValueError("Non-finite ground truth")
            valid = np.linalg.norm(complete, axis=-1) > 1e-6
            if not contact.any() or not valid[contact].all() or not np.allclose(contact_normal[contact], complete[contact]):
                raise ValueError("Contact/normal GT mismatch")
            io_ms = (time.perf_counter()-t0)*1000
            base, source = resolve_mask(raw, split_dir, sid, reference_cfg, args.mask_source)
            marker = expand_mask(base, reference_cfg["interpolation"]["dilate_iterations"]) | expanded_bg
            ring = expand_mask(marker, 5) & ~marker
            frame_data[sid] = {"raw": raw, "normal": complete, "gradient": normals_to_gradients(complete),
                "depth": depth, "contact": contact, "valid": valid, "marker": marker, "ring": ring,
                "base_mask": base, "io_ms": io_ms, "mask_source": source}
        except Exception as exc:
            invalid.append({"id": sid, "error": str(exc)})
    report["invalid_gt_frames"] = invalid
    if invalid:
        save_json(output / "preflight_failure.json", report)
        raise ValueError("Invalid GT; fixed-set evaluation aborted (see preflight_failure.json)")
    common_prior_hash = None
    for entry_index, entry in enumerate(entries):
        version = int(entry["version"])
        cfg_path = ROOT / "logs" / "gel_experiment" / f"version_{version}" / "geometry_config.json"
        name = f"{entry_index:02d}_v{version}_{entry.get('source', 'raw')}_{entry.get('method', 'default')}"
        print(f"[Evaluate] {name}: {len(candidates)} frames", flush=True)
        pipeline = GeometryPipeline(version, args.device, config=None if cfg_path.exists() else legacy_config,
            method=entry.get("method"), mask_source=args.mask_source, prior=args.prior, solver_backend=args.solver_backend,
            detector_source=args.detector_source)
        source = entry.get("source", "raw")
        if source not in ("raw", "images_removed"):
            raise ValueError("Unknown experiment image source")
        legacy = source == "images_removed"
        if legacy and cfg_path.exists():
            raise ValueError("images_removed reference is reserved for legacy models")
        cfg = pipeline.config
        prior_hash = file_hash(pipeline.prior_path)
        if common_prior_hash is not None and prior_hash != common_prior_hash:
            raise ValueError("Models use different depth priors; supply a common --prior for comparison")
        common_prior_hash = prior_hash
        use_nir, scale = cfg["input"]["use_nir"], cfg["input"]["nir_scale"]
        bg, nir_bg = _load_frame(split_dir, 0, use_nir, scale, "images_removed" if legacy else "images")
        bg_mask = resolve_mask(reference_bg, split_dir, 0, cfg, args.mask_source)[0] if legacy else None
        pipeline.set_background(bg, nir_bg, split_dir, preprocessed=legacy, marker_mask=bg_mask)
        frames, failures = {}, []
        for sid in candidates:
            try:
                t0 = time.perf_counter()
                rgb, nir = _load_frame(split_dir, sid, use_nir, scale, "images_removed" if legacy else "images")
                frames[sid] = (rgb, nir, (time.perf_counter()-t0)*1000)
            except Exception as exc:
                failures.append({"id": sid, "status": "failed", "error": str(exc)})
        def run(sid, profile=False):
            rgb, nir, _ = frames[sid]
            return pipeline.predict_raw(rgb, nir, sample_id=sid, split_dir=split_dir, profile=profile,
                preprocessed=legacy, marker_mask=frame_data[sid]["base_mask"] if legacy else None)
        if not frames:
            raise ValueError(f"No usable inputs for {name}")
        warm_id = next(iter(frames))
        try:
            for _ in range(args.warmup): run(warm_id)
        except Exception as exc:
            save_json(output / f"{name}_warmup_failure.json", {"id": warm_id, "error": str(exc)})
            raise
        rows, stage_times, e2e_times, time_rows = [], [], [], []
        reference_errors = []
        example_ids = {candidates[0], candidates[len(candidates)//2], candidates[-1]}
        for index, sid in enumerate(candidates):
            if sid not in frames: continue
            data = frame_data[sid]
            try:
                result = run(sid, profile=True)
                t0 = time.perf_counter()
                valid = data["valid"] & np.isnan(pipeline.prior)
                contact = data["contact"] & valid
                regions = {"contact": contact, "marker": contact & data["marker"],
                    "ring": contact & data["ring"], "other": contact & ~data["marker"] & ~data["ring"], "full": valid}
                if not contact.any(): raise ValueError("Empty contact/solver intersection")
                metrics = {k: region_metrics(result, data["gradient"], data["depth"], data["normal"], m) for k, m in regions.items()}
                gt_rebuilt = reconstruct_from_gradients(data["gradient"], pipeline.prior,
                    cfg["depth"]["backend"], args.device)
                ref_d = gt_rebuilt[contact].astype(np.float64) - data["depth"][contact]
                ref = {"id": sid, "pixels": int(contact.sum()), "mae": float(np.abs(ref_d).mean()), "rmse": float(np.sqrt((ref_d**2).mean()))}
                reference_errors.append(ref)
                row = {"id": sid, "status": "ok", "regions": metrics, "completion": result["completion"],
                       "clamped_normal_pixels": result["clamped_normal_pixels"],
                       "mask_source": result["mask_source"], "metrics_ms": (time.perf_counter()-t0)*1000,
                       "input_io_ms": frames[sid][2], "gt_io_ms": data["io_ms"]}
                if sid in example_ids:
                    ts = time.perf_counter()
                    save_example(output / f"{name}_sample{sid}.png", data["raw"], result, data["depth"], contact)
                    row["visualization_save_ms"] = (time.perf_counter()-ts)*1000
                rows.append(row)
                stage_times.append(result["timings"])
                time_rows.append({"id": sid, "pass": "profile", **result["timings"]})
                for repeat in range(args.repeats):
                    timed = run(sid)
                    e2e_times.append(timed["timings"])
                    time_rows.append({"id": sid, "pass": "end_to_end", "repeat": repeat, **timed["timings"]})
            except Exception as exc:
                failures.append({"id": sid, "status": "failed", "error": str(exc)})
            if (index+1) % 15 == 0 or index+1 == len(candidates):
                print(f"[Evaluate] {name}: {index+1}/{len(candidates)}, failures={len(failures)}", flush=True)
        expected = len(candidates)
        success_ids = {r["id"] for r in rows} - {f["id"] for f in failures}
        rows = [r for r in rows if r["id"] in success_ids]
        summary = {"name": name, "version": version, "source": source, "effective_config": cfg,
            "completion_implementation": ("opencv_float32_per_component_unscaled" if cfg["route"] == "gradient" and cfg["interpolation"]["method"] in ("ns", "telea") else "standard"),
            "numerical_notes": (["梯度 Telea 直接对各 float32 分量调用 OpenCV，不缩放、不量化、不裁剪；当前 OpenCV 常量小梯度补洞测试出现明显扰动，属于数值诊断实验。"] if cfg["route"] == "gradient" and cfg["interpolation"]["method"] == "telea" else []),
            "mask_parameter_audit": mask_audits[version],
            "overrides": pipeline.overrides, "checkpoint": pipeline.model.checkpoint_path,
            "verification_only": cfg["training"].get("verification_only", False),
            "clamped_normal_pixels": sum(r["clamped_normal_pixels"] for r in rows),
            "weight_files": {p.name: file_hash(p) for p in (pipeline.version_dir / "checkpoints").glob("*.ckpt")},
            "prior": {"path": pipeline.prior_path, "sha256": file_hash(pipeline.prior_path)},
            "expected_frames": expected, "successful_frames": len(rows), "failed_frames": failures,
            "comparable": len(rows) == expected, "mask_source": args.mask_source,
            "timing_interpretation": "legacy preprocessed replay (repair cost excluded)" if legacy else
                ("online automatic detection" if args.mask_source == "auto" else "assisted/cached mask pipeline"),
            "background_setup_ms": pipeline.background_setup_ms,
            "regions": aggregate_regions(rows) if len(rows) == expected else None,
            "partial_regions_diagnostic_only": aggregate_regions(rows) if len(rows) != expected else None,
            "stage_times_ms": timing_summary(stage_times), "end_to_end_ms": timing_summary(e2e_times),
            "io_and_metrics_ms": timing_summary([{k: r[k] for k in ("input_io_ms", "gt_io_ms", "metrics_ms")} for r in rows]),
            "gt_normal_depth_reference": reference_errors,
            "fallback_pixels": sum(r["completion"]["fallback_pixels"] for r in rows),
            "completed_pixels": sum(r["completion"]["masked_pixels"] for r in rows)}
        summary["fallback_fraction"] = summary["fallback_pixels"] / max(1, summary["completed_pixels"])
        flat = []
        for row in rows:
            flat.append({"id": row["id"], "status": row["status"], "mask_source": row["mask_source"],
                "clamped_normal_pixels": row["clamped_normal_pixels"],
                **row["completion"], **{f"{region}_{k}": v for region, met in row["regions"].items() for k, v in met.items()},
                **{k: row[k] for k in ("input_io_ms", "gt_io_ms", "metrics_ms")}})
        flat.extend(failures)
        write_csv(output / f"{name}_frames.csv", flat)
        write_csv(output / f"{name}_timings.csv", time_rows)
        save_json(output / f"{name}_details.json", rows)
        report["experiments"].append(summary)
        save_json(output / "summary.json", report)
        del pipeline
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    lines = ["# 几何补全回放评测", "", report["interpretation"], "",
        "**小规模流程检查，不能作为完整评测结论。**" if args.limit is not None else "完整共同样本评测。",
        f"数据：{args.record}/{args.split}；共同样本数：{len(candidates)}。",
        f"训练/评测原图重复率：{audit['duplicate_rate']:.1%}。缺少完整几何真值的图像数：{len(missing)}。",
        "梯度单位 mm/pixel，深度单位 mm；主误差按固定接触区域像素加权。",
        "去偏置指标分别在每帧每个区域内减去误差均值，不能代替绝对深度误差。",
        "GT 法向积分参考误差用于检查边界及数值链，不属于模型误差。",
        "阶段分析包含同步开销；端到端时间来自另外的无逐阶段同步测量。背景设置、文件读写、指标计算和绘图不包含在计算端到端耗时中。", ""]
    for ex in report["experiments"]:
        lines.extend([f"## {ex['name']}", "", f"成功 {ex['successful_frames']}/{ex['expected_frames']}；计时口径：{ex['timing_interpretation']}。"])
        lines.extend(ex["numerical_notes"])
        lines.append(f"检测参数来源：{args.detector_source}；实际检测参数：`{json.dumps(ex['effective_config']['detector'], ensure_ascii=False)}`。")
        equality = ex["mask_parameter_audit"].get("mask_equality", {})
        lines.append(f"与训练掩膜一致性已核验 {len(equality.get('verified_sample_ids_including_background', []))} 帧（含背景）；无历史记录 {len(equality.get('unrecorded_sample_ids', []))} 帧。")
        if ex["regions"]:
            m = ex["regions"]["contact"]
            lines.append(f"接触区梯度 MAE：{m['gradient_mae']:.8f}；深度 MAE/RMSE：{m['depth_mae']:.6f}/{m['depth_rmse']:.6f}；去偏置深度 MAE：{m['bias_aligned_depth_mae']:.6f}。")
        else:
            lines.append("存在失败，取消可比较的总分；局部成功结果仅保留为诊断。")
        t = ex["end_to_end_ms"].get("end_to_end_ms")
        if t: lines.append(f"端到端均值/中位数/P95：{t['mean']:.3f}/{t['median']:.3f}/{t['p95']:.3f} ms。")
        lines.append(f"插值回退比例：{ex['fallback_fraction']:.3%}。\n")
        if ex["verification_only"]:
            lines.append("**短训练流程验证版本，不能据此判断方法优劣。**")
        lines.append(f"法向深度分量触发下限保护的像素累计数：{ex['clamped_normal_pixels']}；未从评价区域剔除。")
        if ex["regions"]:
            lines.extend(["", "| 区域 | 像素数 | 梯度 MAE / RMSE | 深度 MAE / RMSE (mm) | 去偏置深度 MAE / RMSE (mm) |",
                          "| --- | ---: | ---: | ---: | ---: |"])
            for key, label in (("contact", "接触区域"), ("marker", "接触内 marker 污染区域"), ("ring", "接触内 5 像素邻域"), ("other", "接触内其余区域"), ("full", "求解域有效 GT")):
                m = ex["regions"][key]
                if not m["pixels"]: continue
                lines.append(f"| {label} | {m['pixels']} | {m['gradient_mae']:.8f} / {m['gradient_rmse']:.8f} | {m['depth_mae']:.6f} / {m['depth_rmse']:.6f} | {m['bias_aligned_depth_mae']:.6f} / {m['bias_aligned_depth_rmse']:.6f} |")
        lines.extend(["", "| 分阶段耗时（ms） | 均值 | 中位数 | P95 |", "| --- | ---: | ---: | ---: |"])
        labels = {"detection_ms": "检测及掩膜膨胀", "image_completion_ms": "图像修复", "input_transfer_ms": "输入转换与传输", "output_transfer_ms": "输出传输", "network_ms": "网络", "gradient_conversion_ms": "梯度转换", "gradient_completion_ms": "梯度补全", "poisson_ms": "Poisson", "normal_output_ms": "法向输出转换", "profile_total_ms": "带阶段同步总时长"}
        for key, t in ex["stage_times_ms"].items():
            lines.append(f"| {labels.get(key, key)} | {t['mean']:.3f} | {t['median']:.3f} | {t['p95']:.3f} |")
        lines.extend(["", f"背景初始化：{ex['background_setup_ms']:.3f} ms。", ""])
        for key, t in ex["io_and_metrics_ms"].items():
            lines.append(f"{key} 均值/中位数/P95：{t['mean']:.3f}/{t['median']:.3f}/{t['p95']:.3f} ms。")
        ref = ex["gt_normal_depth_reference"]
        if ref:
            n = sum(r["pixels"] for r in ref)
            mae = sum(r["mae"]*r["pixels"] for r in ref)/n
            rmse = np.sqrt(sum(r["rmse"]**2*r["pixels"] for r in ref)/n)
            lines.append(f"完整 GT 法向积分参考深度 MAE/RMSE：{mae:.6f}/{rmse:.6f} mm。\n")
        preview = f"{ex['name']}_sample{candidates[len(candidates)//2]}.png"
        if (output / preview).exists():
            lines.append(f"![接触区域误差放大图]({preview})\n")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] {output / 'report.md'}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--versions", nargs="+", type=int, default=[118])
    p.add_argument("--methods", nargs="+", choices=METHODS["gradient"])
    p.add_argument("--experiments", help="JSON with experiments: [{version, method?, source?}]")
    p.add_argument("--record", help="Default: first model's training record")
    p.add_argument("--split", help="Default: first model's validation split")
    p.add_argument("--legacy-reference", action="store_true")
    p.add_argument("--legacy-config")
    p.add_argument("--mask-source", choices=("auto", "saved", "prefer_saved"), default="auto")
    p.add_argument("--detector-source", choices=("training", "model"), default="training",
                   help="Recover actual training-mask detector (default); model explicitly replays the old default detector")
    p.add_argument("--prior")
    p.add_argument("--solver-backend", choices=("cpu", "gpu"), help="Default: first model's saved backend")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--limit", type=int, help="Smoke verification, not full dataset evaluation")
    p.add_argument("--output")
    a = p.parse_args()
    if a.warmup < 0 or a.repeats < 1 or a.cpu_threads < 1 or (a.limit is not None and a.limit < 1):
        p.error("Invalid warmup/repeats/threads/limit")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; explicitly select --device cpu")
    report = evaluate(a)
    if any(ex["failed_frames"] for ex in report["experiments"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
