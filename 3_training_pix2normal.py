import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import L1Loss
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import shutil
import argparse

# 设置浮点数精度
torch.set_float32_matmul_precision('high')
# Training/input settings are versioned in GEOMETRY_CONFIG below.
input_log_interval = 50  # 每隔多少 step 记录一次输入范围到 TensorBoard

from geometry_processing import default_config, load_config, validate_config, save_json, file_hash, resolve_prior
from geometry_dataset import PreparedGeometryDataset
import copy
from pathlib import Path
import model as model_definitions

# Edit this object for interactive/script use; CLI arguments explicitly override it.
GEOMETRY_CONFIG = default_config()


class Lit_Model(pl.LightningModule):
    def __init__(self, use_pos=True, geometry_config=None):
        super().__init__()
        self.geometry_config = validate_config(copy.deepcopy(geometry_config or GEOMETRY_CONFIG))
        self.geometry_config["input"]["use_pos"] = use_pos
        self.save_hyperparameters({"use_pos": use_pos, "geometry_config": self.geometry_config})


        # Windows not yet supported for torch.compile
        # self.basenet =torch.compile(Modelv2(use_pos=use_pos))


        self.basenet = getattr(model_definitions, self.geometry_config["network"]["class"])(
            use_pos=use_pos, nir_training=self.geometry_config["input"]["use_nir"],
            **self.geometry_config["network"]["kwargs"])  # 原始网络
        # self.basenet = Model_quan_l4_32(use_pos=use_pos, nir_training=nir_training)  # 用于量化的网络
        # self.basenet = Model_quan_l3_32(use_pos=use_pos, nir_training=nir_training)

        # self.basenet = Model_monitor_noprint(use_pos=use_pos, nir_training=nir_training)
        # self.metric = my_Metric()
        self.loss = L1Loss()
        # self.loss = CosineSimilarity()
        self.learning_rate = self.geometry_config["training"]["learning_rate"]
        # best-records for px/py style loss (nx/nz, ny/nz)
        self.best_loss_px = float('inf')
        self.best_loss_py = float('inf')
        # self.best_loss = float('inf')
        self.best_gradient_error = float('inf')


    def on_train_start(self) -> None:
        # if self.local_rank == 0 and self.current_epoch == 0:
        #     path_backup = '{}/{}'.format(self.logger.log_dir, "code")
        #     os.makedirs(path_backup, exist_ok=True)
        #     backup_source_code(path_backup)
        return
        # # 记录模型结构（仅在第一次训练时记录）
        # if self.logger is not None:
        #     with torch.no_grad():
        #         example_input = torch.randn(1, 9, 224, 224).to(self.device)
        #         self.logger.experiment.add_graph(self.basenet, example_input)

    def training_step(self, sample, batch_idx):
        # Forward through the network
        if batch_idx % input_log_interval == 0:
            rgb_diff = sample["rgb"] - sample["bg_rgb"]
            self.log("input/rgb_min", rgb_diff.min(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
            self.log("input/rgb_max", rgb_diff.max(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
            self.log("input/rgb_mean", rgb_diff.mean(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
            self.log("input/rgb_std", rgb_diff.std(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)

            if "nir" in sample and "bg_nir" in sample:
                nir_diff = sample["nir"] - sample["bg_nir"]
                self.log("input/nir_min", nir_diff.min(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
                self.log("input/nir_max", nir_diff.max(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
                self.log("input/nir_mean", nir_diff.mean(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
                self.log("input/nir_std", nir_diff.std(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)

            if "position" in sample:
                pos = sample["position"]
                self.log("input/pos_min", pos.min(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
                self.log("input/pos_max", pos.max(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
                self.log("input/pos_mean", pos.mean(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
                self.log("input/pos_std", pos.std(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)

        normal = self.basenet(sample)
        normal_norm = F.normalize(normal, p=2, dim=1)
        gt_norm = F.normalize(sample['gt'], p=2, dim=1)

        loss_nx = self.loss(normal_norm[:, 0, ...], gt_norm[:, 0, ...])
        loss_ny = self.loss(normal_norm[:, 1, ...], gt_norm[:, 1, ...])
        loss_nz = self.loss(normal_norm[:, 2, ...], gt_norm[:, 2, ...])
        
        # New formulation: use px = nx/nz and py = ny/nz as targets
        eps = self.geometry_config["depth"]["eps"]
        nx = normal_norm[:, 0, ...]
        ny = normal_norm[:, 1, ...]
        nz = normal_norm[:, 2, ...]
        px_pred = -nx / nz.clamp_min(eps)
        py_pred = ny / nz.clamp_min(eps)

        gx = gt_norm[:, 0, ...]
        gy = gt_norm[:, 1, ...]
        gz = gt_norm[:, 2, ...]
        px_gt = -gx / gz.clamp_min(eps)
        py_gt = gy / gz.clamp_min(eps)

        loss_px = self.loss(px_pred, px_gt)
        loss_py = self.loss(py_pred, py_gt)
        gradient_error = loss_px + loss_py

        # Use direct per-component normal regression as main loss (no cosine term)
        loss = loss_nx + loss_ny + loss_nz

        # 记录训练损失
        self.log("gradient_error_train", gradient_error, on_step=True, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_px_train", loss_px, on_step=True, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_py_train", loss_py, on_step=True, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_nx_train", loss_nx, on_step=True, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_ny_train", loss_ny, on_step=True, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_nz_train", loss_nz, on_step=True, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_train", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, sample, batch_idx):
        # Forward through the network
        normal = self.basenet(sample)
        normal_norm = F.normalize(normal, p=2, dim=1)
        gt_norm = F.normalize(sample['gt'], p=2, dim=1)

        loss_nx = self.loss(normal_norm[:, 0, ...], gt_norm[:, 0, ...])
        loss_ny = self.loss(normal_norm[:, 1, ...], gt_norm[:, 1, ...])
        loss_nz = self.loss(normal_norm[:, 2, ...], gt_norm[:, 2, ...])

        # New px/py loss computation
        eps = self.geometry_config["depth"]["eps"]
        nx = normal_norm[:, 0, ...]
        ny = normal_norm[:, 1, ...]
        nz = normal_norm[:, 2, ...]
        px_pred = -nx / nz.clamp_min(eps)
        py_pred = ny / nz.clamp_min(eps)

        gx = gt_norm[:, 0, ...]
        gy = gt_norm[:, 1, ...]
        gz = gt_norm[:, 2, ...]
        px_gt = -gx / gz.clamp_min(eps)
        py_gt = gy / gz.clamp_min(eps)

        loss_px = self.loss(px_pred, px_gt)
        loss_py = self.loss(py_pred, py_gt)
        gradient_error = loss_px + loss_py

        # Use direct per-component normal regression as main loss (no cosine term)
        loss = loss_nx + loss_ny + loss_nz

        # 记录验证损失
        self.log("gradient_error_val", gradient_error, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_px_val", loss_px, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_py_val", loss_py, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("loss_val", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Update best metrics using epoch-aggregated losses so checkpointing aligns with logs."""
        try:
            metrics = self.trainer.callback_metrics

            loss_px_val = metrics.get("loss_px_val")
            loss_py_val = metrics.get("loss_py_val")
            gradient_error_val = metrics.get("gradient_error_val")


            # Convert to floats if present

            loss_px_f = float(loss_px_val) if loss_px_val is not None else None
            loss_py_f = float(loss_py_val) if loss_py_val is not None else None
            gradient_error_f = float(gradient_error_val) if gradient_error_val is not None else None

            # Update bests
            if loss_px_f is not None and loss_px_f < self.best_loss_px:
                self.best_loss_px = loss_px_f
            if loss_py_f is not None and loss_py_f < self.best_loss_py:
                self.best_loss_py = loss_py_f
            if gradient_error_f is not None and gradient_error_f < self.best_gradient_error:
                self.best_gradient_error = gradient_error_f


            # log bests (epoch-level) so they can be monitored by ModelCheckpoint
            self.log("best_loss_px", self.best_loss_px, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
            self.log("best_loss_py", self.best_loss_py, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
            self.log("best_gradient_error", self.best_gradient_error, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

            print(
                f"[BestUpdate][epoch] best_loss_px={self.best_loss_px:.6e}, "
                f"best_loss_py={self.best_loss_py:.6e}, "
                f"best_gradient_error={self.best_gradient_error:.6e}"
            )
        except Exception:
            pass

    # def test_step(self, sample, batch_idx):
    #     # Forward through the network
    #     rgb, bg, gt = sample['rgb'], sample['bg'], sample['gt']
    #     # self.metric(gt, normal)
    #     #
    #     return 0

    # def on_train_epoch_end(self,) -> None:
    #     rmse, mae = self.metric.compute()
    #     self.log('RMSE', rmse, sync_dist=True)
    #     self.log('MAE', mae, sync_dist=True)
    #     self.metric.reset()
    #
    # def on_validation_epoch_end(self, ) -> None:
    #     rmse, mae, mae_n = self.metric.compute()
    #     self.log('RMSE', rmse, sync_dist=True)
    #     self.log('MAE', mae, sync_dist=True)
    #     self.log('MAE_normal', mae_n, sync_dist=True)
    #     self.metric.reset()

    def configure_optimizers(self):
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in self.basenet.modules():
            for p_name, p in v.named_parameters(recurse=False):
                if p_name == 'bias':  # bias (no decay)
                    g[2].append(p)
                elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                    g[1].append(p)
                else:
                    g[0].append(p)  # weight (with decay)

        optimizer = torch.optim.Adam(g[2], lr=self.learning_rate)  # adjust beta1 to momentum
        optimizer.add_param_group(
            {'params': g[0], 'weight_decay': self.geometry_config['training']['weight_decay']})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        
        # 使用余弦退火学习率调度器替代原来的硬阶梯衰减
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.geometry_config["training"]["max_epochs"],
            eta_min=self.geometry_config["training"].get("scheduler_eta_min", 1e-5)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def cli_main():
    parser = argparse.ArgumentParser(description="Train a versioned image/gradient geometry pipeline")
    parser.add_argument("--config", help="Complete geometry JSON configuration")
    parser.add_argument("--route", choices=("image", "gradient"))
    parser.add_argument("--method", choices=("telea", "ns", "nearest", "linear", "cubic"))
    parser.add_argument("--record")
    parser.add_argument("--prior")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit-samples", type=int, help="Short verification only; saved in manifest")
    parser.add_argument("--result-json", help="Write an explicit success receipt for batch orchestration")
    args = parser.parse_args()
    receipt_path = Path(args.result_json).resolve() if args.result_json else None
    if receipt_path is not None and receipt_path.exists():
        raise FileExistsError(f"Training receipt already exists: {receipt_path}")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    config = load_config(args.config) if args.config else copy.deepcopy(GEOMETRY_CONFIG)
    if args.route:
        if args.route != config["route"]:
            config["interpolation"]["method"] = "telea" if args.route == "image" else "nearest"
        config["route"] = args.route
        config["training"]["pixel_policy"] = "gt_valid" if args.route == "image" else "exclude_current_background_union"
    if args.method: config["interpolation"]["method"] = args.method
    if args.record: config["data"]["record"] = args.record
    if args.prior: config["depth"]["prior"] = str(Path(args.prior).resolve())
    if args.epochs is not None: config["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None: config["training"]["batch_size"] = args.batch_size
    validate_config(config)
    if config["training"]["max_epochs"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu explicitly")
    pl.seed_everything(config["training"]["seed"], workers=True)
    data_root = root / "data" / config["data"]["record"]
    from geometry_processing import discover_ids, audit_split_duplicates
    datasets = {}
    for name, mode in (("train", "train"), ("val", "eval")):
        split = data_root / config["data"][name + "_split"]
        ids, _ = discover_ids(split, use_nir=config["input"]["use_nir"])
        if args.limit_samples is not None:
            if args.limit_samples < 1: raise ValueError("limit-samples must be positive")
            ids = ids[:args.limit_samples]
        datasets[name] = PreparedGeometryDataset(split, config, mode, sample_ids=ids)
        audit = datasets[name].audit
        print(f"[Data] {name}: {len(datasets[name])} samples; excluded={len(audit['excluded'])}, "
              f"missing_GT={len(audit['missing'])}, cache_hits={audit['cache_hits']}, cache_misses={audit['cache_misses']}", flush=True)
        for exclusion in audit["excluded"]:
            print(f"[Data] excluded {name}/{exclusion['id']}: {exclusion['reason']}", flush=True)
    config["data"]["sample_ids"] = {k: v.sample_ids for k, v in datasets.items()}
    config["data"]["preparation_audit"] = {k: v.audit for k, v in datasets.items()}
    config["data"]["cache_manifest"] = {k: v.preparation_manifest() for k, v in datasets.items()}
    config["data"]["duplicate_audit"] = audit_split_duplicates(
        data_root / config["data"]["train_split"], data_root / config["data"]["val_split"])
    if config["data"]["duplicate_audit"]["duplicate_rate"]:
        print("[Data] WARNING: train/val overlap; results are replay validation, not generalization.")
    logger = TensorBoardLogger(save_dir=str(root / "logs"), name="gel_experiment")
    version_dir = Path(logger.log_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    prior = resolve_prior(config)
    shutil.copy2(prior, version_dir / "depth_prior.npy")
    config["depth"].update(prior="depth_prior.npy", prior_sha256=file_hash(version_dir / "depth_prior.npy"))
    config["training"]["verification_only"] = args.limit_samples is not None or config["training"]["max_epochs"] <= 2
    save_json(version_dir / "geometry_config.json", config)
    for name in ("3_training_pix2normal.py", "dataset.py", "model.py", "geometry_processing.py", "geometry_dataset.py", "geometry_pipeline.py", "depth_reconstruction_core.py"):
        shutil.copy2(root / name, version_dir / name)
    loaders = {k: DataLoader(v, batch_size=config["training"]["batch_size"], shuffle=k == "train",
        num_workers=args.workers, persistent_workers=args.workers > 0) for k, v in datasets.items()}
    model = Lit_Model(config["input"]["use_pos"], config)
    checkpoint = ModelCheckpoint(monitor="gradient_error_val", mode="min",
        filename="{epoch}-{gradient_error_val:.6f}", save_top_k=1, save_last=False)
    trainer = pl.Trainer(accelerator="gpu" if args.device == "cuda" else "cpu", devices=1,
        max_epochs=config["training"]["max_epochs"], precision=32, logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="epoch"), checkpoint,
            pl.callbacks.EarlyStopping(monitor="gradient_error_val",
                min_delta=config["training"].get("early_stopping_min_delta", 1e-5),
                patience=config["training"].get("early_stopping_patience", 40), mode="min")],
        log_every_n_steps=1, enable_progress_bar=False)
    trainer.fit(model, loaders["train"], loaders["val"])
    best = Lit_Model.load_from_checkpoint(checkpoint.best_model_path)
    torch.save(best.state_dict(), version_dir / "float_model_best.pth")
    save_json(version_dir / "training_result.json", {"best_checkpoint": str(Path(checkpoint.best_model_path).relative_to(version_dir)),
        "best_gradient_error": float(checkpoint.best_model_score), "verification_only": config["training"]["verification_only"]})
    if receipt_path is not None:
        receipt = {"version": int(version_dir.name.removeprefix("version_")),
            "version_dir": str(version_dir.resolve()), "checkpoint": str(Path(checkpoint.best_model_path).resolve()),
            "checkpoint_sha256": file_hash(checkpoint.best_model_path),
            "config_sha256": file_hash(version_dir / "geometry_config.json"),
            "requested_config": str(Path(args.config).resolve()) if args.config else None,
            "verification_only": config["training"]["verification_only"]}
        pending = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        save_json(pending, receipt)
        pending.replace(receipt_path)
    print(f"[Train] Saved complete geometry version: {version_dir}")


if __name__ == '__main__':
    cli_main()
