# 适用于曲面视触觉传感器力估计
import os
import torch
import pytorch_lightning as pl
from dataset import Dataset_gel_force_dynamic_weight
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import MSELoss, L1Loss
from model import HyperForceNet  # 使用新模型
import shutil
import json
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Callback

torch.set_float32_matmul_precision('high')

# Hyperparameters
learning_rate = 1e-4
batch_size = 8   # [DEBUG] 降到最低，排除显存问题
max_epochs = 900
weight_decay = 1e-3  # 加一点正则化防止过拟合
num_workers = min(4, os.cpu_count() or 1)
# record = "20251123-095400"  # 平面三指1号
# record = "20251205_173723"  # 曲面三指1号
# record = "20260114_120750"  # 多光谱三指3号   球形压头
# record = "20260306_193600"  # 多光谱三指3号 圆弧形压头
# record = "20260310_151840"  # 多光谱三指3号   小球压头
# record = "20260328_121407"  # 内窥镜三指1号 
record = "20260404_103156"  # 曲面三指2号

# Dataset config
pixmm = 0.034 # 曲面三指
# pixmm = 0.018   # 多光谱三指3号
use_nir = False               # 深度图是否使用加 NIR 的结果
# lag_dz = True                # True: 拉格朗日 dz；False: 简单深度差
lag_dz = False                # True: 拉格朗日 dz；False: 简单深度差
pos_dim = 3                  # 位置编码维度，2 或 3
depth_source = 'pred'          # 'gt' 或 'pred'
normalize_disp = False       # 是否对 dx/dy/dz 做训练集统计量归一化
convert_dz_to_pixel = False   # 是否先将 dz 从渲染坐标系旋到像素坐标系
convert_disp_to_sensor = False  # 是否再将三维位移整体旋到传感器测量坐标系
use_moment_loss = False      # 是否联合优化合力矩 (Mx, My)
moment_arm_mm = 42.3        # 力臂长度，单位 mm
moment_loss_weight = 1.0    # 力矩损失权重
moment_loss_force_equiv = True  # 是否先除以力臂，将力矩误差换算成等效力误差



class Lit_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 输出 3 维力场 (fx, fy, fz)
        self.model = HyperForceNet(kernel_size=5, stride=4, pos_dim=pos_dim)  # 根据输入特征调整 pos_dim
        
        # 暂时禁用 torch.compile，可能导致部分环境闪退
        # try:
        #     self.model = torch.compile(self.model)
        # except Exception as e:
        #     print(f"Warning: torch.compile failed: {e}")
            
        self.loss = L1Loss()
        self.use_moment_loss = use_moment_loss
        self.moment_arm_m = float(moment_arm_mm) * 1e-3
        self.moment_loss_weight = float(moment_loss_weight)
        self.moment_loss_force_equiv = bool(moment_loss_force_equiv)

    def _predict_force_and_moment(self, sample):
        pred_map = self.model(sample)          # [B,3,H,W]
        pred_force = torch.sum(pred_map, dim=(2, 3))  # [B,3]
        pred_moment = torch.stack([
            -pred_force[:, 1] * self.moment_arm_m,
            pred_force[:, 0] * self.moment_arm_m,
        ], dim=1)  # [B,2], 对应 [Mx, My]
        return pred_map, pred_force, pred_moment

    def training_step(self, sample, batch_idx):
        dz = sample['dz']
        dx = sample.get('dx')
        dy = sample.get('dy')

        # 减少日志频率，避免每个 step 都计算统计量造成 CPU-GPU 同步阻塞
        if batch_idx % 50 == 0:
            # 记录 dz (mm) 的范围
            self.log("input/dz_max", dz.max(), on_step=True, on_epoch=False)
            self.log("input/dz_min", dz.min(), on_step=True, on_epoch=False)
            self.log("input/dz_mean", dz.mean(), on_step=True, on_epoch=False)

            # 记录 DX (mm) 的范围
            if dx is not None:
                self.log("input/dx_max", dx.max(), on_step=True, on_epoch=False)
                self.log("input/dx_min", dx.min(), on_step=True, on_epoch=False)
                self.log("input/dx_abs_mean", dx.abs().mean(), on_step=True, on_epoch=False)  # 关注绝对值大小

            # 记录 DY (mm) 的范围
            if dy is not None:
                self.log("input/dy_max", dy.max(), on_step=True, on_epoch=False)
                self.log("input/dy_min", dy.min(), on_step=True, on_epoch=False)
                self.log("input/dy_abs_mean", dy.abs().mean(), on_step=True, on_epoch=False)  # 关注绝对值大小

        pred_map, pred_force, pred_moment = self._predict_force_and_moment(sample)
        target_force = sample['force']                   # [B,3]
        target_moment = sample.get('torque', None)

        force_loss = self.loss(pred_force, target_force)
        loss = force_loss
        moment_loss = None
        if self.use_moment_loss and target_moment is not None:
            target_moment_xy = target_moment[:, :2]
            if self.moment_loss_force_equiv:
                scale = 1.0 / max(self.moment_arm_m, 1e-8)
                moment_loss = self.loss(pred_moment * scale, target_moment_xy * scale)
            else:
                moment_loss = self.loss(pred_moment, target_moment_xy)
            loss = loss + self.moment_loss_weight * moment_loss

        self.log("train/loss", loss, prog_bar=True)  # 去掉 sync_dist=True (单卡训练不需要)
        self.log("train/loss_force", force_loss, prog_bar=False)
        if moment_loss is not None:
            self.log("train/loss_moment", moment_loss, prog_bar=False)

        # 分量 MAE：Fx, Fy, Fz
        with torch.no_grad():
            mae_per_comp = torch.mean(torch.abs(pred_force - target_force), dim=0)  # [3]
            self.log("train/mae_fx", mae_per_comp[0], prog_bar=False)
            self.log("train/mae_fy", mae_per_comp[1], prog_bar=False)
            self.log("train/mae_fz", mae_per_comp[2], prog_bar=False)
            if target_moment is not None:
                target_moment_xy = target_moment[:, :2]
                mae_moment = torch.mean(torch.abs(pred_moment - target_moment_xy), dim=0)  # [2]
                self.log("train/mae_mx", mae_moment[0], prog_bar=False)
                self.log("train/mae_my", mae_moment[1], prog_bar=False)

        if batch_idx % 50 == 0:
            self.log("train/pred_max", pred_force.abs().max(), on_step=True, prog_bar=False)
            self.log("train/target_max", target_force.abs().max(), on_step=True, prog_bar=False)
        return loss

    def validation_step(self, sample, batch_idx):
        _, pred_force, pred_moment = self._predict_force_and_moment(sample)
        target_force = sample['force']
        target_moment = sample.get('torque', None)
        force_loss = self.loss(pred_force, target_force)
        loss = force_loss
        moment_loss = None
        if self.use_moment_loss and target_moment is not None:
            target_moment_xy = target_moment[:, :2]
            if self.moment_loss_force_equiv:
                scale = 1.0 / max(self.moment_arm_m, 1e-8)
                moment_loss = self.loss(pred_moment * scale, target_moment_xy * scale)
            else:
                moment_loss = self.loss(pred_moment, target_moment_xy)
            loss = loss + self.moment_loss_weight * moment_loss

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/loss_force", force_loss, prog_bar=False)
        if moment_loss is not None:
            self.log("val/loss_moment", moment_loss, prog_bar=False)

        # 分量 MAE：Fx, Fy, Fz（按 epoch 聚合）
        mae_per_comp = torch.mean(torch.abs(pred_force - target_force), dim=0)  # [3]
        self.log("val/mae_fx", mae_per_comp[0], prog_bar=True)
        self.log("val/mae_fy", mae_per_comp[1], prog_bar=True)
        self.log("val/mae_fz", mae_per_comp[2], prog_bar=True)
        if target_moment is not None:
            target_moment_xy = target_moment[:, :2]
            mae_moment = torch.mean(torch.abs(pred_moment - target_moment_xy), dim=0)  # [2]
            self.log("val/mae_mx", mae_moment[0], prog_bar=True)
            self.log("val/mae_my", mae_moment[1], prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }


def cli_main():
    pl.seed_everything(42)
    dataset_kwargs = dict(
        lag_dz=lag_dz,
        pixmm=pixmm,
        use_nir=use_nir,
        pos_dim=pos_dim,
        depth_source=depth_source,
        normalize_disp=normalize_disp,
        convert_dz_to_pixel=convert_dz_to_pixel,
        convert_disp_to_sensor=convert_disp_to_sensor,
    )

    print("[Dataset Config]")
    for k, v in dataset_kwargs.items():
        print(f"  {k} = {v}")
    print("[Loss Config]")
    print(f"  use_moment_loss = {use_moment_loss}")
    print(f"  moment_arm_mm = {moment_arm_mm}")
    print(f"  moment_loss_weight = {moment_loss_weight}")
    print(f"  moment_loss_force_equiv = {moment_loss_force_equiv}")

    train_dataset = Dataset_gel_force_dynamic_weight(".", record, mode='train', **dataset_kwargs)
    val_dataset = Dataset_gel_force_dynamic_weight(".", record, mode='val', **dataset_kwargs)

    train_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **train_loader_kwargs,
    )

    val_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **val_loader_kwargs,
    )

    print(f"[Info] DataLoader num_workers={num_workers}, persistent_workers={num_workers > 0}")

    logger = TensorBoardLogger(save_dir='force_logs', name="HyperNet_Run")

    model = Lit_Model()

    # 1. 定义早停策略
    early_stop_callback = EarlyStopping(
        monitor='val/loss',      # 监控的指标
        min_delta=1e-4,          # 认为 loss 变化小于这个值就不算“改善”了
        patience=100,             # 如果连续 100 个 epoch 都没有改善，就停止
        verbose=True,            # 输出提示信息
        mode='min'               # 我们希望 loss 越小越好
    )

    # Checkpoint callback
    save_model_cb = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        filename="{epoch}-{val/loss:.5f}",
        save_last=True,
        save_top_k=1  # 只保存最好的那个
    )


    class SaveCodeCallback(Callback):
        """Callback to save specified source files into the trainer logger's version folder on fit start."""
        def __init__(self, files_to_copy=None, dst_subdir='code'):
            super().__init__()
            # default files: this training script, dataset.py, model.py
            if files_to_copy is None:
                files_to_copy = [os.path.abspath(__file__), os.path.abspath('dataset.py'), os.path.abspath('model.py')]
            self.files_to_copy = files_to_copy
            self.dst_subdir = dst_subdir

        def on_fit_start(self, trainer, pl_module):
            # determine logger directory
            log_dir = getattr(trainer.logger, 'log_dir', None)
            if not log_dir:
                save_dir = getattr(trainer.logger, 'save_dir', '.')
                name = getattr(trainer.logger, 'name', '')
                version = getattr(trainer.logger, 'version', '')
                log_dir = os.path.join(save_dir, name, str(version))

            dst_dir = os.path.join(log_dir, self.dst_subdir)
            os.makedirs(log_dir, exist_ok=True)
            runtime_config = {"record": record, **dataset_kwargs,
                              "use_bg_sub": train_dataset.use_bg_sub}
            with open(os.path.join(log_dir, "force_config.json"), "w", encoding="utf-8") as stream:
                json.dump(runtime_config, stream, indent=2)
            try:
                os.makedirs(dst_dir, exist_ok=True)
                for f in self.files_to_copy:
                    if os.path.exists(f):
                        shutil.copy(f, dst_dir)
            except Exception as e:
                # don't crash training for code-copy failure; log a warning
                print(f"Warning: failed to copy code files to {dst_dir}: {e}")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        precision=32,  # HyperNet 数值计算较多，建议用 32 float 保持精度
        # precision='16-mixed',  # 使用混合精度训练以节省显存
        logger=logger,
        callbacks=[save_model_cb, pl.callbacks.LearningRateMonitor(logging_interval='epoch'), early_stop_callback, SaveCodeCallback()],
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        benchmark=True  # 基准测试模式，加速固定输入尺寸的训练
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    # After training, load best checkpoint and record corresponding val metrics (mae_fx, mae_fy, mae_fz)
    try:
        best_ckpt = save_model_cb.best_model_path
        print("Best ckpt:", best_ckpt)
        if best_ckpt is not None and best_ckpt != "":
            best_model = Lit_Model.load_from_checkpoint(best_ckpt)
            # Run validation with the trainer on the best model to get aggregated metrics
            val_results = trainer.validate(best_model, dataloaders=val_dataloader, verbose=False)
            metrics = val_results[0] if isinstance(val_results, list) and len(val_results) > 0 else val_results

            # pick mae keys robustly
            mae_fx = None
            mae_fy = None
            mae_fz = None
            mae_mx = None
            mae_my = None
            val_loss = None
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    key = k.lower()
                    if 'mae_fx' in key or 'mae/fx' in key or 'mae_fx' in k:
                        mae_fx = v
                    if 'mae_fy' in key or 'mae/fy' in key or 'mae_fy' in k:
                        mae_fy = v
                    if 'mae_fz' in key or 'mae/fz' in key or 'mae_fz' in k:
                        mae_fz = v
                    if 'mae_mx' in key or 'mae/mx' in key or 'mae_mx' in k:
                        mae_mx = v
                    if 'mae_my' in key or 'mae/my' in key or 'mae_my' in k:
                        mae_my = v
                    if 'val/loss' in key or key == 'val_loss' or key == 'loss':
                        # prefer explicit val/loss if present
                        if val_loss is None:
                            val_loss = v

            # determine logger directory
            log_dir = getattr(logger, 'log_dir', None)
            if not log_dir:
                save_dir = getattr(logger, 'save_dir', '.')
                name = getattr(logger, 'name', '')
                version = getattr(logger, 'version', '')
                log_dir = os.path.join(save_dir, name, str(version))

            os.makedirs(log_dir, exist_ok=True)
            out_file = os.path.join(log_dir, 'best_ckpt_val_mae.txt')
            with open(out_file, 'w') as f:
                f.write(f"best_ckpt: {best_ckpt}\n")
                if isinstance(metrics, dict):
                    f.write("metrics:\n")
                    for k, v in metrics.items():
                        f.write(f"  {k}: {v}\n")
                f.write(f"val/loss: {val_loss}\n")
                f.write(f"mae_fx: {mae_fx}\n")
                f.write(f"mae_fy: {mae_fy}\n")
                f.write(f"mae_fz: {mae_fz}\n")
                f.write(f"mae_mx: {mae_mx}\n")
                f.write(f"mae_my: {mae_my}\n")

            print(f"Saved best checkpoint metrics to: {out_file}")
        else:
            print("No best checkpoint found to evaluate.")
    except Exception as e:
        print(f"[Warning] Could not evaluate best checkpoint metrics: {e}")


if __name__ == '__main__':
    try:
        print("[Info] Starting training script...")
        cli_main()
    except Exception as e:
        print(f"[Error] Script crashed with exception: {e}")
        import traceback
        traceback.print_exc()
