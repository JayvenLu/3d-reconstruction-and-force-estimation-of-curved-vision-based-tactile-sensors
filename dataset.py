from torch.distributions.constraints import square
from torch.utils.data import Dataset
import torch
import numpy as np
# from scipy.ndimage import gaussian_filter, map_coordinates
import cv2
import torchvision.transforms.functional as TF
# from utils.img_transform import ImageTransformer
import os
import pandas as pd


def _rotation_matrix_x(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float32)


def _rotation_matrix_y(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float32)


def _rotation_matrix_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


# 坐标系对齐约定（按用户提供的 CAD 测量结果）：
# 用户给出的角度定义是“坐标轴旋转（被动旋转）”，不是“向量主动旋转”。
#
# 1) render -> pixel 坐标轴旋转:
#    先绕当前 Y 轴正向 +24°，再绕“已经旋转后的当前 Z 轴”负向 -90°
# 2) pixel -> sensor 坐标轴旋转:
#    先绕当前 X 轴负向 -34°，再绕“已经旋转后的当前 Y 轴”正向 +180°
#
# 这里采用“内禀 / 局部轴”顺序，因此坐标轴累计矩阵按右乘累积：
#   B_new = B_old @ R_local
#
# 对于列向量分量变换，若已知的是“坐标轴旋转”，真正用于 old_coords -> new_coords
# 的矩阵应取对应主动旋转矩阵的逆。旋转矩阵的逆等于转置，所以这里统一写成：
#   R_coords = (R_axes)^T
# 后续代码虽然用的是行向量 [...,3] @ R.T 的实现形式，但物理意义保持一致。
_RENDER_TO_PIXEL_AXES_ROT = _rotation_matrix_y(24.0) @ _rotation_matrix_z(-90.0)
_PIXEL_TO_SENSOR_AXES_ROT = _rotation_matrix_x(-34.0) @ _rotation_matrix_y(180.0)

RENDER_TO_PIXEL_ROT = _RENDER_TO_PIXEL_AXES_ROT.T
PIXEL_TO_SENSOR_ROT = _PIXEL_TO_SENSOR_AXES_ROT.T


def transform_force_label_to_target_axes(
        fx_sensor: float,
        fy_sensor: float,
        fz_sensor: float,
) -> tuple[float, float, float]:
    """
    将原始传感器测量坐标系下的力标签转换到训练目标坐标系。

    目标约定：
    - 按压传感器时 Fz > 0
    - (Fx, Fy, Fz) 保持右手坐标系

    这里采用“传感器测得的外力”到“视触觉传感器所受力”的反号关系：
    [Fx, Fy, Fz] -> [-Fx, -Fy, -Fz]
    """
    return float(-fx_sensor), float(-fy_sensor), float(-fz_sensor)


def transform_torque_label_to_target_axes(
        tx_sensor: float,
        ty_sensor: float,
        tz_sensor: float,
) -> tuple[float, float, float]:
    """
    将原始传感器测量坐标系下的力矩标签转换到训练目标坐标系。

    与力标签保持同样的作用力/反作用力约定，六维 wrench 统一取反。
    """
    return float(-tx_sensor), float(-ty_sensor), float(-tz_sensor)


def get_force_label_from_row(row) -> tuple[float, float, float]:
    """
    从 CSV 行读取力标签，并根据 force_axes 标记判断是否还需要做坐标变换。

    兼容规则：
    - force_axes == target_vts: 已经是训练目标坐标系，直接使用。
    - 其他情况或缺失: 视为原始传感器测量坐标系，转换到目标坐标系。
    """
    fx_raw = float(row['fx']) if 'fx' in row else 0.0
    fy_raw = float(row['fy']) if 'fy' in row else 0.0
    fz_raw = float(row['fz']) if 'fz' in row else 0.0

    force_axes = str(row.get('force_axes', 'sensor_raw')).strip().lower()
    if force_axes == 'target_vts':
        return fx_raw, fy_raw, fz_raw
    return transform_force_label_to_target_axes(fx_raw, fy_raw, fz_raw)


def get_torque_label_from_row(row) -> tuple[float, float, float]:
    """
    从 CSV 行读取力矩标签，并根据 force_axes 标记判断是否还需要做坐标变换。

    兼容规则与 get_force_label_from_row 保持一致。
    """
    tx_raw = float(row['tx']) if 'tx' in row else 0.0
    ty_raw = float(row['ty']) if 'ty' in row else 0.0
    tz_raw = float(row['tz']) if 'tz' in row else 0.0

    force_axes = str(row.get('force_axes', 'sensor_raw')).strip().lower()
    if force_axes == 'target_vts':
        return tx_raw, ty_raw, tz_raw
    return transform_torque_label_to_target_axes(tx_raw, ty_raw, tz_raw)


def transform_disp_to_model_axes(
        dx_pixel: np.ndarray,
        dy_pixel: np.ndarray,
        dz_render: np.ndarray,
        pixmm: float | None,
        convert_dz_to_pixel: bool = True,
        convert_disp_to_sensor: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按配置将位移逐步变换到目标坐标系。

    约定：
    - dx/dy: 原始图像坐标系，+x 向右，+y 向下，单位先从 pixel 转为 mm
    - dz: 来自渲染坐标系，且正方向与 render z 轴相反，因此先表示为 [0, 0, -dz]
    - convert_dz_to_pixel=True:
        将 dz 从渲染坐标系旋转到像素坐标系，并允许产生 x/y 分量
    - convert_disp_to_sensor=True:
        再将统一后的三维位移整体旋转到传感器测量坐标系

    返回:
      若 convert_disp_to_sensor=False，则返回“像素坐标系下（或伪像素坐标系下）”的位移
      若 convert_disp_to_sensor=True，则返回传感器坐标系下的位移
    """
    dx_pixel = np.asarray(dx_pixel, dtype=np.float32)
    dy_pixel = np.asarray(dy_pixel, dtype=np.float32)
    dz_render = np.asarray(dz_render, dtype=np.float32)

    if pixmm is not None:
        dx_mm = dx_pixel * float(pixmm)
        dy_mm = dy_pixel * float(pixmm)
    else:
        dx_mm = dx_pixel
        dy_mm = dy_pixel

    zeros = np.zeros_like(dz_render, dtype=np.float32)

    if convert_dz_to_pixel:
        # 图像坐标系下的平面位移分量。
        disp_pixel_plane = np.stack([dx_mm, dy_mm, zeros], axis=-1)

        # dz 正方向与 render z 轴相反，因此渲染系分量为 [0, 0, -dz]。
        disp_render_depth = np.stack([zeros, zeros, -dz_render], axis=-1)

        # 将 render 深度分量旋转到 pixel 坐标系，再与 dx/dy 合成统一的 pixel 位移向量。
        disp_depth_in_pixel = disp_render_depth @ RENDER_TO_PIXEL_ROT.T
        disp_pixel = disp_pixel_plane + disp_depth_in_pixel
    else:
        # 不做 render->pixel 旋转时，将 dz 直接作为第三通道保留。
        disp_pixel = np.stack([dx_mm, dy_mm, dz_render], axis=-1)

    if convert_disp_to_sensor:
        disp_out = disp_pixel @ PIXEL_TO_SENSOR_ROT.T
    else:
        disp_out = disp_pixel
    return disp_out[..., 0], disp_out[..., 1], disp_out[..., 2]


def compute_lagrangian_dz(
        bg_depth: np.ndarray,
        curr_depth: np.ndarray,
        dx: np.ndarray | None,
        dy: np.ndarray | None,
) -> np.ndarray:
    """
    根据训练数据集中的定义，计算 Lagrangian 风格的 dz。

    思路（与 Dataset_gel_force_dynamic_weight_real_dz 保持一致）：
    - bg_depth: 背景深度（参考坐标系）
    - curr_depth: 当前帧重建得到的深度
    - dx, dy: 在 "背景坐标系" 下的水平位移场（像素单位），表示
        背景点 (x, y) 在当前帧出现在 (x+dx, y+dy)
    - 先用 (dx,dy) 对 curr_depth 做 remap，得到 curr_depth_warped(x,y)=curr_depth(x+dx,y+dy)
    - 再计算 dz = bg_depth - curr_depth_warped，并裁剪为非负

    若 dx 或 dy 为 None，则回退为简单的 Euler 风格：dz = clip(bg_depth - curr_depth, 0, +inf)
    """
    if bg_depth is None or curr_depth is None:
        raise ValueError("bg_depth and curr_depth must not be None")

    # 兼容 dx/dy 缺失：退回旧定义
    if dx is None or dy is None:
        dz = (bg_depth - curr_depth).astype(np.float32)
        dz = np.clip(dz, 0.0, None)
        return dz

    H_dep, W_dep = curr_depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(W_dep), np.arange(H_dep))

    # 分辨率不一致时插值 dx, dy
    if dx.shape != curr_depth.shape:
        dx_for_warp = cv2.resize(dx, (W_dep, H_dep), interpolation=cv2.INTER_LINEAR)
        dy_for_warp = cv2.resize(dy, (W_dep, H_dep), interpolation=cv2.INTER_LINEAR)
    else:
        dx_for_warp = dx
        dy_for_warp = dy

    map_x = (grid_x + dx_for_warp).astype(np.float32)
    map_y = (grid_y + dy_for_warp).astype(np.float32)

    curr_depth_warped = cv2.remap(
        curr_depth.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    dz = (bg_depth - curr_depth_warped).astype(np.float32)
    # dz = np.clip(dz, 0.0, None)

    # 旧版简单定义（仅供参考）：
    # dz_euler = np.clip((bg_depth - curr_depth).astype(np.float32), 0.0, None)

    return dz


def _compute_output_displacement_fields(
        dep_bg_np: np.ndarray,
        dep_raw: np.ndarray,
        dx_pixel: np.ndarray,
        dy_pixel: np.ndarray,
        pixmm: float,
        lag_dz: bool | None,
        use_bg_sub: bool,
        convert_dz_to_pixel: bool,
        convert_disp_to_sensor: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    统一复用的数据预处理逻辑：
    1. 根据配置计算 dz_render
    2. 再将 (dx, dy, dz) 一起变换到传感器坐标系
    """
    if lag_dz is True:
        dz_render = compute_lagrangian_dz(dep_bg_np, dep_raw, dx_pixel, dy_pixel)
    else:
        if use_bg_sub:
            dz_render = np.clip((dep_bg_np - dep_raw).astype(np.float32), 0.0, None)
        else:
            dz_render = dep_raw.astype(np.float32)

    return transform_disp_to_model_axes(
        dx_pixel=dx_pixel,
        dy_pixel=dy_pixel,
        dz_render=dz_render,
        pixmm=pixmm,
        convert_dz_to_pixel=convert_dz_to_pixel,
        convert_disp_to_sensor=convert_disp_to_sensor,
    )


def _stats_tag_float(v: float) -> str:
    return f"{float(v):.6f}".replace(".", "p").replace("-", "m")


def _load_or_compute_disp_norm_stats(
        stats_cache_path: str,
        train_csv_path: str,
        train_mode_dir: str,
        train_dep_path_template: str,
        train_dh_path_template: str,
        pixmm: float,
        lag_dz: bool | None,
        use_bg_sub: bool,
        convert_dz_to_pixel: bool,
        convert_disp_to_sensor: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从训练集自动统计传感器坐标系下的 dx/dy/dz 的全局均值和标准差。
    返回:
      mean: [3]
      std:  [3]
    """
    if os.path.exists(stats_cache_path):
        cached = np.load(stats_cache_path)
        mean = cached["mean"].astype(np.float32)
        std = cached["std"].astype(np.float32)
        return mean, std

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"训练集 CSV 不存在: {train_csv_path}")
    if not os.path.exists(train_mode_dir):
        raise FileNotFoundError(f"训练集深度目录不存在: {train_mode_dir}")

    train_df = pd.read_csv(train_csv_path)
    existing_files = set(os.listdir(train_mode_dir))
    train_bg_path = train_dep_path_template.format(0)
    if not os.path.exists(train_bg_path):
        raise FileNotFoundError(f"训练集背景深度不存在: {train_bg_path}")
    dep_bg_np = np.load(train_bg_path).astype(np.float32)

    sum_vec = np.zeros(3, dtype=np.float64)
    sq_sum_vec = np.zeros(3, dtype=np.float64)
    total_count = 0
    valid_sample_count = 0

    for _, row in train_df.iterrows():
        file_id = int(row["id"])
        dep_filename = os.path.basename(train_dep_path_template.format(file_id))
        if dep_filename not in existing_files:
            continue

        dep_path = train_dep_path_template.format(file_id)
        dep_raw = np.load(dep_path).astype(np.float32)

        dh_path = train_dh_path_template.format(file_id)
        if os.path.exists(dh_path):
            dh = np.load(dh_path).astype(np.float32)
            if dh.ndim == 3 and dh.shape[2] == 2:
                dx_pixel = dh[..., 0]
                dy_pixel = dh[..., 1]
            elif dh.ndim == 3 and dh.shape[0] == 2:
                dx_pixel = dh[0]
                dy_pixel = dh[1]
            else:
                raise ValueError(f"不支持的 dh 形状: {dh.shape}")
        else:
            dx_pixel = np.zeros_like(dep_raw, dtype=np.float32)
            dy_pixel = np.zeros_like(dep_raw, dtype=np.float32)

        dx_sensor, dy_sensor, dz_sensor = _compute_output_displacement_fields(
            dep_bg_np=dep_bg_np,
            dep_raw=dep_raw,
            dx_pixel=dx_pixel,
            dy_pixel=dy_pixel,
            pixmm=pixmm,
            lag_dz=lag_dz,
            use_bg_sub=use_bg_sub,
            convert_dz_to_pixel=convert_dz_to_pixel,
            convert_disp_to_sensor=convert_disp_to_sensor,
        )

        stacked = np.stack([dx_sensor, dy_sensor, dz_sensor], axis=0).astype(np.float64)  # [3,H,W]
        sum_vec += stacked.sum(axis=(1, 2))
        sq_sum_vec += np.square(stacked).sum(axis=(1, 2))
        total_count += stacked.shape[1] * stacked.shape[2]
        valid_sample_count += 1

    if total_count == 0:
        raise RuntimeError(f"未能从训练集统计到任何位移样本: {train_csv_path}")

    mean = (sum_vec / total_count).astype(np.float32)
    var = (sq_sum_vec / total_count) - np.square(mean.astype(np.float64))
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var).astype(np.float32)

    os.makedirs(os.path.dirname(stats_cache_path), exist_ok=True)
    np.savez(
        stats_cache_path,
        mean=mean,
        std=std,
        total_count=np.array([total_count], dtype=np.int64),
        valid_sample_count=np.array([valid_sample_count], dtype=np.int64),
    )
    return mean, std




class Dataset_gel(Dataset):
    def _read_nir(self, path):
        """
        读取NIR图像：
        1. 直接按 8 位灰度读取
        2. 始终返回 (H, W, 1) 的 uint8
        """
        # 直接按 8 位灰度读取（如果文件是 16-bit，OpenCV 以 IMREAD_GRAYSCALE 仍会做合适读取）
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # 扩展通道以确保形状为 (H, W, 1)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        # 支持可配置的缩放因子（例如用于对 NIR 强度进行归一/放大）
        # 如果实例上未定义 self.nir_scale，则使用 1.0
        scale = float(getattr(self, "nir_scale", 1.0))
        if scale != 1.0:
            img_f = img.astype(np.float32) * scale
            img_f = np.clip(img_f, 0, 255)
            img_u8 = img_f.astype(np.uint8)
        else:
            img_u8 = img.astype(np.uint8)

        return img_u8


    def __init__(self, proj_root, record, sample_num=None, nir_training=True, mode='train', nir_scale: float = 1.0):
        """
        触觉传感器数据集类

        参数：
        proj_root: 项目根目录路径
        record: 数据记录标识符(通常为时间戳格式字符串)
        sample_num: （可选）总样本数量；若为 None，则自动根据 images_removed 中可用图像与 normal_gt 的交集确定
        nir_training: 是否使用近红外(NIR)数据进行训练
        mode: 数据集模式, 'train' (默认) 或 'eval'。
              'train' 模式下会进行随机数据处理。
              'eval' 模式下所有操作都是确定性的，用于校准和测试。
        """
        self.mode = mode  # 保存当前模式

        # 基础路径模板
        self.img_rgb_path = f"{proj_root}/data/{record}/images_removed/" + "{}_rgb.png"
        self.img_nir_path = f"{proj_root}/data/{record}/images_removed/" + "{}_nir.png"
        self.normal_dir = f"{proj_root}/data/{record}/normal_gt"
        self.normal_path = os.path.join(self.normal_dir, "normal{}.npy")

        self.nir_training = nir_training
        # 可选的 NIR 缩放因子（像素乘以该系数后再返回），默认 1.0
        self.nir_scale = float(nir_scale)

        # -------------------------
        # 1. 以 images_removed 中的样本图像为准，筛选存在对应 normalX.npy 的样本 id
        # -------------------------
        if not os.path.isdir(self.normal_dir):
            raise FileNotFoundError(f"normal_gt 目录不存在: {self.normal_dir}")
        images_removed_dir = os.path.dirname(self.img_rgb_path)
        if not os.path.isdir(images_removed_dir):
            raise FileNotFoundError(f"images_removed 目录不存在: {images_removed_dir}")

        all_files = os.listdir(images_removed_dir)
        idx_set = set()
        for name in all_files:
            if not name.endswith("_rgb.png"):
                continue
            core = name[:-len("_rgb.png")]
            if not core.isdigit():
                continue
            idx = int(core)
            if idx == 0:
                continue  # 根据 2_normal_generate_auto 的约定，0 是背景帧，跳过
            if not os.path.exists(self.normal_path.format(idx)):
                continue
            if self.nir_training and not os.path.exists(self.img_nir_path.format(idx)):
                continue
            idx_set.add(idx)

        if not idx_set:
            raise RuntimeError(
                f"在 {images_removed_dir} 中未找到任何同时具备已去标记图像和 normal*.npy 的样本文件（除0外）。"
            )

        # 排序后的真实样本 id 列表（可能不连续）
        self.sample_ids = sorted(idx_set)

        # 若用户传入 sample_num，则可以截断使用前 sample_num 个样本
        if sample_num is not None:
            self.sample_ids = self.sample_ids[:sample_num]

        # -------------------------
        # 2. 加载背景 RGB / NIR / normal0
        # -------------------------
        bg_rgb = cv2.imread(self.img_rgb_path.format(0))
        if bg_rgb is None:
            raise FileNotFoundError(f"背景RGB图像缺失: {self.img_rgb_path.format(0)}")
        self.bg_rgb = bg_rgb

        if self.nir_training:
            # 修改：使用 _read_nir 处理16位情况
            bg_nir = self._read_nir(self.img_nir_path.format(0))
            if bg_nir is None:
                raise FileNotFoundError(f"背景NIR图像缺失: {self.img_nir_path.format(0)}")
            self.bg_nir = bg_nir
        else:
            self.bg_nir = None

        H, W = bg_rgb.shape[:2]
        ref_y = -torch.linspace(-H + 1, H - 1, H) / W
        ref_x = torch.linspace(-W + 1, W - 1, W) / W
        ref_y, ref_x = torch.meshgrid([ref_y, ref_x], indexing='ij')
        self.position = torch.stack([ref_x, ref_y], dim=2)

        self.normal_bg = np.load(self.normal_path.format(0))

        self.N = 2048  # 每个样本采样的点数
        # 总样本数由 sample_ids 决定
        self.sample_num = len(self.sample_ids)
        print(
            f"[Dataset_gel] 从 images_removed / normal_gt 交集中发现 {self.sample_num} 个样本 id: "
            f"{self.sample_ids[:10]}" + (" ..." if len(self.sample_ids) > 10 else "")
        )

    def __getitem__(self, index):
        """
        获取单个样本。
        - 在 'train' 模式下，有50%概率使用真实数据或背景数据，并进行随机采样。
        - 在 'eval' 模式下，固定使用真实数据，并进行确定性采样。
        """
        use_real_data = np.random.rand(1) > 0.5 if self.mode == 'train' else True

        if use_real_data:
            # 通过 sample_ids 映射到真实的样本 id（与 normal/图像文件的编号一致）
            sample_id = self.sample_ids[index]

            img_rgb = cv2.imread(self.img_rgb_path.format(sample_id))
            if img_rgb is None:
                raise FileNotFoundError(f"缺少RGB样本图像: {self.img_rgb_path.format(sample_id)}")

            if self.nir_training:
                # 修改：使用 _read_nir 处理16位情况
                img_nir = self._read_nir(self.img_nir_path.format(sample_id))
                if img_nir is None:
                    raise FileNotFoundError(f"缺少NIR样本图像: {self.img_nir_path.format(sample_id)}")
                # img_nir 已经包含 expand_dims 逻辑 (H, W, 1)

            normal = np.load(self.normal_path.format(sample_id))
            mask = ~(normal == [0, 0, 0]).all(axis=2)

            # Optimization: Sample indices first to avoid creating large intermediate arrays
            valid_indices = np.flatnonzero(mask)
            num_valid = valid_indices.size

            if self.mode == 'train':
                if self.N > num_valid:
                    sampled_indices = np.random.choice(valid_indices, self.N, replace=True)
                else:
                    sampled_indices = np.random.choice(valid_indices, self.N, replace=False)
            else:
                rng = np.random.RandomState(sample_id)
                if self.N > num_valid:
                    sampled_indices = rng.choice(valid_indices, self.N, replace=True)
                else:
                    sampled_indices = rng.choice(valid_indices, self.N, replace=False)

            # Extract data using sampled indices
            # .reshape(-1, C) creates a view (usually), [indices] creates a small copy
            
            rgb_subset = img_rgb.reshape(-1, 3)[sampled_indices]
            rgb = torch.from_numpy(rgb_subset.astype(np.float32) / 255.0).view(self.N, 1, 3).permute(2, 1, 0)
            
            bg_rgb_subset = self.bg_rgb.reshape(-1, 3)[sampled_indices]
            bg_rgb = torch.from_numpy(bg_rgb_subset.astype(np.float32) / 255.0).view(self.N, 1, 3).permute(2, 1, 0)
            
            normal_subset = normal.reshape(-1, 3)[sampled_indices]
            gt = torch.from_numpy(normal_subset.astype(np.float32)).view(self.N, 1, 3).permute(2, 1, 0)
            
            pos_subset = self.position.view(-1, 2)[sampled_indices]
            pos = pos_subset.view(self.N, 1, 2).permute(2, 1, 0)

            if self.nir_training:
                nir_subset = img_nir.reshape(-1, 1)[sampled_indices]
                nir = torch.from_numpy(nir_subset.astype(np.float32) / 255.0).view(self.N, 1, 1).permute(2, 1, 0)
                
                bg_nir_subset = self.bg_nir.reshape(-1, 1)[sampled_indices]
                bg_nir = torch.from_numpy(bg_nir_subset.astype(np.float32) / 255.0).view(self.N, 1, 1).permute(2, 1, 0)
                
                output = {'rgb': rgb, 'bg_rgb': bg_rgb, 'nir': nir, 'bg_nir': bg_nir, 'gt': gt, 'position': pos}
            else:
                output = {'rgb': rgb, 'bg_rgb': bg_rgb, 'gt': gt, 'position': pos}

        else:  # mode == 'train' 背景样本
            # Optimization: Sample indices from total pixels
            H, W = self.bg_rgb.shape[:2]
            total_pixels = H * W
            
            indices = np.random.choice(total_pixels, self.N, replace=False)

            bg_rgb_subset = self.bg_rgb.reshape(-1, 3)[indices]
            bg_rgb = torch.from_numpy(bg_rgb_subset.astype(np.float32) / 255.0).view(self.N, 1, 3).permute(2, 1, 0)
            
            normal_subset = self.normal_bg.reshape(-1, 3)[indices]
            gt = torch.from_numpy(normal_subset.astype(np.float32)).view(self.N, 1, 3).permute(2, 1, 0)

            # Fix: self.position is (H,W,2), flatten to (-1,2)
            pos_subset = self.position.view(-1, 2)[indices]
            pos = pos_subset.view(self.N, 1, 2).permute(2, 1, 0)

            if self.nir_training:
                bg_nir_subset = self.bg_nir.reshape(-1, 1)[indices]
                bg_nir = torch.from_numpy(bg_nir_subset.astype(np.float32) / 255.0).view(self.N, 1, 1).permute(2, 1, 0)
                output = {'rgb': bg_rgb, 'bg_rgb': bg_rgb, 'nir': bg_nir, 'bg_nir': bg_nir, 'gt': gt, 'position': pos}
            else:
                output = {'rgb': bg_rgb, 'bg_rgb': bg_rgb, 'gt': gt, 'position': pos}

        return output

    def __len__(self):
        # 返回根据 normal_gt 自动解析得到的样本数量
        return self.sample_num


class Dataset_gel_xyz(Dataset):
    def __init__(self, proj_root, record, sample_num, nir_training=True, mode='train'):
        """
        触觉传感器数据集类

        参数：
        proj_root: 项目根目录路径
        record: 数据记录标识符(通常为时间戳格式字符串)
        sample_num: 总样本数量
        nir_training: 是否使用近红外(NIR)数据进行训练
        mode: 数据集模式, 'train' (默认) 或 'eval'。
              'train' 模式下会进行随机数据处理。
              'eval' 模式下所有操作都是确定性的，用于校准和测试。
        """
        self.mode = mode  # 保存当前模式

        self.img_rgb_path = f"{proj_root}/data/{record}/" + "{}_rgb.png"
        self.img_nir_path = f"{proj_root}/data/{record}/" + "{}_nir.png"
        self.normal_path = f"{proj_root}/data/{record}/" + "normal{}.npy"
        # --- 修改：新增位置编码文件路径 ---
        self.pos_path = f"{proj_root}/data/{record}/" + "pos_encoding{}.npy"

        # H, W = 480, 640
        H, W = 1080, 1920

        # --- 修改：移除旧的二维位置编码生成 ---
        # ref_y = -torch.linspace(-H + 1, H - 1, H) / W
        # ref_x = torch.linspace(-W + 1, W - 1, W) / W
        # ref_y, ref_x = torch.meshgrid([ref_y, ref_x], indexing='ij')
        # self.position = torch.stack([ref_x, ref_y], dim=2)

        bg_rgb = cv2.imread(self.img_rgb_path.format(0))
        bg_nir = cv2.imread(self.img_nir_path.format(0), cv2.IMREAD_GRAYSCALE)

        self.bg_rgb = bg_rgb
        self.bg_nir = np.expand_dims(bg_nir, axis=2) if bg_nir.ndim == 2 else bg_nir
        self.normal_bg = np.load(self.normal_path.format(0))
        # --- 修改：加载背景的三维位置编码 ---
        self.pos_bg = np.load(self.pos_path.format(0))

        self.nir_training = nir_training
        self.N = 2048  # 每个样本采样的点数
        self.sample_num = sample_num  # 总样本数

    def __getitem__(self, index):
        """
        获取单个样本。
        - 在 'train' 模式下，有50%概率使用真实数据或背景数据，并进行随机采样。
        - 在 'eval' 模式下，固定使用真实数据，并进行确定性采样。
        """
        use_real_data = np.random.rand(1) > 0.5 if self.mode == 'train' else True

        if use_real_data:
            img_rgb = cv2.imread(self.img_rgb_path.format(index + 1))
            img_nir = cv2.imread(self.img_nir_path.format(index + 1), cv2.IMREAD_GRAYSCALE)
            img_nir = np.expand_dims(img_nir, axis=2) if img_nir.ndim == 2 else img_nir

            normal = np.load(self.normal_path.format(index + 1))

            # Optimization: Use loaded pos_bg instead of reloading if applicable, 
            # or continue reloading if it changes (but it looks like it was logic for index 0)
            # Original code: pos_xyz = np.load(self.pos_path.format(0))
            # self.pos_bg has the same content.
            pos_xyz = self.pos_bg

            mask = ~(normal == [0, 0, 0]).all(axis=2)

            # Optimization: Sample indices first
            valid_indices = np.flatnonzero(mask)
            num_valid = valid_indices.size

            if self.mode == 'train':
                if self.N > num_valid:
                    sampled_indices = np.random.choice(valid_indices, self.N, replace=True)
                else:
                    sampled_indices = np.random.choice(valid_indices, self.N, replace=False)
            else:
                rng = np.random.RandomState(index)
                if self.N > num_valid:
                    sampled_indices = rng.choice(valid_indices, self.N, replace=True)
                else:
                    sampled_indices = rng.choice(valid_indices, self.N, replace=False)

            # Data extraction
            rgb_subset = img_rgb.reshape(-1, 3)[sampled_indices]
            rgb = torch.from_numpy(rgb_subset.astype(np.float32) / 255.0).view(self.N, 1, 3).permute(2, 1, 0)
            
            bg_rgb_subset = self.bg_rgb.reshape(-1, 3)[sampled_indices]
            bg_rgb = torch.from_numpy(bg_rgb_subset.astype(np.float32) / 255.0).view(self.N, 1, 3).permute(2, 1, 0)
            
            normal_subset = normal.reshape(-1, 3)[sampled_indices]
            gt = torch.from_numpy(normal_subset.astype(np.float32)).view(self.N, 1, 3).permute(2, 1, 0)
            
            pos_subset = pos_xyz.reshape(-1, 3)[sampled_indices]
            pos = torch.from_numpy(pos_subset.astype(np.float32)).view(self.N, 1, 3).permute(2, 1, 0) 

            if self.nir_training:
                nir_subset = img_nir.reshape(-1, 1)[sampled_indices]
                nir = torch.from_numpy(nir_subset.astype(np.float32) / 255.0).view(self.N, 1, 1).permute(2, 1, 0)
                
                bg_nir_subset = self.bg_nir.reshape(-1, 1)[sampled_indices]
                bg_nir = torch.from_numpy(bg_nir_subset.astype(np.float32) / 255.0).view(self.N, 1, 1).permute(2, 1, 0)
                
                output = {'rgb': rgb, 'bg_rgb': bg_rgb, 'nir': nir, 'bg_nir': bg_nir, 'gt': gt, 'position': pos}
            else:
                output = {'rgb': rgb, 'bg_rgb': bg_rgb, 'gt': gt, 'position': pos}

        else:  # mode == 'train' 背景样本
            # Optim
            H, W = self.bg_rgb.shape[:2]
            total_pixels = H * W
            indices = np.random.choice(total_pixels, self.N, replace=False)

            bg_rgb_subset = self.bg_rgb.reshape(-1, 3)[indices]
            bg_rgb = torch.from_numpy(bg_rgb_subset.astype(np.float32) / 255.0).view(self.N, 1, 3).permute(2, 1, 0)
            
            normal_subset = self.normal_bg.reshape(-1, 3)[indices]
            gt = torch.from_numpy(normal_subset.astype(np.float32)).view(self.N, 1, 3).permute(2, 1, 0)

            # Revert: Use self.position (2D) for Dataset_gel
            pos_subset = self.position.view(-1, 2)[indices]
            pos = pos_subset.view(self.N, 1, 2).permute(2, 1, 0)

            if self.nir_training:
                bg_nir_subset = self.bg_nir.reshape(-1, 1)[indices]
                bg_nir = torch.from_numpy(bg_nir_subset.astype(np.float32) / 255.0).view(self.N, 1, 1).permute(2, 1, 0)
                output = {'rgb': bg_rgb, 'bg_rgb': bg_rgb, 'nir': bg_nir, 'bg_nir': bg_nir, 'gt': gt, 'position': pos}
            else:
                output = {'rgb': bg_rgb, 'bg_rgb': bg_rgb, 'gt': gt, 'position': pos}

        return output

    def __len__(self):
        return self.sample_num


class Dataset_gel_force(Dataset):
    def __init__(
            self,
            proj_root,
            record,
            mode='train',
            depth_source='pred',
            use_bg_sub=True,
            pixmm=0.018,
            normalize_disp=False,
            convert_dz_to_pixel=True,
            convert_disp_to_sensor=True,
    ):
        assert mode in ['train', 'val']
        assert depth_source in ['gt', 'pred']
        self.pixmm = pixmm
        self.depth_source = depth_source
        self.normalize_disp = normalize_disp
        self.convert_dz_to_pixel = convert_dz_to_pixel
        self.convert_disp_to_sensor = convert_disp_to_sensor

        # 定义基础路径
        if self.depth_source == 'gt':
            self.mode_dir = os.path.join(proj_root, 'data', record, mode, 'depth_gt')
            self.dep_path_template = os.path.join(self.mode_dir, "depth{}.npy")
            self.train_mode_dir = os.path.join(proj_root, 'data', record, 'train', 'depth_gt')
            self.train_dep_path_template = os.path.join(self.train_mode_dir, "depth{}.npy")
            self.depth_filename_prefix = "depth"
        else:
            self.mode_dir = os.path.join(proj_root, 'data', record, mode, 'depth_pred')
            self.dep_path_template = os.path.join(self.mode_dir, "depth_pred{}.npy")
            self.train_mode_dir = os.path.join(proj_root, 'data', record, 'train', 'depth_pred')
            self.train_dep_path_template = os.path.join(self.train_mode_dir, "depth_pred{}.npy")
            self.depth_filename_prefix = "depth_pred"

        # 新增：dh 位移场目录（与 train 根目录下的 dh 同级）
        self.dh_dir = os.path.join(proj_root, 'data', record, 'train', 'dh')
        self.dh_path_template = os.path.join(self.dh_dir, "{}_dh.npy")
        self.train_dh_path_template = os.path.join(proj_root, 'data', record, 'train', 'dh', "{}_dh.npy")
        self.train_csv_path = os.path.join(proj_root, 'data', record, 'train', 'force_log.csv')

        self.use_bg_sub = use_bg_sub

        # -------------------------------------------------------------------
        # 1. 加载 CSV 并解析 fx, fy, fz （替换原来只用 fz 的逻辑）
        # -------------------------------------------------------------------
        csv_path = os.path.join(proj_root, 'data', record, mode, 'force_log.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件未找到: {csv_path}")

        df = pd.read_csv(csv_path)

        # 获取当前 mode 文件夹下所有可用的 depth_pred 文件 ID
        existing_files = set(os.listdir(self.mode_dir))

        valid_data = []

        for _, row in df.iterrows():
            file_id = int(row['id'])
            filename = f"{self.depth_filename_prefix}{file_id}.npy"

            # 只有当对应的 numpy 文件在当前 mode 文件夹中存在时，才加入数据集
            if filename in existing_files:
                # 新增：从 CSV 中读取 fx, fy, fz 三个分量
                fx, fy, fz = get_force_label_from_row(row)
                valid_data.append({
                    'img_name': file_id,
                    'fx': fx,
                    'fy': fy,
                    'fz': fz,
                })

        self.meta_data = valid_data
        print(f"Mode [{mode}]: 从 CSV 加载了 {len(self.meta_data)} 条有效数据。")

        # -------------------------------------------------------------------
        # 2. 加载背景深度 (ID 0)
        # -------------------------------------------------------------------
        bg_path = self.dep_path_template.format(0)

        if not os.path.exists(bg_path):
            raise FileNotFoundError("背景深度文件缺失: " + bg_path)

        bg_arr = np.load(bg_path).astype(np.float32)
        self.dep_bg_np = bg_arr
        self.dep_bg = torch.from_numpy(bg_arr)

        # -------------------------------------------------------------------
        # 3. 加载背景位移场 dh0.npy （如果存在），并拆分为 dx_bg, dy_bg
        # -------------------------------------------------------------------
        dh_bg_path = self.dh_path_template.format(0)
        if os.path.exists(dh_bg_path):
            dh_bg = np.load(dh_bg_path).astype(np.float32)  # 形状假定为 [H,W,2] 或 [2,H,W]
            if dh_bg.ndim == 3 and dh_bg.shape[2] == 2:
                dx_bg = dh_bg[..., 0]
                dy_bg = dh_bg[..., 1]
            elif dh_bg.ndim == 3 and dh_bg.shape[0] == 2:
                dx_bg = dh_bg[0]
                dy_bg = dh_bg[1]
            else:
                raise ValueError(f"不支持的 dh 背景形状: {dh_bg.shape}, 期望为 (H,W,2) 或 (2,H,W)")
            self.dx_bg = torch.tensor(dx_bg.astype(np.float32))
            self.dy_bg = torch.tensor(dy_bg.astype(np.float32))
        else:
            # 若无背景 dh，使用全零位移
            self.dx_bg = torch.zeros_like(self.dep_bg)
            self.dy_bg = torch.zeros_like(self.dep_bg)

        if self.normalize_disp:
            stats_dir = os.path.join(proj_root, 'data', record, 'train')
            stats_cache_path = os.path.join(
                stats_dir,
                f"disp_norm_stats_depth_{self.depth_source}_pixmm_{_stats_tag_float(self.pixmm)}"
                f"_bgsub_{int(self.use_bg_sub)}"
                f"_dz2pixel_{int(self.convert_dz_to_pixel)}"
                f"_disp2sensor_{int(self.convert_disp_to_sensor)}.npz"
            )
            mean, std = _load_or_compute_disp_norm_stats(
                stats_cache_path=stats_cache_path,
                train_csv_path=self.train_csv_path,
                train_mode_dir=self.train_mode_dir,
                train_dep_path_template=self.train_dep_path_template,
                train_dh_path_template=self.train_dh_path_template,
                pixmm=self.pixmm,
                lag_dz=False,
                use_bg_sub=self.use_bg_sub,
                convert_dz_to_pixel=self.convert_dz_to_pixel,
                convert_disp_to_sensor=self.convert_disp_to_sensor,
            )
            self.disp_mean = mean
            self.disp_std = std
            print(
                f"Mode [{mode}]: 启用位移归一化 (depth_{self.depth_source}, dz2pixel={self.convert_dz_to_pixel}, disp2sensor={self.convert_disp_to_sensor})，"
                f"mean={self.disp_mean.tolist()}, std={self.disp_std.tolist()}"
            )
        else:
            self.disp_mean = None
            self.disp_std = None

        # -------------------------------------------------------------------
        # 4. 生成位置编码 (Positional Encoding)
        # -------------------------------------------------------------------
        H, W = self.dep_bg.shape
        yy = torch.linspace(-1.0, 1.0, H)
        xx = torch.linspace(-1.0, 1.0, W)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')

        self.pos = torch.stack([xx, yy], dim=0).to(self.dep_bg.dtype)  # [2,H,W]

    def __getitem__(self, index):
        meta = self.meta_data[index]
        file_id = meta['img_name']
        fx = meta['fx']
        fy = meta['fy']
        fz = meta['fz']

        # 深度 dz
        dep_path = self.dep_path_template.format(file_id)
        dep_raw = np.load(dep_path).astype(np.float32)
        if dep_raw is None:
            raise IOError(f"读取失败: {dep_path}")

        # 背景减除（dz 保持原逻辑）
        if self.use_bg_sub:
            dep = self.dep_bg.numpy() - dep_raw
        else:
            dep = dep_raw
        dep = np.clip(dep, 0.0, None)

        # 水平位移场 dh：dx, dy（不做背景减除，本身就是位移）
        dh_path = self.dh_path_template.format(file_id)
        if os.path.exists(dh_path):
            dh = np.load(dh_path).astype(np.float32)
            if dh.ndim == 3 and dh.shape[2] == 2:
                dx = dh[..., 0]
                dy = dh[..., 1]
            elif dh.ndim == 3 and dh.shape[0] == 2:
                dx = dh[0]
                dy = dh[1]
            else:
                raise ValueError(f"不支持的 dh 形状: {dh.shape}, 期望为 (H,W,2) 或 (2,H,W)")
        else:
            dx = np.zeros_like(dep_raw, dtype=np.float32)
            dy = np.zeros_like(dep_raw, dtype=np.float32)

        dx_sensor, dy_sensor, dz_sensor = _compute_output_displacement_fields(
            dep_bg_np=self.dep_bg_np,
            dep_raw=dep_raw,
            dx_pixel=dx,
            dy_pixel=dy,
            pixmm=self.pixmm,
            lag_dz=False,
            use_bg_sub=self.use_bg_sub,
            convert_dz_to_pixel=self.convert_dz_to_pixel,
            convert_disp_to_sensor=self.convert_disp_to_sensor,
        )

        if self.normalize_disp:
            dx_sensor = (dx_sensor - self.disp_mean[0]) / self.disp_std[0]
            dy_sensor = (dy_sensor - self.disp_mean[1]) / self.disp_std[1]
            dz_sensor = (dz_sensor - self.disp_mean[2]) / self.disp_std[2]

        dep = torch.from_numpy(dz_sensor).unsqueeze(0)
        dx = torch.from_numpy(dx_sensor).unsqueeze(0)
        dy = torch.from_numpy(dy_sensor).unsqueeze(0)

        # 合成 dep_pos： [5,H,W] = [dz_sensor, dx_sensor, dy_sensor, posX, posY]
        dep_pos = torch.cat([dep, dx, dy, self.pos], dim=0)

        force = torch.tensor([fx, fy, fz], dtype=torch.float32)

        return {
            'dep': dep,
            'dx': dx,
            'dy': dy,
            'dep_pos': dep_pos,
            'force': force,
            'id': file_id
        }

    def __len__(self):
        return len(self.meta_data)



class Dataset_gel_force_dynamic_weight(Dataset):
    def __init__(
            self,
            proj_root,
            record,
            mode='train',
            depth_source='gt',
            use_bg_sub=True,
            lag_dz=True,
            pixmm=0.034,
            use_nir=False,
            pos_dim=3,
            normalize_disp=False,
            convert_dz_to_pixel=True,
            convert_disp_to_sensor=True,
    ):
        assert mode in ['train', 'val']
        assert depth_source in ['gt', 'pred']

        self.pixmm = pixmm
        self.depth_source = depth_source
        self.pos_dim = pos_dim
        self.normalize_disp = normalize_disp
        self.convert_dz_to_pixel = convert_dz_to_pixel
        self.convert_disp_to_sensor = convert_disp_to_sensor

        # 定义基础路径
        if self.depth_source == 'gt':
            self.mode_dir = os.path.join(proj_root, 'data', record, mode, 'depth_gt')
            self.dep_path_template = os.path.join(self.mode_dir, "depth{}.npy")
            self.train_mode_dir = os.path.join(proj_root, 'data', record, 'train', 'depth_gt')
            self.train_dep_path_template = os.path.join(self.train_mode_dir, "depth{}.npy")
            self.depth_filename_prefix = "depth"
        else:
            pred_dir_name = 'depth_pred_nir' if use_nir else 'depth_pred'
            self.mode_dir = os.path.join(proj_root, 'data', record, mode, pred_dir_name)
            self.dep_path_template = os.path.join(self.mode_dir, "depth_pred{}.npy")
            self.train_mode_dir = os.path.join(proj_root, 'data', record, 'train', pred_dir_name)
            self.train_dep_path_template = os.path.join(self.train_mode_dir, "depth_pred{}.npy")
            self.depth_filename_prefix = "depth_pred"

        # 新增：dh 位移场目录
        self.dh_dir = os.path.join(proj_root, 'data', record, mode, 'dh')
        self.dh_path_template = os.path.join(self.dh_dir, "{}_dh.npy")
        self.train_dh_path_template = os.path.join(proj_root, 'data', record, 'train', 'dh', "{}_dh.npy")
        self.train_csv_path = os.path.join(proj_root, 'data', record, 'train', 'force_log.csv')

        self.use_bg_sub = use_bg_sub
        self.lag = lag_dz  # 使用拉格朗日 dz 计算

        # 1. 加载 CSV
        csv_path = os.path.join(proj_root, 'data', record, mode, 'force_log.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件未找到: {csv_path}")

        df = pd.read_csv(csv_path)
        existing_files = set(os.listdir(self.mode_dir))

        valid_data = []
        for _, row in df.iterrows():
            file_id = int(row['id'])
            filename = f"{self.depth_filename_prefix}{file_id}.npy"
            if filename in existing_files:
                fx, fy, fz = get_force_label_from_row(row)
                tx, ty, tz = get_torque_label_from_row(row)
                valid_data.append({
                    'img_name': file_id,
                    'fx': fx,
                    'fy': fy,
                    'fz': fz,
                    'tx': tx,
                    'ty': ty,
                    'tz': tz,
                })

        self.meta_data = valid_data
        print(f"Mode [{mode}]: 从 CSV 加载了 {len(self.meta_data)} 条有效数据。")

        # 2. 加载背景深度 (ID 0)
        bg_path = self.dep_path_template.format(0)
        if not os.path.exists(bg_path):
            raise FileNotFoundError("背景深度文件缺失: " + bg_path)

        bg_arr = np.load(bg_path).astype(np.float32)
        self.dep_bg_np = bg_arr
        self.dep_bg = torch.from_numpy(bg_arr)

        # 3. 生成位置编码 (X, Y, 可选 Z)
        H, W = self.dep_bg.shape
        yy = torch.linspace(-1.0, 1.0, H)
        xx = torch.linspace(-1.0, 1.0, W)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')

        if self.pos_dim == 3:
            z_min, z_max = self.dep_bg.min(), self.dep_bg.max()
            if z_max - z_min > 1e-6:
                zz = (self.dep_bg - z_min) / (z_max - z_min)  # [0, 1]
                zz = zz * 2.0 - 1.0  # [-1, 1]
            else:
                zz = torch.zeros_like(self.dep_bg)
            self.pos = torch.stack([xx, yy, zz], dim=0).to(self.dep_bg.dtype)  # [3,H,W]
        elif self.pos_dim == 2:
            self.pos = torch.stack([xx, yy], dim=0).to(self.dep_bg.dtype)  # [2,H,W]
        else:
            raise ValueError(f"Unsupported pos_dim: {self.pos_dim}. Must be 2 or 3.")

        # 4. dh 背景（若存在）
        dh_bg_path = self.dh_path_template.format(0)
        if os.path.exists(dh_bg_path):
            dh_bg = np.load(dh_bg_path).astype(np.float32)
            if dh_bg.ndim == 3 and dh_bg.shape[2] == 2:
                dx_bg = dh_bg[..., 0]
                dy_bg = dh_bg[..., 1]
            elif dh_bg.ndim == 3 and dh_bg.shape[0] == 2:
                dx_bg = dh_bg[0]
                dy_bg = dh_bg[1]
            else:
                raise ValueError(f"不支持的 dh 背景形状: {dh_bg.shape}")
            self.dx_bg = torch.tensor(dx_bg.astype(np.float32))
            self.dy_bg = torch.tensor(dy_bg.astype(np.float32))
        else:
            self.dx_bg = torch.zeros_like(self.dep_bg)
            self.dy_bg = torch.zeros_like(self.dep_bg)

        if self.normalize_disp:
            stats_dir = os.path.join(proj_root, 'data', record, 'train')
            stats_cache_path = os.path.join(
                stats_dir,
                f"disp_norm_stats_depth_{self.depth_source}_pixmm_{_stats_tag_float(self.pixmm)}"
                f"_lag_{int(self.lag)}"
                f"_bgsub_{int(self.use_bg_sub)}"
                f"_dz2pixel_{int(self.convert_dz_to_pixel)}"
                f"_disp2sensor_{int(self.convert_disp_to_sensor)}.npz"
            )
            mean, std = _load_or_compute_disp_norm_stats(
                stats_cache_path=stats_cache_path,
                train_csv_path=self.train_csv_path,
                train_mode_dir=self.train_mode_dir,
                train_dep_path_template=self.train_dep_path_template,
                train_dh_path_template=self.train_dh_path_template,
                pixmm=self.pixmm,
                lag_dz=self.lag,
                use_bg_sub=self.use_bg_sub,
                convert_dz_to_pixel=self.convert_dz_to_pixel,
                convert_disp_to_sensor=self.convert_disp_to_sensor,
            )
            self.disp_mean = mean
            self.disp_std = std
            print(
                f"Mode [{mode}]: 启用位移归一化 (depth_{self.depth_source}, dz2pixel={self.convert_dz_to_pixel}, disp2sensor={self.convert_disp_to_sensor})，"
                f"mean={self.disp_mean.tolist()}, std={self.disp_std.tolist()}"
            )
        else:
            self.disp_mean = None
            self.disp_std = None

    def __getitem__(self, index):
        meta = self.meta_data[index]
        file_id = meta['img_name']
        fx = meta['fx']
        fy = meta['fy']
        fz = meta['fz']
        tx = meta.get('tx', 0.0)
        ty = meta.get('ty', 0.0)
        tz = meta.get('tz', 0.0)

        # 1. 读取前景深度 (Raw Foreground Depth)
        dep_path = self.dep_path_template.format(file_id)
        dep_raw = np.load(dep_path).astype(np.float32)  # Shape: (H, W)
        if dep_raw is None:
            raise IOError(f"读取失败: {dep_path}")

        # 2. 读取位移场 (Pixel Unit)
        dh_path = self.dh_path_template.format(file_id)
        if os.path.exists(dh_path):
            dh = np.load(dh_path).astype(np.float32)
            # 处理形状兼容性
            if dh.ndim == 3 and dh.shape[2] == 2:
                dx_pixel = dh[..., 0]
                dy_pixel = dh[..., 1]
            elif dh.ndim == 3 and dh.shape[0] == 2:
                dx_pixel = dh[0]
                dy_pixel = dh[1]
            else:
                raise ValueError(f"不支持的 dh 形状: {dh.shape}")
        else:
            dx_pixel = np.zeros_like(dep_raw, dtype=np.float32)
            dy_pixel = np.zeros_like(dep_raw, dtype=np.float32)

        # 3. 计算 dz
        dx_sensor, dy_sensor, dz_sensor = _compute_output_displacement_fields(
            dep_bg_np=self.dep_bg_np,
            dep_raw=dep_raw,
            dx_pixel=dx_pixel,
            dy_pixel=dy_pixel,
            pixmm=self.pixmm,
            lag_dz=self.lag,
            use_bg_sub=self.use_bg_sub,
            convert_dz_to_pixel=self.convert_dz_to_pixel,
            convert_disp_to_sensor=self.convert_disp_to_sensor,
        )

        if self.normalize_disp:
            dx_sensor = (dx_sensor - self.disp_mean[0]) / self.disp_std[0]
            dy_sensor = (dy_sensor - self.disp_mean[1]) / self.disp_std[1]
            dz_sensor = (dz_sensor - self.disp_mean[2]) / self.disp_std[2]

        dz = torch.from_numpy(dz_sensor).unsqueeze(0)  # [1, H, W]
        dx = torch.from_numpy(dx_sensor).unsqueeze(0)
        dy = torch.from_numpy(dy_sensor).unsqueeze(0)

        force = torch.tensor([fx, fy, fz], dtype=torch.float32)
        torque = torch.tensor([tx, ty, tz], dtype=torch.float32)

        return {
            'dz': dz,
            'dx': dx,
            'dy': dy,
            'pos': self.pos,
            'force': force,
            'torque': torque,
            'id': file_id
        }

    def __len__(self):
        return len(self.meta_data)
