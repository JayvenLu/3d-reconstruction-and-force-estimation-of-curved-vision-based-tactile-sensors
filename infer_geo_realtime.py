import cupy as cp
import numpy as np
import torch
from model import Modelv2_ori, Modelv3
from runtime_resources import geometry_settings
import cv2
import torchvision.transforms.functional as TF
# from scipy.fftpack import dst, idst
from cupyx.scipy.fft import dst, idst
import time
from torch.utils.dlpack import to_dlpack
# pip install cupy-cuda11x
# python -m cupyx.tools.install_library --cuda 11.x --library cutensor
import torch.nn.functional as F
import os
import glob
import datetime

from gel_marker_inpaint_viewer import generate_marker_mask, inpaint_markers  # 新增：复用标记点去除与插值逻辑


def _resolve_checkpoint_path(version: int, ckpt_dir: str) -> str:
    preferred = [
        os.path.join(ckpt_dir, "best.ckpt"),
        os.path.join(ckpt_dir, "last.ckpt"),
    ]
    for path in preferred:
        if os.path.isfile(path):
            print(f"[GeoModel] Loading checkpoint: {path}")
            return path

    best_ckpts = [
        path for path in glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True)
        if "best" in os.path.basename(path).lower()
    ]
    if best_ckpts:
        path = max(best_ckpts, key=os.path.getmtime)
        print(f"[GeoModel] best.ckpt not found, loading searched best checkpoint: {path}")
        return path

    ckpts = glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True)
    if ckpts:
        path = max(ckpts, key=os.path.getmtime)
        print(f"[GeoModel] best.ckpt not found, loading latest checkpoint: {path}")
        return path

    raise FileNotFoundError(f"No .ckpt checkpoint found for geometry version {version} in {ckpt_dir}")

def _renorm_normals(n: np.ndarray) -> np.ndarray:
    m = np.linalg.norm(n, axis=2, keepdims=True).astype(np.float32)
    m[m < 1e-6] = 1e-6
    return (n / m).astype(np.float32)

def smooth_normal_spatiotemporal(
    normal: np.ndarray,
    prev_ema: np.ndarray | None,
    gauss_sigma: float = 1.0,
    bilateral_sigma_color: float | None = 0.2,  # None 则用高斯
    bilateral_sigma_space: int = 3,
    ema_alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    - 先做空间平滑（双边或高斯），再做帧间 EMA，最后单位化到 |n|=1
    - 返回 (当前平滑后的法线, 新的 ema 缓存)
    """
    n = normal.astype(np.float32)
    if bilateral_sigma_color is not None:
        # OpenCV 支持 3 通道 float 的双边滤波
        n = cv2.bilateralFilter(n, d=0,
                                sigmaColor=float(bilateral_sigma_color),
                                sigmaSpace=int(bilateral_sigma_space))
    else:
        n = cv2.GaussianBlur(n, ksize=(0, 0),
                             sigmaX=gauss_sigma, sigmaY=gauss_sigma,
                             borderType=cv2.BORDER_REFLECT)

    if prev_ema is None:
        ema = n
    else:
        ema = ema_alpha * n + (1.0 - ema_alpha) * prev_ema

    ema = _renorm_normals(ema)
    return ema, ema  # 第一个做为当前输出，第二个保存为下一帧的 prev_ema



class Geo_Model():
    def __init__(self, version=0, use_pos=None, use_nir=None, enable_channels_last=False, device='cuda'):
        saved = geometry_settings(version)
        if use_pos is not None and use_pos != saved["use_pos"]:
            raise ValueError("use_pos conflicts with the saved geometry configuration")
        if use_nir is not None and use_nir != saved["nir_training"]:
            raise ValueError("use_nir conflicts with the saved geometry configuration")
        use_pos, use_nir = saved["use_pos"], saved["nir_training"]
        self.use_nir = use_nir
        self.device = torch.device(device)
        self.model = torch.nn.Module()
        self.model.basenet = Modelv2_ori(use_pos=use_pos, nir_training=use_nir)
        ckpt_dir = f'logs/gel_experiment/version_{version}/checkpoints'
        ckpt_path = _resolve_checkpoint_path(version, ckpt_dir)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)

        self.use_pos = use_pos
        self.sample = {}
        self.position = None

        self.bg_rgb = None
        self.bg_nir = None
        self._warmed = False

    def _make_position(self, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
        ref_y = -torch.linspace(-h + 1, h - 1, h, device=self.device, dtype=dtype) / w
        ref_x = torch.linspace(-w + 1, w - 1, w, device=self.device, dtype=dtype) / w
        ref_y, ref_x = torch.meshgrid([ref_y, ref_x], indexing='ij')
        return torch.cat([ref_x.unsqueeze(0), ref_y.unsqueeze(0)], dim=0).unsqueeze(0)

    def _ensure_position(self, image_tensor: torch.Tensor) -> None:
        if not self.use_pos:
            return
        _, _, h, w = image_tensor.shape
        dtype = image_tensor.dtype if image_tensor.is_floating_point() else torch.float32
        need_new = (
            self.position is None
            or self.position.shape[-2:] != (h, w)
            or self.position.device != self.device
            or self.position.dtype != dtype
        )
        if need_new:
            self.position = self._make_position(h, w, dtype)
        self.sample['position'] = self.position

    def load_bg(self, bg_rgb, bg_nir=None):
        if bg_rgb is None:
            raise ValueError("bg_rgb must not be None")
        bg_rgb_tensor = TF.to_tensor(bg_rgb).unsqueeze(0).to(self.device)
        self.bg_rgb = bg_rgb_tensor
        self.sample['bg_rgb'] = bg_rgb_tensor
        self._ensure_position(bg_rgb_tensor)

        if self.use_nir:
            if bg_nir is None:
                raise ValueError("use_nir=True requires bg_nir input")
            bg_nir_tensor = TF.to_tensor(bg_nir).unsqueeze(0).to(self.device)
            self.bg_nir = bg_nir_tensor
            self.sample['bg_nir'] = bg_nir_tensor
        else:
            self.bg_nir = None
            self.sample.pop('bg_nir', None)

    def warmup(self, runs=10):
        if self._warmed:
            return
        if 'rgb' not in self.sample:
            raise RuntimeError("Please call predict() once (measure=False) or preload sample before warmup().")
        with torch.inference_mode():
            for _ in range(runs):
                _ = self.model.basenet(self.sample)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        self._warmed = True

    def _forward_once(self):
        return self.model.basenet(self.sample)

    def predict(self, rgb_fg: torch.Tensor, nir_fg: torch.Tensor | None = None,
                measure: bool = False, repeats: int = 50):
        """与离线版一致：仅在 use_nir=True 时使用 nir_fg。"""
        rgb_fg = rgb_fg.to(self.device)
        self.sample['rgb'] = rgb_fg
        self._ensure_position(rgb_fg)
        if self.use_nir:
            if nir_fg is None:
                raise ValueError("use_nir=True requires nir_fg input")
            self.sample['nir'] = nir_fg.to(self.device)
        else:
            self.sample.pop('nir', None)

        if measure and not self._warmed:
            self.warmup(runs=10)

        if not measure:
            with torch.inference_mode():
                pred = self._forward_once()
            pred = F.normalize(pred, p=2, dim=1).squeeze(0).permute(1, 2, 0)
            return pred, 0.0

        with torch.inference_mode():
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                for _ in range(repeats):
                    pred = self._forward_once()
                ender.record()
                torch.cuda.synchronize()
                avg_ms = starter.elapsed_time(ender) / repeats
            else:
                t0 = time.perf_counter()
                for _ in range(repeats):
                    pred = self._forward_once()
                t1 = time.perf_counter()
                avg_ms = (t1 - t0) * 1000.0 / repeats

        pred = F.normalize(pred, p=2, dim=1).squeeze(0).permute(1, 2, 0)
        return pred, avg_ms



if __name__ == '__main__':
    # --- 1. 初始化模型 ---
    # model_idx = 29  # 自研五指
    # model_idx = 32  # 拾玥食指1号
    # model_idx = 36  # 自研新三指1号
    # model_idx = 38  # 自研新三指1号 FPGA推理
    # model_idx = 40  # 自研五指1号
    # model_idx = 50  # 内窥镜三指2号
    model_idx = 62  # 曲面三指1号
    model = Geo_Model(version=model_idx, use_pos=True, use_nir=False)

    enable_inpaint = True  # 新增：是否对输入帧进行标记点去除与插值
    inpaint_radius = 3     # inpaint 半径（像素）
    inpaint_dilate_iter = 5  # 掩膜膨胀迭代次数

    # --- 2. 打开摄像头 ---
    cap = cv2.VideoCapture(0) # 0 代表默认摄像头
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        exit()
    # 显式设置分辨率为 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"实际相机分辨率: {actual_w} x {actual_h}", flush=True)

    # --- 3. 设置循环和控制标志 ---
    bg_set = False
    print("请将相机对准背景，按 'b' 键设置背景帧。")
    print("实时推断时，按 'q' 键退出。")

    ema_normal = None  # 新增：法线的 EMA 缓存

    while True:
        # 初始化计时器
        torch.cuda.synchronize()
        prev_time = time.time()

        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头帧")
            break

        # 获取键盘输入
        key = cv2.waitKey(1) & 0xFF

        # 新增：运行时切换是否启用 inpaint
        if key == ord('i'):
            enable_inpaint = not enable_inpaint
            print(f"[实时] 标记点 inpaint 预处理已切换为: {enable_inpaint}")

        if not bg_set:
            # 在设置背景前，只显示原始图像并提示
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press 'b' to set background", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            window_name = 'Original Camera Feed'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, int(actual_w), int(actual_h))
            cv2.imshow(window_name, display_frame)

            if key == ord('b'):
                # 对背景帧也做一次 inpaint 以保持前景/背景风格一致（可选）
                if enable_inpaint:
                    try:
                        bg_mask = generate_marker_mask(frame, method="v")
                        bg_inpainted, _ = inpaint_markers(frame, bg_mask, method="telea",
                                                          radius=inpaint_radius,
                                                          dilate_iter=inpaint_dilate_iter)
                        model.load_bg(bg_inpainted)
                        print("背景已设置（已去除标记点），开始实时推断...")
                    except Exception as e:
                        print(f"[警告] 背景 inpaint 失败，使用原始背景: {e}")
                        model.load_bg(frame)
                else:
                    model.load_bg(frame)
                    print("背景已设置（未做 inpaint），开始实时推断...")

                bg_set = True
                cv2.destroyWindow('Original Camera Feed') # 关闭提示窗口
        else:
            # 可选：对当前帧进行标记点去除
            proc_frame = frame
            if enable_inpaint:
                try:
                    mask = generate_marker_mask(frame, method="v")
                    proc_frame, _ = inpaint_markers(frame, mask, method="telea",
                                                    radius=inpaint_radius,
                                                    dilate_iter=inpaint_dilate_iter)
                except Exception as e:
                    print(f"[警告] 实时 inpaint 失败，使用原始帧: {e}")
                    proc_frame = frame

            rgb_fg_tensor = TF.to_tensor(proc_frame).unsqueeze(0).to(model.device)
            if model.use_nir:
                nir_fg_tensor = TF.to_tensor(proc_frame).unsqueeze(0).to(model.device)  # TODO: 替换为真实 NIR
                pred_normal, _ = model.predict(rgb_fg_tensor, nir_fg_tensor, measure=False)
            else:
                pred_normal, _ = model.predict(rgb_fg_tensor, measure=False)

            # --- 5. 准备并显示结果 ---
            normal_np = pred_normal.cpu().detach().numpy()
            normal_np_copy = normal_np.copy()

            # 在可视化前对前两个分量进行缩放
            if normal_np_copy.ndim == 3 and normal_np_copy.shape[2] >= 2:
                normal_np_copy[..., :2] = normal_np_copy[..., :2] / 0.1

            # normal_np, ema_normal = smooth_normal_spatiotemporal(
            #     normal_np,
            #     prev_ema=ema_normal,
            #     gauss_sigma=1.0,
            #     bilateral_sigma_color=0.2,
            #     bilateral_sigma_space=3,
            #     ema_alpha=0.3,
            # )

            normal_display = cv2.cvtColor(normal_np_copy * 0.5 + 0.5, cv2.COLOR_RGB2BGR)

            cv2.imshow("Original", frame)
            cv2.putText(normal_display, f"inpaint: {enable_inpaint}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow("Predicted Normal Map", normal_display)

        # 按 'q' 键退出循环
        if key == ord('q'):
            print("退出程序...")
            break
        elif key == ord('s'):
            # 获取当前时间并格式化为字符串，例如 '20230715_153000'
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            # 使用时间戳生成动态文件名
            npy_filename = f'normal_pred_{timestamp}.npy'
            png_filename = f'normal_pred_{timestamp}.png'
            orig_filename = f'original_frame_{timestamp}.png'

            # 保存文件
            np.save(npy_filename, normal_np)
            cv2.imwrite(png_filename, normal_display * 255)  # 将[0,1]范围的图像转为[0,255]
            cv2.imwrite(orig_filename, frame)

            # 打印保存信息
            print(f"法线预测结果已保存为 '{npy_filename}'")
            print(f"原始帧已保存为 '{orig_filename}'")
            print(f"法线图已保存为 '{png_filename}'")

    # --- 6. 释放资源 ---
    cap.release()
    cv2.destroyAllWindows()
