# --- 曲面视触觉传感器实时力估计 ---
import ast
import glob
import os
import subprocess
import threading
import time
import datetime
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.ticker as mticker

from infer_geo_realtime import Geo_Model
from runtime_resources import geometry_settings, force_settings
from depth_reconstruction_core import load_depth_prior, reconstruct_with_boundary_prior
from model import HyperForceNet
from dataset import (
    _compute_output_displacement_fields,
    _load_or_compute_disp_norm_stats,
    _stats_tag_float,
)

# 新增：标记点去除与位移场计算依赖
from gel_marker_inpaint_viewer import generate_marker_mask, inpaint_markers
import setting
import find_marker
import marker_detection
import RBF
from offline_perspective_dfs_match import (
    get_perspective_matrix,
    perspective_transform,
    backwarp_points_to_raw,
    draw_markers,
    visualize_dense_field_arrows,
)

ENABLE_RT_DISPLACEMENT_VIS = False
LOW_LATENCY_CAPTURE = True
CAMERA_DROP_GRABS = 3
USE_LATEST_FRAME_THREAD = True

# 曲面三指1号
# PTS_SRC = np.float32([[166, 144], [520, 138], [584, 438], [98, 456]])

# 曲面三指2号
# PTS_SRC = np.float32([[122, 96], [528, 94], [584, 434], [48, 444]])

# 曲面三指2号 20260827
PTS_SRC = np.float32([[50, 50], [590, 50], [590, 430], [50, 430]])

# 华威科样品1号 20260827
# PTS_SRC = np.float32([[34, 26], [578, 24], [500, 394], [96, 420]])

# 多光谱三指3号
# PTS_SRC = np.float32([[594, 440], [1344, 442], [1488, 1018], [382, 998]])

# 曲面三指2号
V_BLOCKSIZE_DIS = 165
V_C_DIS = 20
V_BLOCKSIZE_INPAINT = 165
V_C_INPAINT = 15 
CIRCULARITY_MIN = 0.61

# 多光谱三指3号
# V_BLOCKSIZE_DIS = 211
# V_C_DIS = 14
# V_BLOCKSIZE_INPAINT = 211
# V_C_INPAINT = 15 
# CIRCULARITY_MIN = 0.61

# 曲面三指1号
# PRESET_GRID_PATH = "20260120-165420/grid_preset_rgb_perspective.npy"

# 曲面三指1号
# PRESET_GRID_PATH = "20260120-165420/grid_preset_rgb_perspective.npy"

# 曲面三指2号
# PRESET_GRID_PATH = "data/20260404_103156/train/grid_preset_rgb_perspective.npy"

# 曲面三指2号 20260827
PRESET_GRID_PATH = "cam/20260827-145539/grid_preset_rgb_perspective.npy"

# 华威科样品1号 20260827
# PRESET_GRID_PATH = "cam/20260827-150753/grid_preset_rgb_perspective.npy"

# 多光谱三指3号
# PRESET_GRID_PATH = "data/20260114_120750/grid_preset_rgb_perspective.npy"

# ==========================================
# 工具函数
# ==========================================
_PRIOR_CACHE: dict[tuple[str, tuple[int, int]], np.ndarray | None] = {}


def _read_literal_assignments(py_path: str, wanted_keys: set[str]) -> dict[str, object]:
    if not os.path.exists(py_path):
        return {}
    try:
        with open(py_path, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read(), filename=py_path)
    except Exception as exc:
        print(f"[Config] Warning: failed to parse {py_path}: {exc}")
        return {}

    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted_keys:
                values[target.id] = value
    return values


def _load_version_config(version_dir: str, candidates: list[str], defaults: dict[str, object]) -> tuple[dict[str, object], str | None]:
    config = dict(defaults)
    wanted = set(defaults)
    for rel_path in candidates:
        path = os.path.join(version_dir, rel_path)
        values = _read_literal_assignments(path, wanted)
        if values:
            config.update(values)
            return config, path
    return config, None


def _resolve_checkpoint_path(version: int, ckpt_dir: str, preferred_names: tuple[str, ...], label: str) -> str:
    for name in preferred_names:
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            print(f"[{label}] Loading checkpoint: {path}")
            return path

    all_ckpts = glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True)
    best_ckpts = [path for path in all_ckpts if "best" in os.path.basename(path).lower()]
    if best_ckpts:
        path = max(best_ckpts, key=os.path.getmtime)
        print(f"[{label}] Preferred checkpoint not found, loading searched best checkpoint: {path}")
        return path

    if all_ckpts:
        path = max(all_ckpts, key=os.path.getmtime)
        print(f"[{label}] Preferred checkpoint not found, loading latest checkpoint: {path}")
        return path

    raise FileNotFoundError(f"No .ckpt checkpoint found for {label} version {version} in {ckpt_dir}")


def _camera_device_path(camera_index) -> str | None:
    if isinstance(camera_index, int):
        return f"/dev/video{camera_index}"
    if isinstance(camera_index, str) and camera_index.startswith("/dev/video"):
        return camera_index
    return None


def configure_v4l2_camera(camera_index, width: int = 640, height: int = 480, fps: int = 20, pixfmt: str = "MJPG") -> None:
    device = _camera_device_path(camera_index)
    if device is None or not os.path.exists(device):
        return

    cmd = [
        "v4l2-ctl",
        "-d", device,
        f"--set-fmt-video=width={width},height={height},pixelformat={pixfmt}",
        f"--set-parm={fps}",
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("[Camera] v4l2-ctl not found; falling back to OpenCV camera settings.")
        return

    if result.returncode == 0:
        print(f"[Camera] v4l2 configured {device}: {width}x{height} {pixfmt} {fps}fps")
    else:
        msg = (result.stderr or result.stdout or "").strip()
        print(f"[Camera] Warning: v4l2 config failed for {device}: {msg}")


class RealtimeProfiler:
    def __init__(self, report_interval_sec: float = 2.0):
        self.report_interval_sec = report_interval_sec
        self.last_report = time.perf_counter()
        self.frame_count = 0
        self.totals: dict[str, float] = {}

    def add(self, name: str, elapsed_sec: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + elapsed_sec

    def finish_frame(self, total_sec: float) -> None:
        self.add("total", total_sec)
        self.frame_count += 1
        now = time.perf_counter()
        if now - self.last_report < self.report_interval_sec:
            return

        fps = self.frame_count / max(now - self.last_report, 1e-9)
        ordered = [
            "capture",
            "inpaint",
            "geo",
            "displacement",
            "depth_dz",
            "force",
            "visualize",
            "waitkey",
            "total",
        ]
        parts = [f"FPS={fps:.2f}"]
        for name in ordered:
            if name in self.totals:
                avg_ms = self.totals[name] * 1000.0 / max(self.frame_count, 1)
                parts.append(f"{name}={avg_ms:.1f}ms")
        print("[Profiler] " + " | ".join(parts))
        self.last_report = now
        self.frame_count = 0
        self.totals.clear()


def read_camera_frame(cap: cv2.VideoCapture, low_latency: bool = True, drop_grabs: int = 3):
    if low_latency:
        for _ in range(max(0, int(drop_grabs))):
            cap.grab()
    return cap.read()


class LatestFrameCapture:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.running = True
        self.ret = False
        self.frame = None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame if ret else None
            if not ret:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)


def load_geo_version_config(version: int) -> dict[str, object]:
    return geometry_settings(version)


def load_force_version_config(version: int, pixmm_fallback: float) -> dict[str, object]:
    # pixmm_fallback is retained for call compatibility, never used to guess metadata.
    return force_settings(version)


def _renorm_normals(n: np.ndarray) -> np.ndarray:
    m = np.linalg.norm(n, axis=2, keepdims=True).astype(np.float32)
    m[m < 1e-6] = 1e-6
    return (n / m).astype(np.float32)


def smooth_normal_spatiotemporal(
        normal: np.ndarray,
        prev_ema: np.ndarray | None,
        gauss_sigma: float = 1.0,
        bilateral_sigma_color: float | None = 0.2,
        bilateral_sigma_space: int = 3,
        ema_alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    n = normal.astype(np.float32)
    if bilateral_sigma_color is not None:
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
    return ema, ema


def get_resized_prior(depth_prior_path: str | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not depth_prior_path:
        return None
    key = (depth_prior_path, shape_hw)
    if key in _PRIOR_CACHE:
        return _PRIOR_CACHE[key]
    if not os.path.exists(depth_prior_path):
        _PRIOR_CACHE[key] = None
        return None
    try:
        prior = load_depth_prior(depth_prior_path, shape_hw=shape_hw)
        _PRIOR_CACHE[key] = prior
        return prior
    except Exception:
        _PRIOR_CACHE[key] = None
        return None


def reconstruct_depth_via_poisson(
        normal_np: np.ndarray,
        depth_prior_path: str | None,
        beta: float = 0.0,
) -> np.ndarray:
    H, W = normal_np.shape[:2]
    prior = get_resized_prior(depth_prior_path, (H, W))
    depth = reconstruct_with_boundary_prior(
        normal=normal_np,
        depth_prior=prior,
        mask=None,
        beta=beta,
    )
    return depth.astype(np.float32)


def preprocess_rgb_with_inpaint(rgb_img, inpaint_radius=3, dilate_iter=5):
    """对单帧 RGB 做标记点检测+inpaint 去除，沿用 gel_marker_inpaint_viewer 的参数。"""
    if rgb_img is None:
        return None, None
    mask = generate_marker_mask(rgb_img, block_size=V_BLOCKSIZE_INPAINT, c=V_C_INPAINT)
    inpainted, _ = inpaint_markers(
        rgb_img,
        mask,
        method="telea",
        radius=inpaint_radius,
        dilate_iter=dilate_iter,
    )
    return inpainted, mask


class DisplacementRuntime:
    """
    实时位移场计算封装类。
    关键逻辑：
    1. 只有一个 matcher 实例。
    2. set_background 时运行一次，锁定 matcher.O (参考网格)。
    3. compute_dx_dy 时运行，计算当前点与 matcher.O 的差异。
    """

    def __init__(self, W_raw, H_raw):
        # 透视变换矩阵，用于原图 <-> 校正图 的相互变换
        pts_dst = np.float32([[0, 0], [W_raw, 0], [W_raw, H_raw], [0, H_raw]])
        self.M = get_perspective_matrix(PTS_SRC, pts_dst)
        self.M_inv = np.linalg.inv(self.M)

        self.roi_mask = None
        self.H_raw = H_raw
        self.W_raw = W_raw

        # RBF 插值配置（与 offline_perspective_dfs_match 保持一致的默认值）
        from offline_perspective_dfs_match import CURRENT_KERNEL_TYPE, CURRENT_EPSILON, CURRENT_REGULARIZATION
        self.rbf_kernel_type = CURRENT_KERNEL_TYPE
        self.rbf_epsilon = CURRENT_EPSILON
        self.rbf_regularization = CURRENT_REGULARIZATION

        # 初始化匹配器参数（从 setting 模块读取网格参数）
        setting.init()
        # 【关键】Matcher 仅初始化一次，保持状态
        self.matcher = find_marker.Matching(
            N_=setting.N_,
            M_=setting.M_,
            fps_=setting.fps_,
            x0_=setting.x0_,
            y0_=setting.y0_,
            dx_=setting.dx_,
            dy_=setting.dy_,
            preset_grid_path=PRESET_GRID_PATH,
        )

        self._last_du = None
        self._last_dv = None
        self._last_mc = None
        self._last_flow = None  # 用于 draw_marker_flow_on_raw
        self._frame_id = 1

    def set_background(self, bg_rgb: np.ndarray):
        """
        使用背景帧校准网格。
        这会触发 matcher 内部的 flag_record=1 逻辑，将 self.O 更新为背景点位置。
        """
        self.H_raw, self.W_raw = bg_rgb.shape[:2]

        # 1) 透视变换到校正视角
        bg_warp = perspective_transform(bg_rgb, self.M, (self.W_raw, self.H_raw))
        # 2) 图像增强
        bg_enhanced = marker_detection.init(bg_warp)
        # 3) 生成 V 通道掩膜并检测标记点
        mask_v = self._calculate_V_mark_runtime(bg_enhanced)
        mc_bg = list(self._detect_markers(mask_v))

        print(f"[DispRuntime] Background markers detected: {len(mc_bg)}")

        if len(mc_bg) < 4:
            print("[DispRuntime] Warning: Too few markers in background!")

        # 4) 运行匹配器以锁定背景网格
        # find_marker.Matching 只在 current_frame_id == 0 时把检测结果写回 O。
        self.matcher.flag_record = 1
        self.matcher.set_current_frame_id(0)
        self.matcher.init(mc_bg)
        self.matcher.run()

        # 检查是否成功校准
        if self.matcher.minf == -1:
            print("[DispRuntime] Warning: Background matching failed! Displacement will be wrong.")
        else:
            print("[DispRuntime] Background calibrated successfully. Reference grid locked.")
            self._frame_id = 1

    def _calculate_V_mark_runtime(self, frame, debug: bool = False, debug_prefix: str = "Vmark_rt"):
        """轻量版 V 通道标记点掩膜生成。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        v_blur = cv2.GaussianBlur(v, (35, 35), 0)
        mask = cv2.adaptiveThreshold(
            src=v_blur,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=V_BLOCKSIZE_DIS,
            C=V_C_DIS,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        mask_dilate = cv2.dilate(mask_closed, kernel, iterations=1)
        mask_erode = cv2.erode(
            mask_dilate,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        return mask_erode

    def compute_dx_dy(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        对单帧计算致密位移场 dx, dy。
        由于 matcher 已在 set_background 中校准，这里的 get_flow() 返回的
        Ox, Oy 就是背景坐标，Cx, Cy 就是当前坐标。
        """
        if self.H_raw is None or self.W_raw is None:
            # 这是一个保险，如果用户忘了调 set_background，这里调一次
            # 但这会导致第一帧位移为0
            self.set_background(frame_rgb)

        # 1) 透视变换
        frame_warp = perspective_transform(frame_rgb, self.M, (self.W_raw, self.H_raw))
        # 2) 图像增强
        frame_enhanced = marker_detection.init(frame_warp)
        # 3) 检测标记点
        mask_v = self._calculate_V_mark_runtime(frame_enhanced)
        mc = list(self._detect_markers(mask_v))

        # --- Visualization of Marker Detection ---
        mask_vis = cv2.cvtColor(mask_v, cv2.COLOR_GRAY2BGR)
        mask_vis = cv2.resize(mask_vis, (640, 480))
        # Be careful with coordinate scaling if frame_enhanced size != 640x480.
        # frame_enhanced is from perspective_transform output size (self.W_raw, self.H_raw).
        # We assume self.W_raw, self.H_raw matches input frame resolution (e.g. 640x480 or similar).
        scale_x = 640 / self.W_raw
        scale_y = 480 / self.H_raw
        for pt in mc:
            cv2.circle(mask_vis, (int(pt[0] * scale_x), int(pt[1] * scale_y)), 3, (0, 255, 0), -1)

        cv2.putText(mask_vis, f"Block: {V_BLOCKSIZE_DIS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(mask_vis, f"C: {V_C_DIS}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(mask_vis, f"Circ: {CIRCULARITY_MIN:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(mask_vis, "Keys: [/] Block   -/= C   n/m Circ", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Disp Marker Mask", mask_vis)
        # -----------------------------------------

        # 4) 匹配
        # 此时 matcher.flag_record 应为 0，self.O 保持为背景网格
        self.matcher.flag_record = 0
        self.matcher.set_current_frame_id(self._frame_id)
        self._frame_id += 1
        self.matcher.init(mc)
        self.matcher.run()

        # 获取稀疏光流
        # Ox, Oy: 背景网格坐标 (Fixed)
        # Cx, Cy: 当前匹配坐标
        # Occupied: 网格状态
        flow = self.matcher.get_flow()
        Ox, Oy, Cx, Cy, Occupied, dx_g, dy_g, distance = flow

        # 缓存用于可视化
        self._last_mc = mc
        self._last_flow = flow

        # 5) 收集 RBF 控制点 (反投影回原图)
        grid_src_raw = []
        grid_dst_raw = []
        N = len(Ox)
        M_grid = len(Ox[0]) if N > 0 else 0

        for i in range(N):
            for j in range(M_grid):
                # 只有当该网格点在当前帧被占据时，才是有意义的位移
                # 注意：find_marker 中 Ox 是始终存在的（全网格），但只有 Occupied > -1 的点才有对应的 Cx
                if Occupied[i][j] > -1:
                    # 原图坐标系下的起点 (背景)
                    pt1_warp = np.array([[[Ox[i][j], Oy[i][j]]]], dtype=np.float32)
                    pt1_raw = cv2.perspectiveTransform(pt1_warp, self.M_inv)[0, 0]

                    # 原图坐标系下的终点 (当前)
                    pt2_warp = np.array([[[Cx[i][j], Cy[i][j]]]], dtype=np.float32)
                    pt2_raw = cv2.perspectiveTransform(pt2_warp, self.M_inv)[0, 0]

                    grid_src_raw.append([float(pt1_raw[0]), float(pt1_raw[1])])
                    grid_dst_raw.append([float(pt2_raw[0]), float(pt2_raw[1])])

        grid_src_raw = np.asarray(grid_src_raw, dtype=np.float32)
        grid_dst_raw = np.asarray(grid_dst_raw, dtype=np.float32)

        # 6) RBF 插值得到全图致密位移场
        if grid_src_raw.shape[0] >= 4:
            # print(f"[RT-RBF] Using kernel_type = {self.rbf_kernel_type}, epsilon = {self.rbf_epsilon:.2f}, reg = {self.rbf_regularization:.4f}")
            du_field, dv_field = RBF.compute_displacement_field_in_raw(
                src_points=grid_src_raw,
                dst_points=grid_dst_raw,
                height=self.H_raw,
                width=self.W_raw,
                kernel_type=self.rbf_kernel_type,
                epsilon=self.rbf_epsilon,
                regularization=self.rbf_regularization,
                mask=None,
                show=False,
            )
        else:
            du_field = np.zeros((self.H_raw, self.W_raw), dtype=np.float32)
            dv_field = np.zeros((self.H_raw, self.W_raw), dtype=np.float32)

        self._last_du = du_field
        self._last_dv = dv_field

        return du_field, dv_field

    @staticmethod
    def _detect_markers(mask):
        """简单的标记点检测，基于圆度过滤。"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 0.0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)

            if circularity < CIRCULARITY_MIN:
                continue

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                markers.append((cX, cY))
        return markers

    def draw_marker_flow_on_raw(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        在原图上画出标记点位移箭头。
        使用 _last_flow 中的 (Ox,Oy) 作为起点，(Cx,Cy) 作为终点。
        """
        vis = raw_frame.copy()
        if not hasattr(self, '_last_flow') or self._last_flow is None:
            return vis

        Ox, Oy, Cx, Cy, Occupied, dx_g, dy_g, distance = self._last_flow
        N = len(Ox)
        M_grid = len(Ox[0]) if N > 0 else 0

        for i in range(N):
            for j in range(M_grid):
                if Occupied[i][j] > -1:
                    pt1_warp = np.array([[[Ox[i][j], Oy[i][j]]]], dtype=np.float32)
                    pt2_warp = np.array([[[Cx[i][j], Cy[i][j]]]], dtype=np.float32)
                    pt1_raw = cv2.perspectiveTransform(pt1_warp, self.M_inv)[0, 0]
                    pt2_raw = cv2.perspectiveTransform(pt2_warp, self.M_inv)[0, 0]

                    u0, v0 = float(pt1_raw[0]), float(pt1_raw[1])
                    u1, v1 = float(pt2_raw[0]), float(pt2_raw[1])

                    scale = 2.0
                    dx = (u1 - u0) * scale
                    dy = (v1 - v0) * scale

                    iu0, iv0 = int(round(u0)), int(round(v0))
                    iu1, iv1 = int(round(u0 + dx)), int(round(v0 + dy))

                    cv2.arrowedLine(vis, (iu0, iv0), (iu1, iv1), (0, 255, 0), 2, tipLength=0.3)
                    cv2.circle(vis, (iu0, iv0), 2, (80, 80, 80), -1)
                else:
                    # 可以在这里画出未匹配的背景点位置(Ox, Oy)作为调试
                    pass
        return vis

    def get_last_displacement_fields(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """返回最近一帧缓存的致密位移场 (du, dv)。若尚未计算则返回 (None, None)。"""
        return self._last_du, self._last_dv

    def visualize_dense_displacement_on_raw(
            self,
            raw_frame: np.ndarray,
            step: int = 25,
            scale: float = 5.0,
            min_rel: float = 0.05,
    ) -> np.ndarray:
        """在原图上叠加可视化稀疏箭头，展示插值后的致密位移场。"""
        if self._last_du is None or self._last_dv is None:
            return raw_frame
        if raw_frame is None:
            return raw_frame
        return visualize_dense_field_arrows(raw_frame, self._last_du, self._last_dv,
                                            step=step, scale=scale, min_rel=min_rel)

    def visualize_displacement_magnitude_colormap(self) -> np.ndarray | None:
        """将最近一帧位移场的幅值转为伪彩色图像，用于单独窗口显示。"""
        if self._last_du is None or self._last_dv is None:
            return None
        du_field = self._last_du
        dv_field = self._last_dv
        mag = np.sqrt(du_field ** 2 + dv_field ** 2)
        if not np.isfinite(mag).any():
            return None
        # 参考 offline_perspective_dfs_match.py 中的可视化逻辑
        mag_flat = mag.reshape(-1)
        mag_valid = mag_flat[np.isfinite(mag_flat)]
        if mag_valid.size == 0:
            return None
        vmax = float(np.percentile(mag_valid, 99.0))
        if vmax < 1e-8:
            return None
        mag_clip = np.clip(mag, 0.0, vmax)
        mag_norm = cv2.normalize(mag_clip, None, 0, 255, cv2.NORM_MINMAX)
        mag_norm = mag_norm.astype(np.uint8)
        mag_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
        return mag_color


# ==========================================
# 核心模型类
# ==========================================
class HyperForce_Model:
    def __init__(self, version: int, device: torch.device, pixmm_fallback: float = 0.034):
        self.device = device
        self.config = load_force_version_config(version, pixmm_fallback)
        self.record = self.config.get("record")
        self.pixmm = float(self.config.get("pixmm", pixmm_fallback))
        self.use_nir = bool(self.config.get("use_nir", False))
        self.lag_dz = bool(self.config.get("lag_dz", True))
        self.pos_dim = int(self.config.get("pos_dim", 3))
        self.depth_source = str(self.config.get("depth_source", "pred"))
        self.normalize_disp = bool(self.config.get("normalize_disp", False))
        self.convert_dz_to_pixel = bool(self.config.get("convert_dz_to_pixel", True))
        self.convert_disp_to_sensor = bool(self.config.get("convert_disp_to_sensor", True))
        self.use_bg_sub = bool(self.config.get("use_bg_sub", True))
        self.disp_mean = None
        self.disp_std = None
        self._init_disp_norm_stats()

        self.model = torch.nn.Module()
        self.model.basenet = HyperForceNet(kernel_size=5, stride=4, pos_dim=self.pos_dim).to(self.device)
        self.pos = None
        ckpt_dir = f'force_logs/HyperNet_Run/version_{version}/checkpoints'
        ckpt_path = _resolve_checkpoint_path(version, ckpt_dir, ("last.ckpt", "best.ckpt"), "ForceModel")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_key = k.replace('model.', 'basenet.', 1)
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def _init_disp_norm_stats(self) -> None:
        if not self.normalize_disp:
            return
        if not self.record:
            raise ValueError("normalize_disp=True requires the matching training record and statistics")

        pred_dir_name = "depth_pred_nir" if self.use_nir else "depth_pred"
        depth_dir_name = pred_dir_name if self.depth_source == "pred" else "depth_gt"
        depth_prefix = "depth_pred" if self.depth_source == "pred" else "depth"
        train_mode_dir = os.path.join("data", str(self.record), "train", depth_dir_name)
        train_dep_path_template = os.path.join(train_mode_dir, f"{depth_prefix}{{}}.npy")
        train_dh_path_template = os.path.join("data", str(self.record), "train", "dh", "{}_dh.npy")
        train_csv_path = os.path.join("data", str(self.record), "train", "force_log.csv")
        stats_cache_path = os.path.join(
            "data",
            str(self.record),
            "train",
            f"disp_norm_stats_depth_{self.depth_source}_pixmm_{_stats_tag_float(self.pixmm)}"
            f"_lag_{int(self.lag_dz)}"
            f"_bgsub_{int(self.use_bg_sub)}"
            f"_dz2pixel_{int(self.convert_dz_to_pixel)}"
            f"_disp2sensor_{int(self.convert_disp_to_sensor)}.npz",
        )
        try:
            self.disp_mean, self.disp_std = _load_or_compute_disp_norm_stats(
                stats_cache_path=stats_cache_path,
                train_csv_path=train_csv_path,
                train_mode_dir=train_mode_dir,
                train_dep_path_template=train_dep_path_template,
                train_dh_path_template=train_dh_path_template,
                pixmm=self.pixmm,
                lag_dz=self.lag_dz,
                use_bg_sub=self.use_bg_sub,
                convert_dz_to_pixel=self.convert_dz_to_pixel,
                convert_disp_to_sensor=self.convert_disp_to_sensor,
            )
            print(
                "[ForceModel] Loaded displacement normalization "
                f"mean={self.disp_mean.tolist()}, std={self.disp_std.tolist()}"
            )
        except Exception as exc:
            raise RuntimeError("Cannot load the model's displacement normalization statistics; see README.md") from exc

    def set_background(self, bg_arr: np.ndarray):
        H, W = bg_arr.shape
        yy = torch.linspace(-1.0, 1.0, H, device=self.device)
        xx = torch.linspace(-1.0, 1.0, W, device=self.device)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        if self.pos_dim == 3:
            bg_tensor = torch.from_numpy(bg_arr).to(self.device)
            z_min, z_max = bg_tensor.min(), bg_tensor.max()
            if z_max - z_min > 1e-6:
                zz = (bg_tensor - z_min) / (z_max - z_min)
                zz = zz * 2.0 - 1.0
            else:
                zz = torch.zeros_like(bg_tensor)
            pos = torch.stack([xx, yy, zz], dim=0)
        elif self.pos_dim == 2:
            pos = torch.stack([xx, yy], dim=0)
        else:
            raise ValueError(f"Unsupported force pos_dim: {self.pos_dim}")
        self.pos = pos.unsqueeze(0).float()
        print(f"[ForceModel] Background set. Pos encoding shape: {self.pos.shape}")

    def prepare_displacement_inputs(
            self,
            bg_depth: np.ndarray,
            curr_depth: np.ndarray,
            dx_pixel: np.ndarray,
            dy_pixel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx_model, dy_model, dz_model = _compute_output_displacement_fields(
            dep_bg_np=bg_depth,
            dep_raw=curr_depth,
            dx_pixel=dx_pixel,
            dy_pixel=dy_pixel,
            pixmm=self.pixmm,
            lag_dz=self.lag_dz,
            use_bg_sub=self.use_bg_sub,
            convert_dz_to_pixel=self.convert_dz_to_pixel,
            convert_disp_to_sensor=self.convert_disp_to_sensor,
        )
        if self.normalize_disp and self.disp_mean is not None and self.disp_std is not None:
            dx_model = (dx_model - self.disp_mean[0]) / self.disp_std[0]
            dy_model = (dy_model - self.disp_mean[1]) / self.disp_std[1]
            dz_model = (dz_model - self.disp_mean[2]) / self.disp_std[2]
        return dx_model.astype(np.float32), dy_model.astype(np.float32), dz_model.astype(np.float32)

    @torch.no_grad()
    def predict(self, dep_np: np.ndarray) -> torch.Tensor:
        if self.pos is None:
            raise RuntimeError("请先调用 set_background 设置背景深度！")
        dep = torch.from_numpy(dep_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        sample = {'dep': dep, 'pos': self.pos}
        pred = self.model.basenet(sample)
        return pred

    @torch.no_grad()
    def predict_full(self, dz_np: np.ndarray, dx_np: np.ndarray, dy_np: np.ndarray) -> np.ndarray:
        """根据位移场 (dx,dy,dz) 与 pos_3d 预测三维力场 [3,H,W]。"""
        if self.pos is None:
            raise RuntimeError("请先调用 set_background 设置背景深度！")
        dz = torch.from_numpy(dz_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        dx = torch.from_numpy(dx_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        dy = torch.from_numpy(dy_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        sample = {
            'dz': dz,
            'dx': dx,
            'dy': dy,
            'pos': self.pos,
        }
        pred = self.model.basenet(sample)  # [1,3,H,W]
        return pred.squeeze(0).detach().cpu().numpy()  # [3,H,W]


# ==========================================
# 可视化工具
# ==========================================

def draw_colorbar_with_ticks(
        height: int,
        bar_width: int,
        total_width: int,
        cmap_id: int,
        vmin: float,
        vmax: float,
        unit: str,
        num_ticks: int = 6
) -> np.ndarray:
    """
    绘制精细颜色条，避免文字重叠
    """
    canvas = np.zeros((height, total_width, 3), dtype=np.uint8)

    # 1. 颜色梯度
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
    gradient = cv2.resize(gradient, (bar_width, height))
    bar_img = cv2.applyColorMap(gradient, cmap_id)
    canvas[:, :bar_width] = bar_img

    # 2. 刻度与文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)
    thickness = 1

    top_margin = 35
    bottom_margin = 15

    if abs(vmax - vmin) < 1e-9:
        ticks = [vmin]
        y_positions = [height // 2]
    else:
        ticks = np.linspace(vmax, vmin, num_ticks)
        y_positions = np.linspace(top_margin, height - bottom_margin, num_ticks, dtype=int)

    tick_len = 5
    text_offset = bar_width + tick_len + 5

    for val, y in zip(ticks, y_positions):
        cv2.line(canvas, (bar_width, y), (bar_width + tick_len, y), color, 1)
        if abs(val) < 0.01 and abs(val) > 0:
            label = f"{val:.1e}"
        else:
            label = f"{val:.2f}"
        cv2.putText(canvas, label, (text_offset, y + 4), font, font_scale, color, thickness)

    # 3. 单位
    unit_label = f"[{unit}]"
    cv2.putText(canvas, unit_label, (bar_width + 5, 25), font, 0.5, (0, 255, 255), 1)

    return canvas


def visualize_map_with_bar(
        data_map: np.ndarray,
        out_size: tuple[int, int],
        cmap_id: int,
        unit: str,
        scale_factor: float = 1.0
) -> np.ndarray:
    bar_section_width = 100
    bar_width = 25

    if data_map is None or data_map.size == 0 or not np.isfinite(data_map).any():
        total_w = out_size[0] + bar_section_width
        return np.zeros((out_size[1], total_w, 3), dtype=np.uint8)

    d_scaled = data_map * scale_factor

    if unit == "N":
        vmin, vmax = -5e-4, 5e-4
    elif unit == "mm/sub":
        vmin, vmax = 0, 1.2
    else:
        vmin, vmax = d_scaled.min(), d_scaled.max()

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6

    d_norm = np.clip((d_scaled - vmin) / (vmax - vmin), 0.0, 1.0)
    d_resized = cv2.resize(d_norm, out_size, interpolation=cv2.INTER_LINEAR)
    d_color = cv2.applyColorMap((d_resized * 255).astype(np.uint8), cmap_id)

    sidebar = draw_colorbar_with_ticks(
        height=out_size[1],
        bar_width=bar_width,
        total_width=bar_section_width,
        cmap_id=cmap_id,
        vmin=vmin,
        vmax=vmax,
        unit=unit,
        num_ticks=7
    )

    return np.hstack((d_color, sidebar))


def visualize_map_core_only(
        data_map: np.ndarray,
        out_size: tuple[int, int],
        cmap_id: int,
        scale_factor: float = 1.0,
) -> np.ndarray:
    """仅返回带 colormap 的数据区域（无 colorbar），用于保存纯图像。"""
    if data_map is None or data_map.size == 0 or not np.isfinite(data_map).any():
        return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)

    d_scaled = data_map * scale_factor
    vmin, vmax = -50, 50
    # vmin, vmax = d_scaled.min(), d_scaled.max()
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    d_norm = (d_scaled - vmin) / (vmax - vmin)
    d_resized = cv2.resize(d_norm, out_size, interpolation=cv2.INTER_LINEAR)
    d_color = cv2.applyColorMap((d_resized * 255).astype(np.uint8), cmap_id)
    return d_color


def visualize_normal(normal_np: np.ndarray, out_size: tuple[int, int], pixmm: float = 1.0) -> np.ndarray:
    n_vis = normal_np.copy().astype(np.float32)
    pixmm_safe = max(abs(float(pixmm)), 1e-6)

    n_vis[..., 0] = n_vis[..., 0] / pixmm_safe
    n_vis[..., 1] = n_vis[..., 1] / pixmm_safe

    norm = np.linalg.norm(n_vis, axis=-1, keepdims=True).astype(np.float32)
    norm[norm < 1e-6] = 1e-6
    n_vis = n_vis / norm

    normal_vis = ((n_vis * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    resized = cv2.resize(normal_vis, out_size, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


def visualize_displacement_arrows(frame_rgb: np.ndarray,
                                  dx: np.ndarray,
                                  dy: np.ndarray,
                                  dz: np.ndarray | None,
                                  step: int = 20) -> np.ndarray:
    """在原图上画二维箭头表示二维位移 (dx,dy)，用 dz 大小做颜色编码。"""
    vis = frame_rgb.copy()
    H, W = vis.shape[:2]
    if dz is None:
        dz = np.zeros((H, W), dtype=np.float32)
    for v in range(0, H, step):
        for u in range(0, W, step):
            dx_ = dx[v, u]
            dy_ = dy[v, u]
            dz_ = dz[v, u]
            if dx_ == 0 and dy_ == 0 and dz_ == 0:
                continue
            scale = 3.0
            end_u = int(u + dx_ * scale)
            end_v = int(v + dy_ * scale)
            mag_z = min(max(abs(dz_) * 1000.0, 0), 255)  # 简单缩放
            color = (0, int(255 - mag_z), int(mag_z))  # 蓝->红
            cv2.arrowedLine(vis, (u, v), (end_u, end_v), color, 1, tipLength=0.3)
    return vis


def visualize_force_composite(fx: np.ndarray,
                              fy: np.ndarray,
                              fz: np.ndarray,
                              step: int = 20) -> np.ndarray:
    """
    可视化三维分布力：
    1. 背景：Fz 的热力图 (Inferno)
    2. 前景：Fx, Fy 的白色箭头场
    """
    H, W = fx.shape

    # 1. 生成 Fz 热力图背景
    # 归一化 Fz 到 0-255，并使用 INFERNO 色图 (黑色->紫色->橙色->黄色，适合表示强度)
    # 注意：根据实际力的范围调整 vmin, vmax
    vmax_fz = np.percentile(np.abs(fz), 99) + 1e-6
    fz_norm = np.clip(np.abs(fz) / vmax_fz, 0, 1)
    vis_bg = cv2.applyColorMap((fz_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    # 2. 叠加切向力箭头
    for v in range(step // 2, H, step):
        for u in range(step // 2, W, step):
            val_fx = fx[v, u]
            val_fy = fy[v, u]

            # 过滤微小力，保持画面整洁
            mag_xy = np.sqrt(val_fx ** 2 + val_fy ** 2)
            if mag_xy < 0.05 * vmax_fz:  # 阈值可调
                continue

            # 箭头参数
            scale = 100.0  # 根据力的量级调整
            start_point = (u, v)
            end_point = (int(u + val_fx * scale), int(v + val_fy * scale))  # 注意坐标系方向

            # 画白色箭头
            cv2.arrowedLine(vis_bg, start_point, end_point, (255, 255, 255), 1, tipLength=1.0)

    return vis_bg


def visualize_force_arrows_on_raw(frame_rgb: np.ndarray,
                                  fx: np.ndarray | None,
                                  fy: np.ndarray | None,
                                  step: int = 25,
                                  scale: float = 50.0,
                                  min_rel: float = 0.05) -> np.ndarray:
    """在原图上用箭头可视化水平力场 (Fx,Fy)，类似 marker_flow_on_raw 的效果。

    参数:
        frame_rgb: 原始相机图像，BGR
        fx, fy: 水平力分量，单位 N，形状 [H,W]
        step: 采样步长（像素），步长越大箭头越稀疏
        scale: 箭头长度缩放系数，把牛顿值映射到像素位移
        min_rel: 相对阈值，小于 max(|F|)*min_rel 的力不画，避免噪点
    """
    vis = frame_rgb.copy()
    if fx is None or fy is None:
        return vis
    if fx.shape != fy.shape:
        return vis

    H, W = fx.shape
    # 计算全局最大切向力幅值，用于阈值
    mag = np.sqrt(fx ** 2 + fy ** 2)
    if not np.isfinite(mag).any():
        return vis
    max_mag = float(np.nanmax(mag))
    if max_mag < 1e-8:
        return vis

    thr = max_mag * float(min_rel)

    for v in range(step // 2, H, step):
        for u in range(step // 2, W, step):
            fx_ = float(fx[v, u])
            fy_ = float(fy[v, u])
            m_ = np.sqrt(fx_ ** 2 + fy_ ** 2)
            if m_ < thr:
                continue
            # 将力大小映射为箭头长度（像素）
            du = int(round(fx_ * scale))
            dv = int(round(fy_ * scale))
            p0 = (u, v)
            p1 = (u + du, v + dv)
            # 根据力大小做简单颜色编码：弱->蓝，强->红
            rel = np.clip(m_ / max_mag, 0.0, 1.0)
            g = int(255 * (1.0 - rel))
            r = int(255 * rel)
            color = (0, g, r)
            cv2.arrowedLine(vis, p0, p1, color, 1, tipLength=0.3)

    return vis


# 统一 Fx/Fy/Fz 力分量热力图的函数定义（不再接受 unit/scale_factor 关键字）

def visualize_force_component_maps(
        fx: np.ndarray | None,
        fy: np.ndarray | None,
        fz: np.ndarray | None,
        out_size: tuple[int, int],
) -> np.ndarray:
    """分别可视化 Fx / Fy / Fz 的热力图，每个带颜色条，更直观对比三个分量。

    返回拼接后的 BGR 图像： [Fx | Fy | Fz]
    若某一分量为 None，则用全黑占位。
    """

    def _vis_one(comp: np.ndarray | None, cmap: int) -> np.ndarray:
        return visualize_map_with_bar(
            comp if comp is not None else None,
            out_size,
            cmap,
            unit="N",  # 统一使用 N 作为单位
            scale_factor=1.0,  # 统一缩放因子
        )

    fx_vis = _vis_one(fx, cv2.COLORMAP_INFERNO)
    fy_vis = _vis_one(fy, cv2.COLORMAP_INFERNO)
    fz_vis = _vis_one(fz, cv2.COLORMAP_INFERNO)

    return np.hstack((fx_vis, fy_vis, fz_vis))


def visualize_dx_dy_colormaps(dx: np.ndarray | None,
                              dy: np.ndarray | None,
                              out_size: tuple[int, int],
                              scale_factor: float = 1.0
                              ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """将 dx、dy 分量分别转为伪彩色图像，用于独立显示。

    返回 (dx_color, dy_color)，若对应分量为 None 则返回 None。
    显示时建议使用 COLORMAP_JET 或其他颜色图。
    """
    if dx is None or not np.isfinite(dx).any():
        dx_color = None
    else:
        dx_color = visualize_map_core_only(dx, out_size, cv2.COLORMAP_JET, scale_factor=scale_factor)

    if dy is None or not np.isfinite(dy).any():
        dy_color = None
    else:
        dy_color = visualize_map_core_only(dy, out_size, cv2.COLORMAP_JET, scale_factor=scale_factor)

    return dx_color, dy_color


# ==========================================
# 3D 力分布可视化（从 infer_force_single_offline 复用）
# ==========================================

def visualize_3d_force_distribution(depth_map,
                                    fx,
                                    fy,
                                    fz,
                                    step: int = 15,
                                    scale_arrow: float = 0.5,
                                    title: str = "3D Force Distribution",
                                    save_path: str | None = None,
                                    view_elev: float = 35,
                                    view_azim: float = -60,
                                    cbar_pad: float = 0.08,
                                    x_tick_count: int = 5,
                                    tick_fontsize: int = 8,
                                    ) -> None:
    """三个窗口显示力分量 (z=force) + 三个窗口 depth+color (z=depth)。

    参数:
        depth_map: (H, W) 深度图
        fx, fy, fz: (H, W) 力分量
        step: 采样步长
        scale_arrow: 保留参数（旧接口兼容）
        title: 图标题前缀
        save_path: 若不为 None，则将图像保存到该路径（自动追加 _force/_depth）
    """
    if fx is None or fy is None or fz is None:
        return

    H, W = fx.shape
    y_idx = np.arange(0, H, step)
    x_idx = np.arange(0, W, step)
    X, Y = np.meshgrid(x_idx, y_idx)

    def get_norm(C: np.ndarray) -> plt.Normalize:
        vmin = float(np.nanmin(C))
        vmax = float(np.nanmax(C))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
        return plt.Normalize(vmin=vmin, vmax=vmax)

    def plot_force_surface(component: np.ndarray, label: str, fig_path: str | None) -> None:
        Z = component[0:H:step, 0:W:step].astype(np.float32)
        if not np.isfinite(Z).any():
            return
        norm = get_norm(Z)
        cmap = plt.cm.viridis
        facecolors = cmap(norm(Z))
        facecolors[~np.isfinite(Z)] = (0, 0, 0, 0)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
        ax.set_title(f"{title} - {label}")
        ax.set_xlabel('X Coordinate (pixels)')
        ax.set_ylabel('Y Coordinate (pixels)')
        ax.set_zlabel(f"{label} (N)")
        ax.view_init(elev=view_elev, azim=view_azim)

        # reduce X/Y tick density to avoid overlap
        try:
            xticks = np.linspace(0, W - 1, min(x_tick_count, W)).astype(int)
            yticks = np.linspace(0, H - 1, min(x_tick_count, H)).astype(int)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([str(int(t)) for t in xticks], fontsize=tick_fontsize)
            ax.set_yticklabels([str(int(t)) for t in yticks], fontsize=tick_fontsize)
        except Exception:
            pass

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=12, pad=cbar_pad)
        # format colorbar ticks in scientific notation (e format)
        cbar.formatter = mticker.FormatStrFormatter('%.1e')
        cbar.update_ticks()
        cbar.set_label(f"{label} (N)")

        plt.tight_layout()
        if fig_path is not None:
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def plot_depth_color(component: np.ndarray, label: str, fig_path: str | None) -> None:
        if depth_map is None or depth_map.shape != component.shape:
            return
        Z = depth_map[0:H:step, 0:W:step].astype(np.float32)
        C = component[0:H:step, 0:W:step].astype(np.float32)
        valid = np.isfinite(Z) & np.isfinite(C)
        if not np.isfinite(C).any():
            return
        norm = get_norm(C)
        cmap = plt.cm.viridis
        facecolors = cmap(norm(C))
        facecolors[~valid] = (0, 0, 0, 0)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
        ax.set_title(f"{title} (Depth+Color) - {label}")
        ax.set_xlabel('X Coordinate (pixels)')
        ax.set_ylabel('Y Coordinate (pixels)')
        ax.set_zlabel('Depth (mm)')
        ax.view_init(elev=view_elev, azim=view_azim)

        # reduce X/Y tick density to avoid overlap
        try:
            xticks = np.linspace(0, W - 1, min(x_tick_count, W)).astype(int)
            yticks = np.linspace(0, H - 1, min(x_tick_count, H)).astype(int)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([str(int(t)) for t in xticks], fontsize=tick_fontsize)
            ax.set_yticklabels([str(int(t)) for t in yticks], fontsize=tick_fontsize)
        except Exception:
            pass

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=12, pad=cbar_pad)
        # format colorbar ticks in scientific notation (e format)
        cbar.formatter = mticker.FormatStrFormatter('%.1e')
        cbar.update_ticks()
        cbar.set_label(f"{label} (N)")

        plt.tight_layout()
        if fig_path is not None:
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    base_path = None
    ext = None
    if save_path is not None:
        base_path, ext = os.path.splitext(save_path)
        if not ext:
            ext = ".png"

    plot_force_surface(fx, "Fx", f"{base_path}_force_Fx{ext}" if base_path else None)
    plot_force_surface(fy, "Fy", f"{base_path}_force_Fy{ext}" if base_path else None)
    plot_force_surface(fz, "Fz", f"{base_path}_force_Fz{ext}" if base_path else None)

    plot_depth_color(fx, "Fx", f"{base_path}_depth_Fx{ext}" if base_path else None)
    plot_depth_color(fy, "Fy", f"{base_path}_depth_Fy{ext}" if base_path else None)
    plot_depth_color(fz, "Fz", f"{base_path}_depth_Fz{ext}" if base_path else None)


# ==========================================
# 主程序
# ==========================================
def main(
        camera_index: int = 0,
        model_version: int = 56,
        depth_prior_path: str | None = None,
        save_dir: str = ".",
        enable_force: bool = True,
        force_model_version: int = 13,
        pixmm: float = 0.034,  # 与 Dataset_gel_force_dynamic_weight 中保持一致的像素->毫米系数
):
    global V_BLOCKSIZE_DIS, V_C_DIS, CIRCULARITY_MIN
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initialize Geometry Model...")
    geo_config = load_geo_version_config(model_version)
    geo_use_pos = bool(geo_config.get("use_pos", True))
    geo_use_nir = bool(geo_config.get("nir_training", False))
    if geo_use_nir:
        raise RuntimeError("The selected geometry model was trained with NIR, but realtime NIR input is not wired here.")
    geo_model = Geo_Model(version=model_version, use_pos=geo_use_pos, use_nir=geo_use_nir, device=device)

    force_model = None
    if enable_force:
        try:
            print(f"Initialize Force Model (Version {force_model_version})...")
            force_model = HyperForce_Model(version=force_model_version, device=device, pixmm_fallback=pixmm)
            pixmm = force_model.pixmm
        except Exception as e:
            print(f"Force Model Error: {e}")
            enable_force = False

    # 显示与保存的 dz 使用所选力模型版本保存的预处理配置。
    # 即使力模型未启用或加载失败，也按该版本的配置选择 Lagrangian dz 或普通深度差。
    if force_model is not None:
        display_lag_dz = force_model.lag_dz
        display_use_bg_sub = force_model.use_bg_sub
    else:
        force_dz_config = load_force_version_config(force_model_version, pixmm)
        display_lag_dz = bool(force_dz_config.get("lag_dz", True))
        display_use_bg_sub = bool(force_dz_config.get("use_bg_sub", True))
    print(
        "[DisplayDZ] Using force model preprocessing config: "
        f"version={force_model_version}, lag_dz={display_lag_dz}, use_bg_sub={display_use_bg_sub}"
    )

    configure_v4l2_camera(camera_index, width=640, height=480, fps=20, pixfmt="MJPG")
    # cap = cv2.VideoCapture(camera_index)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 20)
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    cap_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap_fourcc_str = "".join(chr((cap_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(f"[Camera] source={camera_index}, size={cap_w}x{cap_h}, fps={cap_fps:.2f}, fourcc={cap_fourcc_str!r}")

    # 新增：实时位移场计算器
    disp_runtime = DisplacementRuntime(cap_w, cap_h)

    bg_set = False
    bg_depth: np.ndarray | None = None
    bg_rgb_clean: np.ndarray | None = None

    last_normal_np = None
    last_depth = None
    last_force_map = None
    last_dz = None

    print("\n=== Controls ===")
    print("  'b': Set Background")
    print("  'Enter': Save Snapshot")
    print("  '[' / ']': Adjust BlockSize")
    print("  '-' / '=': Adjust C")
    print("  'q': Quit")
    print("================")
    profiler = RealtimeProfiler(report_interval_sec=2.0)
    latest_capture = LatestFrameCapture(cap) if USE_LATEST_FRAME_THREAD else None
    if latest_capture is not None:
        print("[Camera] Latest-frame capture thread enabled.")

    try:
        while True:
            frame_t0 = time.perf_counter()
            stage_t = frame_t0
            if latest_capture is not None:
                ret, frame = latest_capture.read()
            else:
                ret, frame = read_camera_frame(cap, low_latency=LOW_LATENCY_CAPTURE, drop_grabs=CAMERA_DROP_GRABS)
            now = time.perf_counter()
            profiler.add("capture", now - stage_t)
            stage_t = now
            if not ret:
                time.sleep(0.005)
                continue
            H_img, W_img = frame.shape[:2]

            # 标记点去除
            frame_clean, marker_mask = preprocess_rgb_with_inpaint(frame, inpaint_radius=3, dilate_iter=5)
            now = time.perf_counter()
            profiler.add("inpaint", now - stage_t)
            stage_t = now
    
            if not bg_set:
                disp = frame.copy()
                cv2.putText(disp, "Press 'b' to set Background", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Camera Feed", disp)
    
                key = cv2.waitKey(1) & 0xFF
                now = time.perf_counter()
                profiler.add("waitkey", now - stage_t)
                if key == ord('b'):
                    print("Setting background...")
                    bg_rgb_clean = frame_clean.copy()
                    geo_model.load_bg(bg_rgb_clean)
                    img_t = TF.to_tensor(bg_rgb_clean).unsqueeze(0).to(device)
                    norm_t, _ = geo_model.predict(img_t)
                    bg_norm = norm_t.detach().cpu().numpy()
                    try:
                        bg_depth = reconstruct_depth_via_poisson(bg_norm, depth_prior_path)
                        if enable_force and force_model:
                            force_model.set_background(bg_depth)
                        # 位移模块背景：必须用带标记点的原始背景图
                        disp_runtime.set_background(frame)
                        bg_set = True
                        cv2.destroyWindow("Camera Feed")
                        print("Background set.")
                    except Exception as e:
                        print(f"BG Depth Error: {e}")
                if key == ord('q'):
                    break
                profiler.finish_frame(time.perf_counter() - frame_t0)
                continue
    
            # 几何：法向预测（基于去标记点后的图像）
            img_t = TF.to_tensor(frame_clean).unsqueeze(0).to(device)
            norm_t, _ = geo_model.predict(img_t)
            normal_np = norm_t.detach().cpu().numpy()
            last_normal_np = normal_np
            now = time.perf_counter()
            profiler.add("geo", now - stage_t)
            stage_t = now
    
            # 实时位移场 dx, dy（必须基于原始图像 frame，不能用去标记点后的 frame_clean）
            dx_np = np.zeros_like(bg_depth, dtype=np.float32) if bg_depth is not None else None
            dy_np = np.zeros_like(bg_depth, dtype=np.float32) if bg_depth is not None else None
            dx_mm_np = dy_mm_np = None
            marker_flow_vis = None
            if bg_depth is not None:
                try:
    
                    # 用原始 RGB 帧计算标记点位移场
                    dx_np, dy_np = disp_runtime.compute_dx_dy(frame)
                    # dx_np[np.abs(dx_np) < 1.0] = 0  # 强力去噪掩膜
                    # dy_np[np.abs(dy_np) < 1.0] = 0
    
                    # 打印 dx/dy 统计信息，便于选择阈值去除背景噪声
                    def _print_disp_stats(dx, dy, prefix="DISP"):
                        try:
                            if dx is None or dy is None:
                                print(f"{prefix}: dx/dy is None")
                                return
                            # 全图统计
                            dx_min, dx_max = float(np.min(dx)), float(np.max(dx))
                            dy_min, dy_max = float(np.min(dy)), float(np.max(dy))
                            dx_mean, dy_mean = float(np.mean(dx)), float(np.mean(dy))
                            dx_std, dy_std = float(np.std(dx)), float(np.std(dy))
                            dx_p1, dx_p99 = float(np.percentile(dx, 1)), float(np.percentile(dx, 99))
                            dy_p1, dy_p99 = float(np.percentile(dy, 1)), float(np.percentile(dy, 99))
                            print(
                                f"{prefix} FULL dx[min,max,mean,std,p1,p99]=[{dx_min:.4f},{dx_max:.4f},{dx_mean:.4f},{dx_std:.4f},{dx_p1:.4f},{dx_p99:.4f}] \n"
                                f"{prefix} FULL dy[min,max,mean,std,p1,p99]=[{dy_min:.4f},{dy_max:.4f},{dy_mean:.4f},{dy_std:.4f},{dy_p1:.4f},{dy_p99:.4f}]")
                        except Exception:
                            print(f"{prefix}: failed to compute stats")
    
                    # _print_disp_stats(dx_np, dy_np, prefix="DISP_RUNTIME")
    
                    # 将水平位移从像素转换为毫米，保证与训练数据集中 dx, dy 的单位一致
                    if pixmm is not None:
                        dx_mm_np = dx_np * pixmm
                        dy_mm_np = dy_np * pixmm
                    else:
                        dx_mm_np = dx_np
                        dy_mm_np = dy_np
    
                    # 新增：在原图上画标记点位移箭头（类似 offline_perspective_dfs_match 的 origin_with_markers）
                    marker_flow_vis = disp_runtime.draw_marker_flow_on_raw(frame)
                except Exception as e:
                    print(f"Displacement Error: {e}")
            now = time.perf_counter()
            profiler.add("displacement", now - stage_t)
            stage_t = now
    
            # 深度重建及 dz 计算；计算方式与所选力模型版本的预处理配置一致。
            curr_depth = None
            dz_np = None
            try:
                curr_depth = reconstruct_depth_via_poisson(normal_np, depth_prior_path)
                last_depth = curr_depth
    
                if bg_depth is not None:
                    # 复用力模型的统一位移预处理入口，但不做坐标系转换，保留原始 dz 供显示与保存。
                    dx_for_dz = dx_np if dx_np is not None else np.zeros_like(bg_depth, dtype=np.float32)
                    dy_for_dz = dy_np if dy_np is not None else np.zeros_like(bg_depth, dtype=np.float32)
                    _, _, dz_np = _compute_output_displacement_fields(
                        dep_bg_np=bg_depth,
                        dep_raw=curr_depth,
                        dx_pixel=dx_for_dz,
                        dy_pixel=dy_for_dz,
                        pixmm=pixmm,
                        lag_dz=display_lag_dz,
                        use_bg_sub=display_use_bg_sub,
                        convert_dz_to_pixel=False,
                        convert_disp_to_sensor=False,
                    )
                    dz_np = dz_np.astype(np.float32)
                    last_dz = dz_np
            except Exception as e:
                print(f"Depth / dz Error: {e}")
            now = time.perf_counter()
            profiler.add("depth_dz", now - stage_t)
            stage_t = now
    
            # 力场预测
            force_map_np = None
            fx_map = fy_map = fz_map = None
            total_force = 0.0
            if enable_force and force_model and bg_depth is not None and curr_depth is not None and dx_np is not None and dy_np is not None:
                try:
                    # Use the preprocessing config saved with the force model version.
                    dx_input, dy_input, dz_input = force_model.prepare_displacement_inputs(
                        bg_depth=bg_depth,
                        curr_depth=curr_depth,
                        dx_pixel=dx_np,
                        dy_pixel=dy_np,
                    )
                    force_map_np = force_model.predict_full(dz_input, dx_input, dy_input)  # [3,H,W]
    
                    fx_map, fy_map, fz_map = force_map_np
                    total_force = np.sum(force_map_np, axis=(1, 2))  # 每个分量的总力
                    last_force_map = force_map_np
                except Exception as e:
                    print(f"Force Error: {e}")
            now = time.perf_counter()
            profiler.add("force", now - stage_t)
            stage_t = now
    
            # ================== 可视化 ==================
            scale = 0.6
            vis_w = int(W_img * scale)
            vis_h = int(H_img * scale)
            vis_size = (vis_w, vis_h)
    
            # 检查尺寸是否匹配，若不匹配（下采样）则放大
            fx_map_3d = fy_map_3d = fz_map_3d = None
            if fx_map is not None and fy_map is not None and fz_map is not None and curr_depth is not None:
                if fx_map.shape != curr_depth.shape:
                    H_full, W_full = curr_depth.shape
                    fx_map_3d = cv2.resize(fx_map, (W_full, H_full), interpolation=cv2.INTER_LINEAR)
                    fy_map_3d = cv2.resize(fy_map, (W_full, H_full), interpolation=cv2.INTER_LINEAR)
                    fz_map_3d = cv2.resize(fz_map, (W_full, H_full), interpolation=cv2.INTER_LINEAR)
                else:
                    fx_map_3d = fx_map
                    fy_map_3d = fy_map
                    fz_map_3d = fz_map
    
    
            # 窗口1：原图（去标记点）、法向、深度
            img_vis = cv2.resize(frame, vis_size)
            norm_vis = visualize_normal(normal_np, vis_size, pixmm=pixmm)
            depth_vis_full = visualize_map_with_bar(
                curr_depth, vis_size, cv2.COLORMAP_JET, unit="mm", scale_factor=1.0
            )
            # 仅用于保存的“无 colorbar 深度图”
            depth_vis_nobar = visualize_map_core_only(
                curr_depth, vis_size, cv2.COLORMAP_JET, scale_factor=1.0
            )
            win1_img = np.hstack((img_vis, norm_vis, depth_vis_full))
            cv2.imshow("Camera & Geometry", win1_img)
    
            # 新增：显示标记点 mask 和 inpainted 结果
            if marker_mask is not None:
                marker_mask_r = cv2.resize(marker_mask, vis_size)
                cv2.imshow("Marker Mask", marker_mask_r)
            if frame_clean is not None:
                frame_clean_r = cv2.resize(frame_clean, vis_size)
                cv2.imshow("Inpainted Frame", frame_clean_r)
    
            # 窗口2：只显示深度差 dz（不再叠加 Fx_sum/Fy_sum/Fz_sum 文本）
            diff_vis_full = visualize_map_with_bar(
                dz_np, vis_size, cv2.COLORMAP_JET, unit="mm/sub", scale_factor=1.0
            )
            # 仅用于保存的“无 colorbar 深度差图”
            dz_vis_nobar = visualize_map_core_only(
                dz_np, vis_size, cv2.COLORMAP_JET, scale_factor=1.0
            )
            win2_img = diff_vis_full
            cv2.imshow("Depth Diff (dz)", win2_img)
    
            # 新增：dx、dy 分量的伪彩色可视化（显示时带 colorbar，由外部组合控制，这里用无 bar 图像）
            dx_vis_nobar, dy_vis_nobar = visualize_dx_dy_colormaps(dx_np, dy_np, vis_size, scale_factor=1.0)
            if dx_vis_nobar is not None:
                cv2.imshow("Displacement dx (no-bar)", dx_vis_nobar)
            if dy_vis_nobar is not None:
                cv2.imshow("Displacement dy (no-bar)", dy_vis_nobar)
    
            # 新增窗口：分别显示 Fx / Fy / Fz 的热力图（和可视化法向量类似，更直观）
            force_maps_vis = None
            force_fx_nobar = force_fy_nobar = force_fz_nobar = None
            if fx_map is not None and fy_map is not None and fz_map is not None:
                force_maps_vis = visualize_force_component_maps(
                    fx_map, fy_map, fz_map,
                    out_size=vis_size,
                )
    
                # 在组合图上标注 Fx / Fy / Fz 名称，方便区分对应区域
                h_fc, w_fc = force_maps_vis.shape[:2]
                single_w = w_fc // 3
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                color = (255, 255, 255)
    
                cv2.putText(force_maps_vis, 'Fx', (int(single_w * 0.05), 30), font, font_scale, color, thickness)
                cv2.putText(force_maps_vis, 'Fy', (int(single_w * 1.05), 30), font, font_scale, color, thickness)
                cv2.putText(force_maps_vis, 'Fz', (int(single_w * 2.05), 30), font, font_scale, color, thickness)
    
                # 在每一列下方显示该分量的总力值（或局部统计值），便于定量对比
                try:
                    # total_force 在外面计算：np.sum(force_map_np, axis=(1,2))
                    if isinstance(total_force, (list, tuple, np.ndarray)) and len(total_force) >= 3:
                        fx_sum = float(total_force[0])
                        fy_sum = float(total_force[1])
                        fz_sum = float(total_force[2])
                    else:
                        # 兜底：直接从 map 求和（单位与训练一致）
                        fx_sum = float(np.nansum(fx_map)) if fx_map is not None else 0.0
                        fy_sum = float(np.nansum(fy_map)) if fy_map is not None else 0.0
                        fz_sum = float(np.nansum(fz_map)) if fz_map is not None else 0.0
    
                    val_font_scale = 0.6
                    val_thickness = 2
                    # 文本位置：在每列标题下方稍微偏下
                    y_pos = 60
                    cv2.putText(force_maps_vis, f"Fx_sum: {fx_sum:.3f} N", (int(single_w * 0.05), y_pos), font, val_font_scale, color, val_thickness)
                    cv2.putText(force_maps_vis, f"Fy_sum: {fy_sum:.3f} N", (int(single_w * 1.05), y_pos), font, val_font_scale, color, val_thickness)
                    cv2.putText(force_maps_vis, f"Fz_sum: {fz_sum:.3f} N", (int(single_w * 2.05), y_pos), font, val_font_scale, color, val_thickness)
                except Exception:
                    # 如果任何计算失败，不阻塞主循环
                    pass
    
                cv2.imshow("Force Estimation", force_maps_vis)
    
                # 仅用于保存的、每个分量对应的“无 colorbar 力场图”
                force_fx_nobar = visualize_map_core_only(
                    fx_map, vis_size, cv2.COLORMAP_INFERNO, scale_factor=1.0
                )
                force_fy_nobar = visualize_map_core_only(
                    fy_map, vis_size, cv2.COLORMAP_INFERNO, scale_factor=1.0
                )
                force_fz_nobar = visualize_map_core_only(
                    fz_map, vis_size, cv2.COLORMAP_INFERNO, scale_factor=1.0
                )
            else:
                force_maps_vis = np.zeros((vis_size[1], vis_size[0], 3), dtype=np.uint8)
                cv2.putText(force_maps_vis, "Force estimation unavailable", (20, vis_size[1] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Force Estimation", force_maps_vis)
    
            # 窗口3：位移箭头 & 力箭头（保留作为辅助，直观上主要看上面的热力图窗口）
            force_arrow = None
            if dx_np is not None and dy_np is not None:
                disp_arrow = visualize_displacement_arrows(bg_rgb_clean, dx_np, dy_np, dz_np, step=25)
            else:
                disp_arrow = bg_rgb_clean.copy() if bg_rgb_clean is not None else frame_clean.copy()
            if fx_map is not None and fy_map is not None and fz_map is not None:
                force_arrow = visualize_force_composite(fx_map, fy_map, fz_map, step=25)
            else:
                force_arrow = np.zeros_like(disp_arrow)
            disp_arrow_r = cv2.resize(disp_arrow, vis_size)
            force_arrow_r = cv2.resize(force_arrow, vis_size)
            win3_img = np.hstack((disp_arrow_r, force_arrow_r))
            # cv2.imshow("3D Displacement & Force (arrows)", win3_img)
    
            # 新增窗口：原图上的标记点位移箭头
            if marker_flow_vis is not None:
                marker_flow_vis_r = cv2.resize(marker_flow_vis, vis_size)
                cv2.imshow("Marker Flow on Raw", marker_flow_vis_r)
    
            # 新增：实时插值出的全图位移场可视化（箭头 + 幅值伪彩）
            if ENABLE_RT_DISPLACEMENT_VIS and dx_np is not None and dy_np is not None:
                # 1) 在原始相机图像上画稀疏箭头（来自致密位移场 du,dv）
                dense_disp_arrow = disp_runtime.visualize_dense_displacement_on_raw(frame, step=25, scale=5.0, min_rel=0.05)
                dense_disp_arrow_r = cv2.resize(dense_disp_arrow, vis_size)
                cv2.imshow("RT Dense Displacement (arrows)", dense_disp_arrow_r)
    
                # 2) 位移幅值 |d| 的伪彩色图
                dense_mag = disp_runtime.visualize_displacement_magnitude_colormap()
                if dense_mag is not None:
                    dense_mag_r = cv2.resize(dense_mag, vis_size)
                    cv2.imshow("RT Dense Displacement |d|", dense_mag_r)
            now = time.perf_counter()
            profiler.add("visualize", now - stage_t)
            stage_t = now
    
            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()
            profiler.add("waitkey", now - stage_t)
            profiler.finish_frame(now - frame_t0)
    
            if key == ord('b'):
                print("Re-setting background to reduce drift...")
                bg_rgb_clean = frame_clean.copy()
                geo_model.load_bg(bg_rgb_clean)
                img_t = TF.to_tensor(bg_rgb_clean).unsqueeze(0).to(device)
                norm_t, _ = geo_model.predict(img_t)
                bg_norm = norm_t.detach().cpu().numpy()
                try:
                    bg_depth = reconstruct_depth_via_poisson(bg_norm, depth_prior_path)
                    if enable_force and force_model:
                        force_model.set_background(bg_depth)
                    # 位移模块背景：必须用带标记点的原始背景图
                    disp_runtime.set_background(frame)
                    print("Background reset done.")
                except Exception as e:
                    print(f"BG Reset Error: {e}")
                continue # Skip the rest of this frame's inference to avoid glitches
    
            if key in (13, 10):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Save under force_logs/version_{force_model_version}/{timestamp}
                save_folder = os.path.join("force_logs/HyperNet_Run", f"version_{force_model_version}", ts)
                os.makedirs(save_folder, exist_ok=True)
                print(f"Saving to {save_folder} ...")
    
                # Prepare full-resolution visuals for saving (original image size)
                full_size = (W_img, H_img)  # (width, height) for cv2.resize
                raw_full = frame.copy()
                clean_full = frame_clean.copy()
    
                # full-res normal visualization
                try:
                    norm_vis_full = visualize_normal(normal_np, full_size, pixmm=pixmm)
                except Exception:
                    norm_vis_full = visualize_normal(normal_np, vis_size, pixmm=pixmm)
    
                # full-res depth visualizations (with and without colorbar)
                try:
                    depth_vis_full_full = visualize_map_with_bar(curr_depth, full_size, cv2.COLORMAP_JET, unit="mm", scale_factor=1.0)
                except Exception:
                    depth_vis_full_full = visualize_map_with_bar(curr_depth, vis_size, cv2.COLORMAP_JET, unit="mm", scale_factor=1.0)
                try:
                    depth_vis_nobar_full = visualize_map_core_only(curr_depth, full_size, cv2.COLORMAP_JET, scale_factor=1.0)
                except Exception:
                    depth_vis_nobar_full = visualize_map_core_only(curr_depth, vis_size, cv2.COLORMAP_JET, scale_factor=1.0)
    
                # full-res dz visualization (no colorbar)
                try:
                    dz_vis_nobar_full = visualize_map_core_only(dz_np, full_size, cv2.COLORMAP_JET, scale_factor=1.0)
                except Exception:
                    dz_vis_nobar_full = visualize_map_core_only(dz_np, vis_size, cv2.COLORMAP_JET, scale_factor=1.0)
    
                # full-res dx/dy colormaps
                try:
                    dx_vis_full, dy_vis_full = visualize_dx_dy_colormaps(dx_np, dy_np, full_size, scale_factor=1.0)
                except Exception:
                    dx_vis_full, dy_vis_full = visualize_dx_dy_colormaps(dx_np, dy_np, vis_size, scale_factor=1.0)
    
                # full-res force component maps
                try:
                    force_maps_vis_full = visualize_force_component_maps(fx_map, fy_map, fz_map, out_size=full_size)
                except Exception:
                    force_maps_vis_full = force_maps_vis
                try:
                    force_fx_nobar_full = visualize_map_core_only(fx_map, full_size, cv2.COLORMAP_INFERNO, scale_factor=1.0)
                    force_fy_nobar_full = visualize_map_core_only(fy_map, full_size, cv2.COLORMAP_INFERNO, scale_factor=1.0)
                    force_fz_nobar_full = visualize_map_core_only(fz_map, full_size, cv2.COLORMAP_INFERNO, scale_factor=1.0)
                except Exception:
                    force_fx_nobar_full = force_fx_nobar
                    force_fy_nobar_full = force_fy_nobar
                    force_fz_nobar_full = force_fz_nobar
    
                # disp_arrow and marker_flow_vis are already full-resolution images
                disp_arrow_full = disp_arrow
                marker_flow_full = marker_flow_vis
    
                # 1) 基础原图（去标记点）
                cv2.imwrite(os.path.join(save_folder, f"img_clean_{ts}.png"), clean_full)
    
                # 2) win1_geo 中的三个子图分别保存（深度图使用无 colorbar 版本)
                cv2.imwrite(os.path.join(save_folder, f"raw_{ts}.png"), raw_full)
                cv2.imwrite(os.path.join(save_folder, f"normal_{ts}.png"), norm_vis_full)
                cv2.imwrite(os.path.join(save_folder, f"depth_{ts}.png"), depth_vis_nobar_full)
    
                # 如仍需保留整体拼接图（带 colorbar，仅用于参考），可选保存：
                try:
                    win1_full = np.hstack((raw_full, norm_vis_full, depth_vis_full_full))
                    cv2.imwrite(os.path.join(save_folder, f"win1_geo_{ts}.png"), win1_full)
                except Exception:
                    # fallback to scaled composite
                    cv2.imwrite(os.path.join(save_folder, f"win1_geo_{ts}.png"), win1_img)
    
                # 3) win2_depth_force：只保存深度差 dz 图的无 colorbar 版本
                cv2.imwrite(os.path.join(save_folder, f"dz_map_{ts}.png"), dz_vis_nobar_full)
    
                # 4) win3_disp_force：只保存位移场图（左半部分 disp_arrow)
                cv2.imwrite(os.path.join(save_folder, f"disp_field_{ts}.png"), disp_arrow_full)
                try:
                    # attempt to build full-resolution win3 by concatenating full disp and full force arrow if available
                    if force_arrow is not None:
                        win3_full = np.hstack((disp_arrow_full, force_arrow))
                        cv2.imwrite(os.path.join(save_folder, f"win3_disp_force_{ts}.png"), win3_full)
                    else:
                        cv2.imwrite(os.path.join(save_folder, f"win3_disp_force_{ts}.png"), win3_img)
                except Exception:
                    cv2.imwrite(os.path.join(save_folder, f"win3_disp_force_{ts}.png"), win3_img)
    
                # 5) 新增：保存 dx、dy 位移伪彩色图（无 colorbar）
                if dx_vis_full is not None:
                    cv2.imwrite(os.path.join(save_folder, f"dx_map_{ts}.png"), dx_vis_full)
                if dy_vis_full is not None:
                    cv2.imwrite(os.path.join(save_folder, f"dy_map_{ts}.png"), dy_vis_full)
    
                # 5.1) 新增：保存 marker flow 可视化 (原始与缩放版)，仅在存在时保存
                if marker_flow_vis is not None:
                    # 保存原始尺寸的 marker flow（full-res）
                    try:
                        cv2.imwrite(os.path.join(save_folder, f"marker_flow_raw_{ts}.png"), marker_flow_full)
                    except Exception as e:
                        print(f"Warning: failed to save marker_flow_raw: {e}")
    
                    # 保存缩放到可视化尺寸的预览（与屏幕一致），失败不影响其它保存
                    try:
                        marker_flow_vis_r = cv2.resize(marker_flow_vis, vis_size)
                        cv2.imwrite(os.path.join(save_folder, f"marker_flow_{ts}.png"), marker_flow_vis_r)
                    except Exception as e:
                        print(f"Warning: failed to resize/save marker_flow preview: {e}")
    
                # 6) 力分布相关图像单独保存（若存在力场），使用无 colorbar 版本
                if force_fx_nobar_full is not None and force_fy_nobar_full is not None and force_fz_nobar_full is not None:
                    cv2.imwrite(os.path.join(save_folder, f"Fx_map_{ts}.png"), force_fx_nobar_full)
                    cv2.imwrite(os.path.join(save_folder, f"Fy_map_{ts}.png"), force_fy_nobar_full)
                    cv2.imwrite(os.path.join(save_folder, f"Fz_map_{ts}.png"), force_fz_nobar_full)
                # 如需组合参考图（含 colorbar 与标签），可以继续保存：
                if force_maps_vis_full is not None:
                    cv2.imwrite(os.path.join(save_folder, f"force_components_{ts}.png"), force_maps_vis_full)
    
                if force_arrow is not None:
                    cv2.imwrite(os.path.join(save_folder, f"force_arrow_{ts}.png"), force_arrow)
    
                # 7) 数值数据
                if last_normal_np is not None:
                    np.save(os.path.join(save_folder, f"normal_{ts}.npy"), last_normal_np)
                if last_depth is not None:
                    np.save(os.path.join(save_folder, f"depth_arr_{ts}.npy"), last_depth)
                if last_dz is not None:
                    np.save(os.path.join(save_folder, f"dz_{ts}.npy"), last_dz)
                if last_force_map is not None:
                    np.save(os.path.join(save_folder, f"force_{ts}.npy"), last_force_map)
    
                # 8) 3D 力分布可视化图保存（使用当前帧的 depth & force ）
                try:
                    if curr_depth is not None and fx_map is not None and fy_map is not None and fz_map is not None:
                        fig_path = os.path.join(save_folder, f"force_3d_{ts}.png")
                        visualize_3d_force_distribution(
                            depth_map=curr_depth,
                            fx=fx_map_3d,
                            fy=fy_map_3d,
                            fz=fz_map_3d,
                            step=5,
                            scale_arrow=300.0,
                            title="3D Force Vectors on Sensor Surface (Realtime)",
                            save_path=fig_path,
                        )
                except Exception as e:
                    print(f"Warning: failed to save 3D force figure: {e}")
    
                print(f"Saved to: {save_folder}")
    
            # Adjust params for displacement marker detection
            if key == ord('['):
                V_BLOCKSIZE_DIS = max(3, V_BLOCKSIZE_DIS - 2)
                print(f"V_BLOCKSIZE_DIS updated to {V_BLOCKSIZE_DIS}")
            if key == ord(']'):
                V_BLOCKSIZE_DIS += 2
                print(f"V_BLOCKSIZE_DIS updated to {V_BLOCKSIZE_DIS}")
            if key == ord('-'):
                V_C_DIS = max(0, V_C_DIS - 1)
                print(f"V_C_DIS updated to {V_C_DIS}")
            if key == ord('='):
                V_C_DIS += 1
                print(f"V_C_DIS updated to {V_C_DIS}")
            if key == ord('n'):
                CIRCULARITY_MIN = min(1.0, CIRCULARITY_MIN + 0.01)
                print(f"CIRCULARITY_MIN updated to {CIRCULARITY_MIN:.2f}")
            if key == ord('m'):
                CIRCULARITY_MIN = max(0.0, CIRCULARITY_MIN - 0.01)
                print(f"CIRCULARITY_MIN updated to {CIRCULARITY_MIN:.2f}")
    
            if key == ord('q'):
                break
    
    finally:
        if latest_capture is not None:
            latest_capture.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        camera_index=0, # windows
        # camera_index='/dev/video0', # linux
        # model_version=56,   # 平面三指标记点1号
        # model_version=62,  # 曲面三指1号
        model_version=118,  # 曲面三指2号
        # depth_prior_path='dep_p_20251125_221351.npy',
        # depth_prior_path='dep_p_20251209_094825.npy',   # 曲面三指标记点1号
        depth_prior_path='depth_prior/dep_p_20260407_181422.npy',   # 曲面三指2号
        save_dir="output_realtime",
        enable_force=True,
        # force_model_version=9,  # 平面三指标记点1号
        # force_model_version=40,  # 曲面三指标记点1号
        # force_model_version=74,  # 曲面三指2号，lag_dz=True
        force_model_version=76,  # 曲面三指2号，lag_dz=False
        pixmm=0.034,  # 曲面三指
    )
