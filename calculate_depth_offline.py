import os
import cv2
import numpy as np
import glob
import time
from scipy.signal import convolve2d
from infer_geo_offline import Geo_Model
from calculate_depth_single import Normal_Integration
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from gel_marker_inpaint_viewer import generate_marker_mask, inpaint_markers

# Open3D 可视化支持（可选）
try:
    import open3d as o3d
    _HAS_OPEN3D = True
except Exception:
    o3d = None
    _HAS_OPEN3D = False


# ========================= 可视化与误差统计开关 =========================
# 开启后：每个样本弹出多个窗口，按任意键进入下一个样本
VISUALIZE_EACH_SAMPLE = True



def normal_to_px_py(normal: np.ndarray, eps: float = 1e-6):
    """把法线转换为梯度 px, py。

    在本仓库的定义里（参考 fast_poisson_from_normals / 法线->深度梯度关系）：
      p = ∂z/∂x = -nx/nz
      q = ∂z/∂y = -ny/nz

    返回:
      px, py: (H,W) float32
    """
    if normal is None:
        return None, None
    n = normal
    if n.ndim == 3 and n.shape[0] == 3 and n.shape[2] != 3:
        n = np.transpose(n, (1, 2, 0))
    n = n.astype(np.float32)
    nlen = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.clip(nlen, eps, None)
    nz = np.clip(n[..., 2], eps, None)
    px = (-n[..., 0] / nz).astype(np.float32)
    py = (-n[..., 1] / nz).astype(np.float32)
    return px, py


def compute_px_py_error(pred_normal: np.ndarray, gt_normal: np.ndarray, mask: np.ndarray | None = None):
    """计算预测法线与GT法线对应的梯度(px,py)误差。

    返回:
      (px_mae, py_mae, px_rmse, py_rmse)
    """
    if pred_normal is None or gt_normal is None:
        return None
    px_p, py_p = normal_to_px_py(pred_normal)
    px_g, py_g = normal_to_px_py(gt_normal)
    if px_p is None or px_g is None:
        return None

    valid = np.isfinite(px_p) & np.isfinite(py_p) & np.isfinite(px_g) & np.isfinite(py_g)
    if mask is not None:
        valid &= (mask > 0)
    if not np.any(valid):
        return None

    dpx = (px_p - px_g)[valid]
    dpy = (py_p - py_g)[valid]
    px_mae = float(np.mean(np.abs(dpx)))
    py_mae = float(np.mean(np.abs(dpy)))
    px_rmse = float(np.sqrt(np.mean(dpx ** 2)))
    py_rmse = float(np.sqrt(np.mean(dpy ** 2)))
    return px_mae, py_mae, px_rmse, py_rmse


def _to_u8_for_show(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imgf = img.astype(np.float32)
    mn, mx = float(np.nanmin(imgf)), float(np.nanmax(imgf))
    if mx <= mn:
        mx = mn + 1e-6
    out = (255.0 * (imgf - mn) / (mx - mn)).clip(0, 255).astype(np.uint8)
    return out


def resize_to_height(img: np.ndarray, target_height: int = 480) -> np.ndarray:
    """将图像高度调整为 target_height，保持宽高比。"""
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == target_height:
        return img
    aspect = w / h
    new_w = int(target_height * aspect)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)


def normal_to_vis_bgr(normal: np.ndarray) -> np.ndarray:
    if normal is None:
        return None
    n = normal
    if n.ndim == 3 and n.shape[0] == 3 and n.shape[2] != 3:
        n = np.transpose(n, (1, 2, 0))
    n = np.nan_to_num(n.astype(np.float32))
    n[..., 2] /= 10
    nlen = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.clip(nlen, 1e-6, None)
    
    vis = ((n + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def depth_to_colormap(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if depth is None:
        return None
    d = depth.astype(np.float32)
    valid = np.isfinite(d)
    if mask is not None:
        valid &= (mask > 0)
    if not np.any(valid):
        norm = np.zeros_like(d, dtype=np.float32)
    else:
        mn = float(np.nanmin(d[valid]))
        mx = float(np.nanmax(d[valid]))
        if mx <= mn:
            mx = mn + 1e-6
        norm = (d - mn) / (mx - mn)
        norm = np.clip(norm, 0.0, 1.0)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cm


def show_six_windows_and_wait(
    rgb: np.ndarray | None,
    nir: np.ndarray | None,
    pred_normal: np.ndarray | None,
    gt_normal: np.ndarray | None,
    depth_gt: np.ndarray | None,
    depth_from_gt_normal: np.ndarray | None,
    depth_from_pred_normal: np.ndarray | None,
    sample_id: str,
    pxpy_err: tuple[float, float, float, float] | None,
):
    """每张图一个窗口，任意键进入下一样本。"""
    # 调整所有图像高度为480
    rgb = resize_to_height(rgb)
    nir = resize_to_height(nir)

    # 1) 原图
    if rgb is not None:
        cv2.imshow("RGB", rgb)
    if nir is not None:
        cv2.imshow("NIR", _to_u8_for_show(nir))

    # 2) 法向 (预测)
    pn_vis = normal_to_vis_bgr(pred_normal)
    pn_vis = resize_to_height(pn_vis)
    if pn_vis is not None:
        # 显示 ID
        cv2.putText(pn_vis, f"ID={sample_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if pxpy_err is not None:
            px_mae, py_mae, px_rmse, py_rmse = pxpy_err
            cv2.putText(pn_vis, f"px_mae={px_mae:.4f} py_mae={py_mae:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.putText(pn_vis, f"px_rmse={px_rmse:.4f} py_rmse={py_rmse:.4f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
        cv2.imshow("Normal Pred", pn_vis)

    # 法向 (GT) - 已隐藏
    # gn_vis = normal_to_vis_bgr(gt_normal)
    # gn_vis = resize_to_height(gn_vis)
    # if gn_vis is not None:
    #     cv2.imshow("Normal GT", gn_vis)

    # 3) 深度 (GT)
    if depth_gt is not None:
        depth_gt_vis = depth_to_colormap(depth_gt)
        depth_gt_vis = resize_to_height(depth_gt_vis)
        cv2.imshow("Depth GT", depth_gt_vis)

    # Depth Recon from GT Normal - 已隐藏
    # if depth_from_gt_normal is not None:
        # ...

    # 4) 深度重建结果 (from Pred Normal)
    if depth_from_pred_normal is not None:
        depth_pred_recon_vis = depth_to_colormap(depth_from_pred_normal)
        depth_pred_recon_vis = resize_to_height(depth_pred_recon_vis)
        cv2.imshow("Depth Recon (Pred)", depth_pred_recon_vis)

    key = cv2.waitKey(0)
    # ESC key to exit
    if key == 27:
        print("[用户中止] ESC按下，正在退出...")
        import sys
        sys.exit(0)
    # 不 destroyAllWindows，避免闪烁；下一样本会覆盖同名窗口内容


def depth_to_point_cloud(depth, fx, fy, cx, cy, mask=None):
    """
    将深度图转换为点云（Nx3 numpy array）。
    depth: 2D float ndarray（单位与重建一致，含 NaN 视为无效）
    fx,fy: 焦距（像素）
    cx,cy: 主点（像素坐标）
    mask: 可选布尔数组，仅在 mask True 的像素处保留点
    返回 Nx3 的 float32 numpy 数组（可能为空）
    """
    H, W = depth.shape
    valid = np.isfinite(depth)
    # if mask is not None:
    #     valid = valid & (mask.astype(bool))
    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - float(cx)) * z / float(fx)
    y = (ys.astype(np.float32) - float(cy)) * z / float(fy)
    pts = np.stack([x, y, z], axis=1)
    return pts


def visualize_point_cloud_points(points, window_name='PointCloud'):
    """
    使用 Open3D 显示点云；若没有 Open3D，则抛出异常以触发回退逻辑。
    points: Nx3 numpy
    注意：此函数是阻塞的，窗口关闭前程序会暂停。
    """
    if points.size == 0:
        raise RuntimeError('点云为空，无法显示')
    if not _HAS_OPEN3D:
        raise RuntimeError('Open3D 未安装')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 简单弹窗显示（阻塞，关闭窗口后继续）
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def save_point_cloud_ply(points, out_path):
    """
    将点云保存为 ply（优先使用 Open3D，否则写入简单 ASCII PLY）。
    """
    if points.size == 0:
        # 写入空文件以示意
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex 0\nend_header\n')
        return
    if _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(out_path, pcd)
        return
    # ASCII ply 回退
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


# ------------------ 可选 DST 初始化（鲁棒） ------------------
_USE_CUPY = False
_USE_SCIPY = False
_CP = None


def _as_real_array(x, xp):
    arr = xp.asarray(x)
    if arr.dtype.kind == 'c':
        raise TypeError('DST-I only supports real-valued inputs')
    if arr.dtype.kind in 'bui':
        return arr.astype(xp.float32, copy=False)
    return arr.astype(xp.promote_types(arr.dtype, xp.float32), copy=False)


def _dst1_fft_backend(x, xp, axis=-1, norm='ortho'):
    """DST-I via odd extension FFT. Matches scipy.fft.dst(type=1)."""
    if norm not in (None, 'backward', 'ortho'):
        raise ValueError('Only norm=None/"backward" or "ortho" is supported for DST-I')

    arr = _as_real_array(x, xp)
    axis = axis % arr.ndim
    n = arr.shape[axis]
    if n < 1:
        raise ValueError('DST-I input length must be at least 1')

    arr = xp.moveaxis(arr, axis, -1)
    ext_shape = arr.shape[:-1] + (2 * (n + 1),)
    ext = xp.zeros(ext_shape, dtype=arr.dtype)
    ext[..., 1:n + 1] = arr
    ext[..., n + 2:] = -arr[..., ::-1]

    spectrum = xp.fft.fft(ext, axis=-1)
    out = -spectrum.imag[..., 1:n + 1]
    if norm == 'ortho':
        out = out / xp.sqrt(2.0 * (n + 1))
    out = xp.moveaxis(out, -1, axis)
    return out.astype(arr.dtype, copy=False)


def _idst1_fft_backend(x, xp, axis=-1, norm='ortho'):
    if norm != 'ortho':
        raise ValueError('This project only uses norm="ortho" for IDST-I')
    return _dst1_fft_backend(x, xp, axis=axis, norm=norm)


def _solve_poisson_inner_dst(inner, xp):
    """Solve the inner Poisson system with separable orthonormal DST-I."""
    n_rows, n_cols = inner.shape
    freq = dst(dst(inner, norm='ortho').T, norm='ortho').T

    col_ids = xp.arange(1, n_cols + 1, dtype=inner.dtype)
    row_ids = xp.arange(1, n_rows + 1, dtype=inner.dtype)
    jj, ii = xp.meshgrid(col_ids, row_ids, indexing='xy')
    denom = (
        (2 * xp.cos(xp.pi * ii / (n_rows + 1)) - 2)
        + (2 * xp.cos(xp.pi * jj / (n_cols + 1)) - 2)
    ).astype(inner.dtype, copy=False)
    denom[denom == 0] = xp.inf

    spectral_depth = freq / denom
    return idst(idst(spectral_depth, norm='ortho').T, norm='ortho').T


def visualize_sample(rgb_img, pred_normal_np, depth_gt, reconstructed_depth, base_name, mode="pred", wait_ms=100):
    """
    可视化单帧：原图、法向量、真值深度、预测深度。
    修改：使用固定窗口名以支持连续播放，wait_ms 控制播放速度。
    """
    # 1. 规范法线到 (H,W,3)
    if pred_normal_np is None:
        normal_vis = np.zeros_like(rgb_img)
    else:
        if pred_normal_np.ndim == 3 and pred_normal_np.shape[0] == 3 and pred_normal_np.shape[2] != 3:
            normal = np.transpose(pred_normal_np, (1, 2, 0))
        else:
            normal = pred_normal_np
        # 防护
        normal = np.nan_to_num(normal)
        if normal.shape[-1] != 3:
            normal = np.stack([normal] * 3, axis=-1)

        # 增大nx,ny对比度
        normal_vis = normal.copy()
        normal_vis[..., :2] /= 0.1

        # 映射到 [0,255] 并转换为 BGR
        normal_vis = ((normal_vis + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        try:
            normal_vis = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    # 2. 深度伪色化辅助
    def depth_to_colormap(d):
        if d is None:
            return np.zeros_like(rgb_img)
        d = np.array(d, dtype=np.float32)
        # 掐头去尾并归一化
        finite = np.isfinite(d)
        if not finite.any():
            norm = np.zeros_like(d)
        else:
            mn = np.nanmin(d[finite])
            mx = np.nanmax(d[finite])
            if mx <= mn:
                mx = mn + 1e-6
            norm = (d - mn) / (mx - mn)
            norm = np.clip(norm, 0.0, 1.0)
        cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cmap

    gt_vis = depth_to_colormap(depth_gt)
    pred_vis = depth_to_colormap(reconstructed_depth)
    error = reconstructed_depth - depth_gt
    error_vis = depth_to_colormap(reconstructed_depth - depth_gt)

    # 3. 将预测深度填充或裁剪到与原图相同大小
    h, w = rgb_img.shape[:2]

    def pad_to(img, H, W):
        ih, iw = img.shape[:2]
        pad_h = max(0, H - ih)
        pad_w = max(0, W - iw)
        if pad_h == 0 and pad_w == 0:
            return img
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    normal_vis = pad_to(normal_vis, h, w)
    gt_vis = pad_to(gt_vis, h, w)
    pred_vis = pad_to(pred_vis, h, w)

    # 4. 添加标签
    def add_label(im, text):
        im = im.copy()
        cv2.putText(im, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        return im

    img_display = add_label(rgb_img, f"Original: {base_name}")
    normal_display = add_label(normal_vis, f"{mode} Normal")
    gt_display = add_label(gt_vis, "Ground Truth Depth")
    pred_display = add_label(pred_vis, f"Reconstructed Depth from {mode} Normal")
    # error_display = add_label(error_vis, "Depth Error (Pred - GT)")

    # 5. 拼接显示（2x2）
    top = np.hstack((img_display, normal_display))
    bottom = np.hstack((gt_display, pred_display))
    combined = np.vstack((top, bottom))

    # 使用固定窗口名称，实现内容刷新而不是新建窗口
    window_name = "Reconstruction Result Visualization"
    cv2.imshow(window_name, combined)

    # 自动等待 wait_ms 毫秒，然后继续。如果为 0 则无限等待。
    key = cv2.waitKey(wait_ms)
    return key


def _init_transforms():
    global _USE_CUPY, _USE_SCIPY, _CP, dst, idst
    dst = idst = None
    _CP = None

    # 1) 尝试 CuPy + type=1
    try:
        import cupy as cp
        from cupyx.scipy.fft import dst as cp_dst, idst as cp_idst

        # 直接探测是否支持 type=1
        try:
            _ = cp_dst(cp.ones((4, 4), dtype=cp.float32), type=1, norm='ortho')
            _ = cp_idst(cp.ones((4, 4), dtype=cp.float32), type=1, norm='ortho')

            def dst(x, norm='ortho'):
                return cp_dst(x, type=1, norm=norm)

            def idst(x, norm='ortho'):
                return cp_idst(x, type=1, norm=norm)

            _USE_CUPY = True
            print("[DST] 使用 CuPy DST-I")
            return
        except Exception:
            print("[DST] CuPy 不支持 DST-I，回退 CPU")
            _USE_CUPY = False
    except Exception:
        _USE_CUPY = False

    # 2) 尝试 SciPy FFT (多数版本支持 type=1)
    try:
        from scipy.fft import dst as sp_dst, idst as sp_idst
        # 探测
        _ = sp_dst(np.ones((4, 4), dtype=np.float32), type=1, norm='ortho')
        _ = sp_idst(np.ones((4, 4), dtype=np.float32), type=1, norm='ortho')

        def dst(x, norm='ortho'):
            return sp_dst(x, type=1, norm=norm)

        def idst(x, norm='ortho'):
            return sp_idst(x, type=1, norm=norm)

        _USE_SCIPY = True
        print("[DST] 使用 SciPy DST-I")
        return
    except Exception:
        _USE_SCIPY = False
        print("[DST] SciPy 不支持 DST-I，继续回退纯 NumPy")

    # 3) 纯 NumPy DST-I 实现（慢，但可用）
    def _dst1_numpy(x):
        # x: (H,W) → DST-I 沿两维分别做
        X = np.asarray(x, dtype=np.float32)
        H, W = X.shape

        # 逐维可分离
        # 先沿列做，再沿行做
        def one_dim_dst1(v):
            n = v.shape[0]
            # 构造扩展数组长度 n+2
            ext = np.zeros(n + 2, dtype=np.float32)
            ext[1:-1] = v
            F = np.fft.rfft(ext)
            # DST-I 结果 (k=1..n) 为 -Im(F[1:n+1])
            return -F.imag[1:n + 1]

        # 列方向
        tmp = np.stack([one_dim_dst1(X[:, j]) for j in range(W)], axis=1)
        # 行方向
        out = np.stack([one_dim_dst1(tmp[i, :]) for i in range(tmp.shape[0])], axis=0)
        # 归一化：与 scipy norm='ortho' 对齐
        # 正交因子：2 / sqrt((n+1)) * 2 / sqrt((m+1)) = 4 / sqrt((n+1)(m+1))
        n, m = H, W
        factor = 4.0 / np.sqrt((n + 1) * (m + 1))
        return out * factor

    def _idst1_numpy(Xc):
        # 逆变换与正变换同形式（DST-I 自反），只需再除以相同归一化因子
        n, m = Xc.shape
        factor = 4.0 / np.sqrt((n + 1) * (m + 1))
        return _dst1_numpy(Xc / factor / factor)  # 近似（两次因子抵消）；为简洁可再优化

    def dst(x, norm='ortho'):
        return _dst1_numpy(x)

    def idst(x, norm='ortho'):
        return _idst1_numpy(x)

    print("[DST] 使用纯 NumPy DST-I (性能较低)")


def _init_transforms():
    global _USE_CUPY, _USE_SCIPY, _CP, dst, idst
    dst = idst = None
    _CP = None

    # Prefer a custom CuPy FFT DST-I because cupyx.scipy.fft does not support type=1.
    try:
        import cupy as cp

        def dst(x, norm='ortho'):
            return _dst1_fft_backend(cp.asarray(x), cp, norm=norm)

        def idst(x, norm='ortho'):
            return _idst1_fft_backend(cp.asarray(x), cp, norm=norm)

        probe = cp.asarray(np.ones((4, 4), dtype=np.float32))
        _ = dst(probe, norm='ortho')
        _ = idst(probe, norm='ortho')

        _CP = cp
        _USE_CUPY = True
        _USE_SCIPY = False
        print("[DST] Using CuPy FFT DST-I")
        return
    except Exception as exc:
        _USE_CUPY = False
        _CP = None
        print(f"[DST] CuPy GPU DST-I init failed: {exc}; falling back to CPU")

    try:
        from scipy.fft import dst as sp_dst, idst as sp_idst

        _ = sp_dst(np.ones((4, 4), dtype=np.float32), type=1, norm='ortho')
        _ = sp_idst(np.ones((4, 4), dtype=np.float32), type=1, norm='ortho')

        def dst(x, norm='ortho'):
            return sp_dst(x, type=1, norm=norm)

        def idst(x, norm='ortho'):
            return sp_idst(x, type=1, norm=norm)

        _USE_SCIPY = True
        print("[DST] Using SciPy DST-I")
        return
    except Exception:
        _USE_SCIPY = False
        print("[DST] SciPy DST-I unavailable; falling back to NumPy FFT")

    def dst(x, norm='ortho'):
        return _dst1_fft_backend(np.asarray(x), np, norm=norm)

    def idst(x, norm='ortho'):
        return _idst1_fft_backend(np.asarray(x), np, norm=norm)

    print("[DST] Using NumPy FFT DST-I (slower fallback)")


_init_transforms()


# ------------------ 核心快速泊松 + 先验融合 ------------------
def fast_poisson_from_normals(normal, mask=None, eps=1e-6):
    """
    基于法线的快速泊松积分（相对深度，零边界）。
    返回: z_rel (H,W)
    """
    if normal.ndim == 3 and normal.shape[0] == 3 and normal.shape[2] != 3:
        normal = np.transpose(normal, (1, 2, 0))

    n = normal.astype(np.float32)
    norm_len = np.linalg.norm(n, axis=-1, keepdims=True).clip(min=1e-6)
    n = n / norm_len
    nx, ny, nz = n[..., 0], n[..., 1], np.clip(n[..., 2], eps, None)

    p = -nx / nz  # ∂z/∂x
    q = -ny / nz  # ∂z/∂y

    H, W = p.shape
    # 散度（简单向前差分）
    dpdx = np.zeros_like(p);
    dpdx[:, 1:] = p[:, 1:] - p[:, :-1]
    dqdy = np.zeros_like(q);
    dqdy[1:, :] = q[1:, :] - q[:-1, :]
    f = dpdx + dqdy

    # 内部尺寸
    inner = f[1:-1, 1:-1].copy()
    n = H - 2
    m = W - 2
    if n <= 0 or m <= 0:
        return np.zeros((H, W), dtype=np.float32)

    # 双向 DST-I
    F = dst(dst(inner, norm='ortho').T, norm='ortho').T

    # 特征值 λ_{i,j} = 2cos(pi*i/(n+1)) -2 + 2cos(pi*j/(m+1)) -2
    jj, ii = np.meshgrid(np.arange(1, m + 1), np.arange(1, n + 1), indexing='xy')
    denom = (2 * np.cos(np.pi * ii / (n + 1)) - 2) + (2 * np.cos(np.pi * jj / (m + 1)) - 2)
    # 避免除零（理论上不为零）
    denom[denom == 0] = np.inf

    Z = F / denom

    z_inner = idst(idst(Z, norm='ortho').T, norm='ortho').T
    z_rel = np.zeros((H, W), dtype=np.float32)
    z_rel[1:-1, 1:-1] = z_inner

    # 去除均值
    if mask is not None:
        msk = (mask > 0)
        if msk.any():
            z_rel -= z_rel[msk].mean()
        else:
            z_rel -= z_rel.mean()
    else:
        z_rel -= z_rel.mean()

    return z_rel


def fast_poisson_from_normals(normal, mask=None, eps=1e-6):
    """
    基于法线的快速 Poisson 积分。
    当 CuPy 可用时，这里会把 DST-I 求解链路切到 GPU。
    """
    if normal.ndim == 3 and normal.shape[0] == 3 and normal.shape[2] != 3:
        normal = np.transpose(normal, (1, 2, 0))

    xp = _CP if (_USE_CUPY and _CP is not None) else np

    n = xp.asarray(normal, dtype=xp.float32)
    norm_len = xp.linalg.norm(n, axis=-1, keepdims=True)
    norm_len = xp.clip(norm_len, 1e-6, None)
    n = n / norm_len
    nx, ny, nz = n[..., 0], n[..., 1], xp.clip(n[..., 2], eps, None)

    p = -nx / nz
    q = -ny / nz

    H, W = p.shape
    dpdx = xp.zeros_like(p)
    dpdx[:, 1:] = p[:, 1:] - p[:, :-1]
    dqdy = xp.zeros_like(q)
    dqdy[1:, :] = q[1:, :] - q[:-1, :]
    f = dpdx + dqdy

    inner = f[1:-1, 1:-1].copy()
    if inner.shape[0] <= 0 or inner.shape[1] <= 0:
        return np.zeros((H, W), dtype=np.float32)

    z_inner = _solve_poisson_inner_dst(inner, xp)
    z_rel = xp.zeros((H, W), dtype=xp.float32)
    z_rel[1:-1, 1:-1] = z_inner

    if mask is not None:
        msk = xp.asarray(mask > 0)
        if bool(msk.any().item()):
            z_rel -= z_rel[msk].mean()
        else:
            z_rel -= z_rel.mean()
    else:
        z_rel -= z_rel.mean()

    if xp is np:
        return z_rel.astype(np.float32, copy=False)
    return _CP.asnumpy(z_rel).astype(np.float32, copy=False)


def fast_reconstruct_quadratic(normal, depth_prior, mask=None):
    """
    极速高精度重建：DST积分 + 二次曲面去偏
    """
    # 1. 快速积分 (DST) - 这一步只求形状，不管绝对位置，耗时 < 10ms
    # 注意：这里我们信任 fast_poisson_from_normals 算出的相对形状
    z_rel = fast_poisson_from_normals(normal, mask=mask)

    # 2. 如果没有先验，直接返回
    if depth_prior is None:
        return z_rel

    # 3. 使用二次曲面拟合先验 (耗时 < 5ms)
    # 这比 plane_align 强在能消除“弯曲”的偏差
    z_final = surface_align_quadratic(z_rel, depth_prior, mask=mask)

    return z_final


def surface_align_quadratic(z_rel, depth_prior, mask=None):
    """
    使用二次曲面 (Quadratic Surface) 拟合误差，将相对深度 z_rel 对齐到深度先验。
    解决了法线积分带来的"碗状"或"拱形"低频畸变。
    拟合模型: Error = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*x*y
    """
    H, W = z_rel.shape

    # 1. 确定有效约束点（通常是边界的一圈先验）
    if mask is not None:
        valid = (mask > 0) & np.isfinite(depth_prior)
    else:
        valid = np.isfinite(depth_prior)

    if valid.sum() < 20:
        return z_rel  # 点太少，无法拟合

    # 2. 提取坐标并归一化 (归一化对高阶拟合的数值稳定性至关重要)
    y_idx, x_idx = np.where(valid)
    z_diff = depth_prior[valid] - z_rel[valid]

    # 归一化坐标到 [-1, 1]
    y_norm = (y_idx - H / 2) / (H / 2)
    x_norm = (x_idx - W / 2) / (W / 2)

    # 3. 构建设计矩阵 A (N x 6)
    # [1, x, y, x^2, y^2, xy]
    A = np.column_stack((
        np.ones_like(x_norm),
        x_norm,
        y_norm,
        x_norm ** 2,
        y_norm ** 2,
        x_norm * y_norm
    ))

    # 4. 最小二乘求解系数
    # c 包含 6 个系数
    c, _, _, _ = np.linalg.lstsq(A, z_diff, rcond=None)

    # 5. 全局重建误差曲面
    yy, xx = np.mgrid[0:H, 0:W]
    yy_norm = (yy - H / 2) / (H / 2)
    xx_norm = (xx - W / 2) / (W / 2)

    global_error_surface = (
            c[0] +
            c[1] * xx_norm +
            c[2] * yy_norm +
            c[3] * xx_norm ** 2 +
            c[4] * yy_norm ** 2 +
            c[5] * xx_norm * yy_norm
    )

    # 6. 矫正
    return z_rel + global_error_surface


def poisson_dirichlet_from_normals_rect_boundary(normal,
                                                 boundary_prior, normal_gt=None, eps=1e-6):
    """
    用法概述:
      对于满足「内部为未知 (NaN) 的矩形区域、外围有一圈已知深度值」的 deep-prior，
      使用法线场恢复该矩形内部的深度，使内部解在边界上满足 Dirichlet 条件。
      使用 DST-I（零边界变换的变体）在频域快速求解五点离散拉普拉斯方程。
    """

    # 兼容 (3,H,W) 的输入格式 -> 转为 (H,W,3)
    if normal.ndim == 3 and normal.shape[0] == 3 and normal.shape[2] != 3:
        normal = np.transpose(normal, (1, 2, 0))

    H, W = boundary_prior.shape
    # 找到 NaN（未知）像素构成的 mask
    nan_mask = np.isnan(boundary_prior)
    if not nan_mask.any():
        # 没有 NaN，说明不是边界型先验
        return None

    ys, xs = np.where(nan_mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # 要求内部为完整矩形（中间不能有非 NaN）
    if not np.all(nan_mask[y0:y1 + 1, x0:x1 + 1]):
        return None
    # 要求矩形内部周围至少被一圈像素包围（不能贴图像边缘）
    if y0 == 0 or x0 == 0 or y1 == H - 1 or x1 == W - 1:
        return None

    # 读取紧邻未知矩形的四条边（作为 Dirichlet 边界）
    top_row = boundary_prior[y0 - 1, x0:x1 + 1]
    bottom_row = boundary_prior[y1 + 1, x0:x1 + 1]
    left_col = boundary_prior[y0:y1 + 1, x0 - 1]
    right_col = boundary_prior[y0:y1 + 1, x1 + 1]
    # 若任一边有 NaN，则边界不完整，放弃
    if any(np.isnan(v).any() for v in [top_row, bottom_row, left_col, right_col]):
        return None

    # 提取包含一圈边界的法线子块（用于计算内部散度）
    ny0, ny1 = y0 - 1, y1 + 1
    nx0, nx1 = x0 - 1, x1 + 1
    sub_normals = normal[ny0:ny1 + 1, nx0:nx1 + 1, :].astype(np.float32)

    # sub_normals_gt = normal_gt[ny0:ny1 + 1, nx0:nx1 + 1, :].astype(np.float32)

    # 归一化法线并保护 nz 接近 0 的情况
    nlen = np.linalg.norm(sub_normals, axis=-1, keepdims=True).clip(min=1e-6)
    nrm = sub_normals / nlen
    nx_, ny_, nz_ = nrm[..., 0], nrm[..., 1], np.clip(nrm[..., 2], eps, None)

    # nlen_gt = np.linalg.norm(sub_normals_gt, axis=-1, keepdims=True).clip(min=1e-6)
    # nrm_gt = sub_normals_gt / nlen_gt
    # nx_gt, ny_gt, nz_gt = nrm_gt[..., 0], nrm_gt[..., 1], np.clip(nrm_gt[..., 2], eps, None)

    # 从法线得到深度梯度近似 p = dz/dx, q = dz/dy
    p = -nx_ / nz_
    q = ny_ / nz_

    # ############ 使用 GT 法线计算散度以提高稳定性 ############
    # p_gt = -nx_gt / nz_gt
    # q_gt = ny_gt / nz_gt

    # 用简单差分构造散度 f = d(p)/dx + d(q)/dy（向前差分 / 零填充）
    dpdx = np.zeros_like(p);
    dpdx[:, 1:] = p[:, 1:] - p[:, :-1]
    dqdy = np.zeros_like(q);
    dqdy[1:, :] = q[1:, :] - q[:-1, :]
    f = dpdx + dqdy

    # # 使用 GT 法线计算的散度
    # dpdx_gt = np.zeros_like(p_gt);
    # dpdx_gt[:, 1:] = p_gt[:, 1:] - p_gt[:, :-1]
    # dqdy_gt = np.zeros_like(q_gt);
    # dqdy_gt[1:, :] = q_gt[1:, :] - q_gt[:-1, :]
    # f_gt = dpdx_gt + dqdy_gt

    # 内部区域（去掉外一圈）的 RHS，用于 DST 解算
    inner = f[1:-1, 1:-1].copy()

    # inner_gt = f_gt[1:-1, 1:-1].copy()

    nH = (y1 - y0 + 1)  # 内部高
    nW = (x1 - x0 + 1)  # 内部宽
    # 形状校验（防止索引/切片不匹配）
    if inner.shape != (nH, nW):
        return None

    # 校验边界长度与内部尺寸匹配
    if top_row.shape[0] != nW or bottom_row.shape[0] != nW:
        return None
    if left_col.shape[0] != nH or right_col.shape[0] != nH:
        return None

    # 五点拉普拉斯：内点的 RHS 需减去已知邻居（边界）贡献
    # 注意：这里直接用整行/整列，不做 [1:-1] 裁切以确保维度一致
    inner[0, :] -= top_row  # 内部第一行的上邻是 top_row
    inner[-1, :] -= bottom_row  # 内部最后一行的下邻是 bottom_row
    inner[:, 0] -= left_col  # 内部第一列的左邻是 left_col
    inner[:, -1] -= right_col  # 内部最后一列的右邻是 right_col

    # ############ 使用 GT 法线计算的散度进行同样的边界调整 ############
    # inner_gt[0, :] -= top_row  # 内部第一行的上邻是 top_row
    # inner_gt[-1, :] -= bottom_row  # 内部最后一行的下邻是 bottom_row
    # inner_gt[:, 0] -= left_col  # 内部第一列的左邻是 left_col
    # inner_gt[:, -1] -= right_col  # 内部最后一列的右邻是 right_col

    # 使用双向 DST-I 在频域解 Δ z = f'（这里 f' 为调整后的 inner）
    F = dst(dst(inner, norm='ortho').T, norm='ortho').T
    # ############ 使用 GT 法线计算的散度进行同样的 DST 变换 ############
    # F_gt = dst(dst(inner_gt, norm='ortho').T, norm='ortho').T

    # 特征值矩阵（与尺寸相关）
    jj, ii = np.meshgrid(np.arange(1, nW + 1), np.arange(1, nH + 1), indexing='xy')
    denom = (2 * np.cos(np.pi * ii / (nH + 1)) - 2) + (2 * np.cos(np.pi * jj / (nW + 1)) - 2)
    denom[denom == 0] = np.inf  # 理论上不为零，防守式处理
    Z = F / denom
    # ############ 使用 GT 法线计算的散度进行同样的频域解算 ############
    # Z_gt = F_gt / denom

    # 逆变换得到内部解
    z_inner = idst(idst(Z, norm='ortho').T, norm='ortho').T
    # ############ 使用 GT 法线计算的散度进行同样的逆变换 ############
    # z_inner_gt = idst(idst(Z_gt, norm='ortho').T, norm='ortho').T

    # 组装输出：内部用解覆盖，外围保留原先验
    depth_out = boundary_prior.copy()
    depth_out[y0:y1 + 1, x0:x1 + 1] = z_inner
    # ############ 使用 GT 法线计算的散度进行同样的组装 ############
    # depth_out_gt = boundary_prior.copy()
    # depth_out_gt[y0:y1 + 1, x0:x1 + 1] = z_inner_gt

    # 转换为物理单位后返回
    return depth_out
    #return depth_out, depth_out_gt


def poisson_dirichlet_from_normals_rect_boundary(normal,
                                                 boundary_prior, normal_gt=None, eps=1e-6):
    """
    使用矩形 Dirichlet 边界先验做 Poisson 重建。
    当 CuPy 可用时，内部 DST-I 频域求解走 GPU。
    """
    if normal.ndim == 3 and normal.shape[0] == 3 and normal.shape[2] != 3:
        normal = np.transpose(normal, (1, 2, 0))

    xp = _CP if (_USE_CUPY and _CP is not None) else np

    H, W = boundary_prior.shape
    nan_mask = np.isnan(boundary_prior)
    if not nan_mask.any():
        return None

    ys, xs = np.where(nan_mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    if not np.all(nan_mask[y0:y1 + 1, x0:x1 + 1]):
        return None
    if y0 == 0 or x0 == 0 or y1 == H - 1 or x1 == W - 1:
        return None

    top_row = boundary_prior[y0 - 1, x0:x1 + 1]
    bottom_row = boundary_prior[y1 + 1, x0:x1 + 1]
    left_col = boundary_prior[y0:y1 + 1, x0 - 1]
    right_col = boundary_prior[y0:y1 + 1, x1 + 1]
    if any(np.isnan(v).any() for v in [top_row, bottom_row, left_col, right_col]):
        return None

    ny0, ny1 = y0 - 1, y1 + 1
    nx0, nx1 = x0 - 1, x1 + 1
    sub_normals = xp.asarray(normal[ny0:ny1 + 1, nx0:nx1 + 1, :], dtype=xp.float32)

    nlen = xp.linalg.norm(sub_normals, axis=-1, keepdims=True)
    nlen = xp.clip(nlen, 1e-6, None)
    nrm = sub_normals / nlen
    nx_, ny_, nz_ = nrm[..., 0], nrm[..., 1], xp.clip(nrm[..., 2], eps, None)

    p = -nx_ / nz_
    q = ny_ / nz_

    dpdx = xp.zeros_like(p)
    dpdx[:, 1:] = p[:, 1:] - p[:, :-1]
    dqdy = xp.zeros_like(q)
    dqdy[1:, :] = q[1:, :] - q[:-1, :]
    f = dpdx + dqdy

    inner = f[1:-1, 1:-1].copy()
    nH = y1 - y0 + 1
    nW = x1 - x0 + 1
    if inner.shape != (nH, nW):
        return None
    if top_row.shape[0] != nW or bottom_row.shape[0] != nW:
        return None
    if left_col.shape[0] != nH or right_col.shape[0] != nH:
        return None

    inner[0, :] -= xp.asarray(top_row, dtype=inner.dtype)
    inner[-1, :] -= xp.asarray(bottom_row, dtype=inner.dtype)
    inner[:, 0] -= xp.asarray(left_col, dtype=inner.dtype)
    inner[:, -1] -= xp.asarray(right_col, dtype=inner.dtype)

    z_inner = _solve_poisson_inner_dst(inner, xp)

    depth_out = boundary_prior.copy()
    if xp is np:
        depth_out[y0:y1 + 1, x0:x1 + 1] = z_inner.astype(np.float32, copy=False)
    else:
        depth_out[y0:y1 + 1, x0:x1 + 1] = _CP.asnumpy(z_inner).astype(np.float32, copy=False)
    return depth_out


def reconstruct_with_boundary_prior(normal, depth_prior, normal_gt=None, mask=None, beta=0.0):
    """
    调度：若 depth_prior 仅提供边界环(内部 NaN) -> 使用狄利克雷泊松；
         否则走原 hybrid (plane_align + 可选融合)。
    """
    # 尝试边界型
    has_nan = depth_prior is not None and np.isnan(depth_prior).any()
    if has_nan:
        # depth_dir, depth_dir_gt = poisson_dirichlet_from_normals_rect_boundary(normal, depth_prior,normal_gt=None)
        depth_dir = poisson_dirichlet_from_normals_rect_boundary(normal, depth_prior, normal_gt=None)
        if depth_dir is not None:
            # return depth_dir, depth_dir_gt
            return depth_dir
    # 回退到原有流程
    return hybrid_reconstruct(normal, depth_prior, mask=mask, beta=beta)


def plane_align(z_rel, depth_prior, mask=None):
    """
    使用平面拟合将相对深度 z_rel 对齐到深度先验 depth_prior 的低频/全局偏移。
    """
    H, W = z_rel.shape
    # 构造像素坐标网格：yy 为行索引，xx 为列索引
    yy, xx = np.mgrid[0:H, 0:W]

    # 若未提供先验，无法对齐，直接返回原始相对深度
    if depth_prior is None:
        return z_rel

    # 确定用于拟合的有效像素集合
    if mask is not None:
        valid = (mask > 0) & np.isfinite(depth_prior)
    else:
        valid = np.isfinite(depth_prior)

    # 若有效像素太少，拟合不可靠，直接返回原始相对深度
    if valid.sum() < 20:
        return z_rel

    # 构造线性系统 A * theta = b
    A = np.stack([xx[valid], yy[valid], np.ones_like(xx[valid])], axis=1)
    b = (depth_prior[valid] - z_rel[valid]).astype(np.float32)

    # 最小二乘求解 theta = [a, b_coef, c]
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, c = theta

    # 将估计的平面加回到整个 z_rel
    return z_rel + (a * xx + b_coef * yy + c)


def hybrid_reconstruct(normal, depth_prior, mask=None, beta=0.0):
    """
    混合重建函数（实时友好）：使用快速泊松积分恢复相对深度，然后使用深度先验校正低频。
    """
    # 1) 用快速泊松从法线恢复相对深度
    z_rel = fast_poisson_from_normals(normal, mask=mask)

    # 2) 若无先验直接返回相对深度
    if depth_prior is None:
        return z_rel

    # 3) 使用先验做平面拟合校正低频（恢复全局偏移/倾斜）
    z_align = plane_align(z_rel, depth_prior, mask)

    # 4) 若不需要低频融合，返回平面对齐结果（最快）
    if beta <= 0:
        return z_align

    # 5) 低频融合：用高斯模糊提取低频并合成
    k = max(5, int(0.1 * min(z_align.shape)))  # 自适应核大小
    if k % 2 == 0:
        k += 1  # 确保奇数核
    lp_prior = cv2.GaussianBlur(depth_prior.astype(np.float32), (k, k), 0)  # 先验低频
    lp_align = cv2.GaussianBlur(z_align.astype(np.float32), (k, k), 0)  # 对齐结果低频
    high = z_align - lp_align  # 保留 z_align 的高频细节
    z_mix = lp_prior + high  # 用先验低频 + 当前高频构建混合深度

    # 6) 返回按 beta 加权的混合结果
    return (1 - beta) * z_align + beta * z_mix


# ------------------ 保存工具 ------------------
def save_depth_for_force(reconstructed_depth, out_dir, idx,
                         target_size=(313, 313),
                         depth_min=10, depth_max=30,
                         save_npy=True,
                         norm=False,
                         use_nir=False):
    os.makedirs(out_dir, exist_ok=True)
    if depth_min is None:
        depth_min = float(np.nanmin(reconstructed_depth))
    if depth_max is None:
        depth_max = float(np.nanmax(reconstructed_depth))
    if depth_max == depth_min:
        depth_max = depth_min + 1e-6
    if norm:
        depth = (reconstructed_depth.astype(np.float32) - depth_min) / (depth_max - depth_min)
        depth = np.clip(depth, 0.0, 1.0)
    else:
        depth = reconstructed_depth.astype(np.float32)
    depth_uint16 = (depth * 32768.0).round().astype(np.uint16)
    if use_nir:
        dep_pred_path = os.path.join(out_dir, 'depth_pred_nir')
    else:
        dep_pred_path = os.path.join(out_dir, 'depth_pred')
    os.makedirs(dep_pred_path, exist_ok=True)
    png_path = os.path.join(dep_pred_path, f"depth_pred{idx}.png")
    # if not cv2.imwrite(png_path, depth_uint16):
    #     raise IOError(f"保存深度 PNG 失败: {png_path}")
    if save_npy:
        np.save(os.path.join(dep_pred_path, f"depth_pred{idx}.npy"), depth)
    return png_path


# ------------------ 主流程 ------------------
def calculate_reconstruction_error(data_folder,
                                    model_version,
                                    depth_prior=None,
                                    method='hybrid',
                                    beta=0.0,
                                    use_nir=None,
                                    enable_stage1_error_stats=True,
                                    enable_inpaint=True,
                                    inpaint_radius=3,
                                    inpaint_dilate_iter=5):
    """深度重建与评估主流程。

    拆分为两部分：
    1）GT 法向重建 + 误差统计：
        - 遍历所有 depth_gt/depth*.npy
        - 使用 normal_complete_gt/normal_completex.npy 做深度积分重建
        - 仅依赖 GT normal，不做法向预测，也不看 dh
        - 计算与 depth_gt 的 MAE 统计
    2）预测法向重建（仅保存，不计误差）：
        - 遍历 dh 目录下所有 x_dh.npy 得到样本 id 列表
        - 对这些样本做法向预测 + 深度积分
        - 可以使用 depth_prior 做全局校正
        - 无论是否有 depth_gt 均执行，但不纳入误差统计
    """
    print("初始化法线预测模型...")

    img_path = os.path.join(data_folder, 'images')
    img_removed_path = os.path.join(data_folder, 'images_removed')
    dep_path = os.path.join(data_folder, 'depth_gt')
    mask_path_base = os.path.join(data_folder, 'mask')
    normal_gt_path = os.path.join(data_folder, 'normal_complete_gt')
    dh_dir = os.path.join(data_folder, 'dh')

    # ------------------- 背景加载（可选 inpaint） -------------------
    bg_rgb_path = os.path.join(img_removed_path, '0_rgb.png')
    bg_rgb_raw = cv2.imread(bg_rgb_path)
    if bg_rgb_raw is None:
        # fallback to original images
        bg_rgb_path = os.path.join(img_path, '0_rgb.png')
        bg_rgb_raw = cv2.imread(bg_rgb_path)
        if bg_rgb_raw is None:
            print(f"背景图像缺失: {bg_rgb_path}")
            return
        
    # 背景直接读取 images_removed，无需 inpaint
    bg_rgb = bg_rgb_raw

    # if enable_inpaint:
    #     try:
    #         bg_rgb = preprocess_rgb_with_inpaint(bg_rgb_raw, inpaint_radius=inpaint_radius,
    #                                              dilate_iter=inpaint_dilate_iter)
    #         print("[离线] 背景已做标记点去除处理。")
    #     except Exception as e:
    #         print(f"[警告] 背景 inpaint 失败，使用原始背景: {e}")
    #         bg_rgb = bg_rgb_raw
    # else:
    #     bg_rgb = bg_rgb_raw
    #     print("[离线] 标记点预处理已关闭，背景使用原始图像。")

    # 获取图像尺寸并初始化模型
    H, W = bg_rgb_raw.shape[:2]
    geo_model = Geo_Model(version=model_version, use_pos=None, use_nir=use_nir, H=H, W=W)
    resolved_use_nir = geo_model.use_nir
    print(f"[calculate_depth_offline] version_{model_version} resolved use_nir={resolved_use_nir}, use_pos={geo_model.use_pos}")

    bg_nir = None
    if resolved_use_nir:
        bg_nir_path = os.path.join(img_removed_path, '0_nir.png')
        if not os.path.exists(bg_nir_path):
            bg_nir_path = os.path.join(img_path, '0_nir.png')
        if not os.path.exists(bg_nir_path):
            print(f"NIR 背景图像缺失: {bg_nir_path}")
            return
        bg_nir_raw = cv2.imread(bg_nir_path, cv2.IMREAD_UNCHANGED)
        bg_nir = _read_and_process_nir(bg_nir_raw)

    geo_model.load_bg(bg_rgb, bg_nir)
    print("背景加载完成")

    # 先验路径
    prior_path = depth_prior
    depth_prior_global = None
    # 保护：depth_prior 可能为 None，避免传入 os.path.exists
    if prior_path is not None and os.path.exists(prior_path):
        depth_prior_global = np.load(prior_path).astype(np.float32)
        print(f"深度先验加载: {prior_path}")
    else:
        if prior_path is None:
            print("未提供 depth_prior，使用 None")
        else:
            print(f"深度先验未找到: {prior_path}，使用 None")
        depth_prior_global = None

    # ------------------- Part 1: GT normal 完全重建 + 误差统计 -------------------
    if enable_stage1_error_stats:
        # 在这里我们也加入 Pred Normal 的流程，以便对比 GT normal vs Pred Normal 的重建效果
        print("\n==== Part1: 深度重建评估 (Pred Normal vs GT Normal) ====")
        depth_files = sorted(glob.glob(os.path.join(dep_path, 'depth*.npy')))

        # 统计列表
        depth_mae_pred_list_p1 = []
        depth_mae_gt_normal_list_p1 = []
        recon_times_pred_p1 = []       # 仅统计 normal->depth 时间 (Pred)
        recon_times_gt_normal_p1 = []  # 仅统计 normal->depth 时间 (GT)

        if not depth_files:
            print("未找到 depth*.npy，跳过 Part1。")
        else:
            for depth_gt_path in depth_files:
                try:
                    base_name = os.path.basename(depth_gt_path).replace('depth', '').replace('.npy', '')
                    if base_name == '0':
                        continue

                    rgb_path = os.path.join(img_path, f"{base_name}_rgb.png")
                    mask_path = os.path.join(mask_path_base, f"mask{base_name}.npy")
                    current_normal_gt_path = os.path.join(normal_gt_path, f"normal_complete{base_name}.npy")

                    print(f"\n[Part1评估] 样本 {base_name}")
                    if not os.path.exists(current_normal_gt_path):
                        print(f"[警告] 缺少 normal_complete{base_name}.npy，跳过。")
                        continue

                    # 1. 加载 GT 数据
                    depth_gt = np.load(depth_gt_path).astype(np.float32)
                    normal_gt = np.load(current_normal_gt_path).astype(np.float32)
                    if os.path.exists(mask_path):
                        mask = np.load(mask_path).astype(np.uint8)
                        bool_mask = mask > 0
                    else:
                        mask = np.ones_like(depth_gt, dtype=np.uint8)
                        bool_mask = mask > 0

                    # 2. 准备 RGB / NIR
                    # Part1: 直接使用 data_folder/images_removed 中的 inpaint 图像进行推理
                    images_removed_dir = os.path.join(data_folder, "images_removed")
                    rgb_removed_path = os.path.join(images_removed_dir, f"{base_name}_rgb.png")

                    rgb_raw = cv2.imread(rgb_removed_path)
                    if rgb_raw is None:
                        # 回退到原始 images 下的 RGB
                        rgb_raw = cv2.imread(rgb_path)
                        if rgb_raw is None:
                            print(f"无法读取 RGB: {rgb_path}")
                            continue

                    rgb_img = rgb_raw

                    rgb_tensor = TF.to_tensor(rgb_img).unsqueeze(0)

                    nir_tensor = None
                    if resolved_use_nir:
                        nir_path = os.path.join(img_removed_path, f"{base_name}_nir.png")
                        if not os.path.exists(nir_path):
                            nir_path = os.path.join(img_path, f"{base_name}_nir.png")
                        if not os.path.exists(nir_path):
                            nir_path = os.path.join(img_path, f"{base_name}_ir.png")

                        if os.path.exists(nir_path):
                            nir_raw = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
                            nir_proc = _read_and_process_nir(nir_raw)
                            nir_tensor = TF.to_tensor(nir_proc).unsqueeze(0)

                    # 3. 法向预测 (Pred Normal)
                    pred_tensor, _ = geo_model.predict(rgb_tensor, nir_tensor, measure=False)
                    pred_normal = pred_tensor.cpu().numpy().astype(np.float32)
                    try:
                        nz_mean = float(np.nanmean(pred_normal[..., 2]))
                        if nz_mean < 0:
                            pred_normal = -pred_normal
                    except Exception:
                        pass

                    # 4. 深度先验准备
                    local_prior = None
                    if depth_prior_global is not None:
                        if depth_prior_global.shape != depth_gt.shape:
                            local_prior = cv2.resize(depth_prior_global, (depth_gt.shape[1], depth_gt.shape[0]),
                                                     interpolation=cv2.INTER_LINEAR)
                        else:
                            local_prior = depth_prior_global

                    # 5. 重建深度1：Pred Normal -> Depth (并计时)
                    t_p = time.time()
                    if method in ('spsolve', 'sparseqr', 'cupy'):
                        temp_path = os.path.join(data_folder, "temp_normal_pred_p1.npy")
                        np.save(temp_path, pred_normal)
                        task = Normal_Integration(nomral_path=temp_path, depth_path=prior_path, solver=method)
                        task.mesh_deformation()
                        v_depth = task.v_depth
                        rec_depth_pred = convolve2d(v_depth, np.array([[0.25, 0.25], [0.25, 0.25]]), mode="valid")
                    elif method == 'poisson':
                        rec_depth_pred = reconstruct_with_boundary_prior(pred_normal, local_prior, normal_gt, mask=None, beta=beta)
                    else:
                        rec_depth_pred = hybrid_reconstruct(pred_normal, local_prior, mask=None, beta=beta)

                    dt_pred = (time.time() - t_p) * 1000.0
                    recon_times_pred_p1.append(dt_pred)

                    # 6. 重建深度2：GT Normal -> Depth (并计时)
                    t_g = time.time()
                    if method in ('spsolve', 'sparseqr', 'cupy'):
                        temp_path = os.path.join(data_folder, "temp_normal_gt_p1.npy")
                        np.save(temp_path, normal_gt)
                        task = Normal_Integration(nomral_path=temp_path, depth_path=prior_path, solver=method)
                        task.mesh_deformation()
                        v_depth = task.v_depth
                        rec_depth_gt = convolve2d(v_depth, np.array([[0.25, 0.25], [0.25, 0.25]]), mode="valid")
                    elif method == 'poisson':
                        rec_depth_gt = reconstruct_with_boundary_prior(normal_gt, local_prior, normal_gt, mask=None, beta=beta)
                    else:
                        rec_depth_gt = hybrid_reconstruct(normal_gt, local_prior, mask=None, beta=beta)

                    dt_gt = (time.time() - t_g) * 1000.0
                    recon_times_gt_normal_p1.append(dt_gt)

                    # 7. 计算误差 (MAE, shape-aligned)
                    if np.sum(bool_mask) > 0:
                        gt_sub = depth_gt[bool_mask]

                        if rec_depth_pred is not None:
                            rec_sub_p = rec_depth_pred[bool_mask]
                            mae_p = np.mean(np.abs((gt_sub - gt_sub.mean()) - (rec_sub_p - rec_sub_p.mean())))
                            depth_mae_pred_list_p1.append(mae_p)
                        else:
                            mae_p = float('nan')

                        if rec_depth_gt is not None:
                            rec_sub_g = rec_depth_gt[bool_mask]
                            mae_g = np.mean(np.abs((gt_sub - gt_sub.mean()) - (rec_sub_g - rec_sub_g.mean())))
                            depth_mae_gt_normal_list_p1.append(mae_g)
                        else:
                            mae_g = float('nan')

                        print(f"  [MAE] Pred: {mae_p:.4f} mm | GT Normal: {mae_g:.4f} mm")
                        print(f"  [Time] Pred: {dt_pred:.1f} ms | GT Normal: {dt_gt:.1f} ms")

                except Exception as e:
                    print(f"[Part1] 样本 {base_name} 出错: {e}")
                    import traceback
                    traceback.print_exc()

            print("\n==== Part1 统计结果 (Pred Normal vs GT Normal) ====")
            if depth_mae_pred_list_p1:
                print(f"Depth MAE (Pred Normal): mean={np.mean(depth_mae_pred_list_p1):.4f}, std={np.std(depth_mae_pred_list_p1):.4f} (N={len(depth_mae_pred_list_p1)})")
            if depth_mae_gt_normal_list_p1:
                 print(f"Depth MAE (GT Normal):   mean={np.mean(depth_mae_gt_normal_list_p1):.4f}, std={np.std(depth_mae_gt_normal_list_p1):.4f} (N={len(depth_mae_gt_normal_list_p1)})")

            if recon_times_pred_p1:
                 print(f"Time (Pred Norm -> Depth): mean={np.mean(recon_times_pred_p1):.2f} ms")
            if recon_times_gt_normal_p1:
                 print(f"Time (GT Norm -> Depth):   mean={np.mean(recon_times_gt_normal_p1):.2f} ms")
    else:
        print("\n==== Part1 已关闭：跳过 GT normal 深度重建误差统计 ====")

    # ------------------- Part 2: 遍历 images 下所有样本（不依赖 dh） -------------------
    def collect_ids_from_images(images_dir: str):
        ids = []
        if not os.path.isdir(images_dir):
            return []
        for p in glob.glob(os.path.join(images_dir, "*_rgb.png")):
            core = os.path.basename(p).split('_')[0]
            if core.isdigit():
                ids.append(int(core))
        return sorted(set(ids))

    print("\n==== Part2: 遍历 images 目录下所有 *_rgb.png 做法向预测 + 深度积分 ====")
    ids_all = collect_ids_from_images(img_path)
    if not ids_all:
        print(f"[预测] images 目录下未找到 *_rgb.png: {img_path}")
        return

    # 关闭可视化时的梯度误差统计
    px_mae_list = []
    py_mae_list = []
    px_rmse_list = []
    py_rmse_list = []
    grad_count = 0
    
    # 耗时统计
    infer_times = []
    recon_times = []

    for idx in tqdm(ids_all, desc="Processing"):
        base_name = str(idx)
        try:
            rgb_path = os.path.join(img_path, f"{base_name}_rgb.png")
            rgb_removed_path = os.path.join(img_removed_path, f"{base_name}_rgb.png")
            current_normal_gt_path = os.path.join(normal_gt_path, f"normal_complete{base_name}.npy")
            depth_gt_path = os.path.join(dep_path, f"depth{base_name}.npy")  # 可能不存在
            mask_path = os.path.join(mask_path_base, f"mask{base_name}.npy")  # 可能不存在

            print(f"\n[预测重建] 样本 {base_name}")
            if enable_inpaint:
                rgb_raw = cv2.imread(rgb_path)
            else:
                rgb_raw = cv2.imread(rgb_removed_path)
                if rgb_raw is None:
                    rgb_raw = cv2.imread(rgb_path)
            if rgb_raw is None:
                print(f"[预测重建] 无法读取 RGB 图像: {rgb_removed_path} / {rgb_path}，跳过。")
                continue

            if enable_inpaint:
                try:
                    rgb_img = preprocess_rgb_with_inpaint(rgb_raw, inpaint_radius=inpaint_radius,
                                                         dilate_iter=inpaint_dilate_iter)
                except Exception as e:
                    print(f"[预测重建] 图像 inpaint 失败，使用原始帧: {e}")
                    rgb_img = rgb_raw
            else:
                rgb_img = rgb_raw

            rgb_tensor = TF.to_tensor(rgb_img).unsqueeze(0)
            nir_tensor = None
            nir_raw_for_vis = None

            if resolved_use_nir:
                nir_path = os.path.join(img_path, f"{base_name}_nir.png")
                nir_removed_path = os.path.join(img_removed_path, f"{base_name}_nir.png")
                if not os.path.exists(nir_path):
                    nir_path = os.path.join(img_path, f"{base_name}_ir.png")
                if not os.path.exists(nir_removed_path):
                    nir_removed_path = os.path.join(img_removed_path, f"{base_name}_ir.png")
                if enable_inpaint:
                    nir_source_path = nir_path
                else:
                    nir_source_path = nir_removed_path if os.path.exists(nir_removed_path) else nir_path

                if os.path.exists(nir_source_path):
                    nir_raw = cv2.imread(nir_source_path, cv2.IMREAD_UNCHANGED)
                    nir_raw_for_vis = nir_raw
                    if enable_inpaint:
                        try:
                            # 尝试使用 RGB 的 mask 来处理 NIR
                            mask_for_nir = generate_marker_mask(rgb_raw) if rgb_raw is not None else None
                            nir_img = preprocess_rgb_with_inpaint(nir_raw, inpaint_radius=inpaint_radius,
                                                                  dilate_iter=inpaint_dilate_iter, mask=mask_for_nir)
                        except Exception as e:
                            print(f"[预测重建] NIR 图像 inpaint 失败，使用原始: {e}")
                            nir_img = nir_raw
                    else:
                        nir_img = nir_raw

                    # 统一处理 NIR 格式（16位转8位）
                    nir_img = _read_and_process_nir(nir_img)

                    nir_tensor = TF.to_tensor(nir_img).unsqueeze(0)
                else:
                    print(f"[预测重建] 模型需要 NIR，但未找到 {nir_removed_path} / {nir_path}")

            # -------- 法向预测（pred_normal / infer_ms）--------
            t0 = time.time()
            pred_tensor, _ = geo_model.predict(rgb_tensor, nir_tensor, measure=False)
            pred_normal = pred_tensor.cpu().numpy().astype(np.float32)
            # 若平均 nz 为负则翻转法向（保证面朝上）
            try:
                nz_mean = float(np.nanmean(pred_normal[..., 2]))
                if nz_mean < 0:
                    pred_normal = -pred_normal
            except Exception:
                pass
            infer_ms = (time.time() - t0) * 1000.0
            # print(f"[预测重建] 法线推理: {infer_ms:.1f} ms")
            infer_times.append(infer_ms)

            # 若存在 GT 深度/法向，可加载，否则 None
            depth_gt = np.load(depth_gt_path).astype(np.float32) if os.path.exists(depth_gt_path) else None
            normal_gt = np.load(current_normal_gt_path).astype(np.float32) if os.path.exists(current_normal_gt_path) else None
            mask = np.load(mask_path).astype(np.uint8) if os.path.exists(mask_path) else None

            # 先验 resize
            local_prior = None
            if depth_prior_global is not None and depth_gt is not None:
                if depth_prior_global.shape != depth_gt.shape:
                    local_prior = cv2.resize(depth_prior_global, (depth_gt.shape[1], depth_gt.shape[0]),
                                             interpolation=cv2.INTER_LINEAR)
                else:
                    local_prior = depth_prior_global
            else:
                local_prior = depth_prior_global

            # 深度积分（预测法线）
            t1 = time.time()
            if method in ('spsolve', 'sparseqr', 'cupy'):
                temp_normal_path = os.path.join(data_folder, "temp_normal_pred.npy")
                np.save(temp_normal_path, pred_normal)
                task = Normal_Integration(nomral_path=temp_normal_path,
                                          depth_path=prior_path,
                                          solver=method)
                task.mesh_deformation()
                v_depth = task.v_depth
                reconstructed_depth = convolve2d(v_depth, np.array([[0.25, 0.25], [0.25, 0.25]]), mode="valid")
            elif method == 'poisson':
                reconstructed_depth = reconstruct_with_boundary_prior(pred_normal, local_prior, normal_gt,
                                                                      mask=None, beta=beta)
            else:
                reconstructed_depth = hybrid_reconstruct(pred_normal, local_prior, mask=None, beta=beta)

            recon_ms = (time.time() - t1) * 1000.0
            # print(f"[预测重建] 深度重建: {recon_ms:.1f} ms")
            recon_times.append(recon_ms)

            # 同时计算 GT 法线重建深度（如果有 GT normal）- 仅用于可视化
            depth_from_gt_normal = None
            if normal_gt is not None:
                try:
                    if method == 'poisson':
                        depth_from_gt_normal = reconstruct_with_boundary_prior(normal_gt, local_prior, normal_gt,
                                                                               mask=None, beta=beta)
                    else:
                        depth_from_gt_normal = hybrid_reconstruct(normal_gt, local_prior, mask=None, beta=beta)
                except Exception as e:
                    print(f"[Warn] GT normal 重建失败: {e}")

            # 梯度(px/py)误差（显示在 pred normal 窗口；关闭可视化时统计）
            pxpy_err = compute_px_py_error(pred_normal, normal_gt, mask=mask) if normal_gt is not None else None
            if (not VISUALIZE_EACH_SAMPLE) and (pxpy_err is not None):
                px_mae, py_mae, px_rmse, py_rmse = pxpy_err
                px_mae_list.append(px_mae)
                py_mae_list.append(py_mae)
                px_rmse_list.append(px_rmse)
                py_rmse_list.append(py_rmse)
                grad_count += 1

            # 可视化：每张图一个窗口，任意键到下一张
            print(f"VISULIZE_EACH_SAMPLE = {VISUALIZE_EACH_SAMPLE}")
            if VISUALIZE_EACH_SAMPLE:
                show_six_windows_and_wait(
                    rgb=rgb_raw,
                    nir=nir_raw_for_vis,
                    pred_normal=pred_normal,
                    gt_normal=normal_gt,
                    depth_gt=depth_gt,
                    depth_from_gt_normal=depth_from_gt_normal,
                    depth_from_pred_normal=reconstructed_depth,
                    sample_id=base_name,
                    pxpy_err=pxpy_err,
                )

            # 保存预测深度（不计误差）
            save_depth_for_force(reconstructed_depth, out_dir=data_folder, idx=base_name,
                                 depth_min=None, depth_max=None, save_npy=True, norm=False, use_nir=resolved_use_nir)

        except Exception as e:
            print(f"[预测重建] 样本 id={idx} 出错: {e}")

    cv2.destroyAllWindows()

    # 输出平均耗时
    if infer_times:
        print(f"\n[平均耗时] 法线推理: {np.mean(infer_times):.2f} ms")
    if recon_times:
        print(f"[平均耗时] 深度重建 (Total): {np.mean(recon_times):.2f} ms")



    # 关闭可视化时，输出梯度误差统计
    if (not VISUALIZE_EACH_SAMPLE) and grad_count > 0:
        print("\n==== Gradient Error Summary (px/py) ====")
        print(f"Count: {grad_count}")
        print(f"px_mae:  mean={float(np.mean(px_mae_list)):.6f}  std={float(np.std(px_mae_list)):.6f}")
        print(f"py_mae:  mean={float(np.mean(py_mae_list)):.6f}  std={float(np.std(py_mae_list)):.6f}")
        print(f"px_rmse: mean={float(np.mean(px_rmse_list)):.6f}  std={float(np.std(px_rmse_list)):.6f}")
        print(f"py_rmse: mean={float(np.mean(py_rmse_list)):.6f}  std={float(np.std(py_rmse_list)):.6f}")


def collect_ids_with_dh(dh_dir):
    """在 data_folder/dh 目录中查找所有 x_dh.npy，对应的 x 作为样本 id 返回。"""
    if not os.path.isdir(dh_dir):
        print(f"[警告] dh 目录不存在: {dh_dir}")
        return []
    ids = []
    for name in os.listdir(dh_dir):
        if not name.endswith('_dh.npy'):
            continue
        core = name[:-len('_dh.npy')]
        if not core.isdigit():
            continue
        ids.append(int(core))
    ids = sorted(set(ids))
    print(f"[信息] 在 dh 目录 {dh_dir} 中共发现 {len(ids)} 个带位移场的样本 id: {ids[:10]}" + (" ..." if len(ids) > 10 else ""))
    return ids


def preprocess_rgb_with_inpaint(rgb_img, inpaint_radius=3, dilate_iter=5, mask=None):
    """对单张 RGB 图像执行标记点区域检测和 inpaint 去除标记点，返回处理后的图像。"""
    if rgb_img is None:
        return None
    if mask is None:
        mask = generate_marker_mask(rgb_img)
    inpainted, _ = inpaint_markers(
        rgb_img,
        mask,
        method="telea",
        radius=inpaint_radius,
        dilate_iter=dilate_iter,
    )
    return inpainted


def _read_and_process_nir(img):
    """
    处理NIR图像：
    1. 判断是否为16位，若是，除以257转换为8位
    2. 始终返回 (H, W, 1) 的 uint8
    """
    if img is None:
        return None
    
        
    # 如果是多通道，转灰度 (H, W)
    if img.ndim == 3:
         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
         img_gray = img

    # 10bit 转 8bit
    if img_gray.dtype == np.uint16:
        img_gray = (img_gray * 255.0 / 1023.0).clip(0, 255).astype(np.uint8)
    elif img_gray.dtype != np.uint8:
        # 其他类型转 uint8
        img_gray = img_gray.astype(np.uint8)
    
    return img_gray


if __name__ == '__main__':
    # 示例
    # data_folder = r'./data/20250915-093000/test'    # 自研五指1号 小数据集
    # data_folder = r'./data/20250918-161136/test'  # 自研五指1号 大数据集
    # data_folder = r'./data/20251123-095400/test'  # 三指 内窥镜 标记点1号
    # data_folder = r'./data/20251205_173723/test'    # 曲面三指1号
    # data_folder = r'./data/20260106_210820/test'  # 多光谱三指1号
    # data_folder = r'./data/20260114_120750/test'  # 多光谱三指3号
    # data_folder = r'./data/20260302_115200/test'  # 多光谱三指3号
    # data_folder = r'./data/20260306_193600/train'  # 多光谱三指3号 半球形压头
    # data_folder = r'./data/20260310_151840/train'  # 多光谱三指3号 小球压头
    # data_folder = r'./data/20260328_121407/train'  # 多光谱三指3号 小球压头
    # data_folder = r'./data/20260114_120750/val'  # 多光谱三指3号
    # data_folder = r'./data/20260421_094400/val'  # 多光谱三指3号 128*128
    data_folder = r'./data/20260404_103156/train'  # 曲面三指2号
    # model_version = 40  # 自研五指1号 小数据集
    # model_version = 57  # 自研五指1号 大数据集
    # model_version = 56  # 三指 内窥镜 标记点1号
    # model_version = 62  # 曲面三指1号
    # model_version = 69  # 多光谱三指1号 NIR
    # model_version = 78  # 多光谱三指3号 NIR
    # model_version = 80  # 多光谱三指3号 RGB only  model_ori
    # model_version = 94  # 多光谱三指3号 RGB only model_quan
    # model_version = 95  # 多光谱三指3号 RGB NIR model_quan
    # model_version = 100  # 多光谱三指3号 RGB NIR model_quan
    # model_version = 116  # 内窥镜三指1号
    # model_version = 114  # 多光谱三指3号 RGB NIR
    model_version = 118  # 曲面三指2号
    # model_version = 119  # 多光谱三指3号
    # model_version = 120  # 多光谱三指3号 128*128
    # model_version = 121  # 多光谱三指3号 128*128
    # model_version = 122  # 多光谱三指3号 128*128
    # model_version = 123  # 多光谱三指3号 128*128
    # model_version = 124  # 多光谱三指3号 128*128
    # depth_prior_path = r'./dep_p_20250913_103542.npy'  # 自研五指1号边界先验
    # depth_prior_path = r'./dep_p_20251125_221351.npy'  # 三指 内窥镜 标记点1号 边界先验 宽度50像素
    # depth_prior_path = r'./dep_p_20251126_092101.npy'   # 三指 内窥镜 标记点1号 边界先验 宽度10像素
    # depth_prior_path = r'./dep_p_20251209_094825.npy'   # 曲面三指1号
    # depth_prior_path = None  
    # depth_prior_path = r'./dep_p_20260115_145555.npy'  # 多光谱三指1号
    # depth_prior_path = r'./dep_p_20260118_103612.npy'  # 多光谱三指3号
    # depth_prior_path = r'./dep_p_inner1900x1060.npy'    # 多光谱三指3号 20260414
    # depth_prior_path = r'./dep_p_inner126x126_20260421.npy'    # 多光谱三指3号 128*128
    # depth_prior_path = r'./dep_p_20260329_171736.npy'  # 内窥镜三指1号
    # depth_prior_path = r'./dep_p_20260404_203516.npy'  # 曲面三指2号 50*50 的边界先验
    depth_prior_path = r'depth_prior/dep_p_20260407_181422.npy'  # 曲面三指2号 inner rhs: 630*466
    
    # 设置为 None 时，根据 model_version 对应的备份训练配置自动决定是否使用 NIR
    use_nir_flag = None
    enable_stage1_error_stats = False  # 是否启用 Part1 的 GT normal 重建误差统计

    # 多光谱三指3号
    inpaint_radius = 5
    inpaint_dilate_iter = 10

    calculate_reconstruction_error(data_folder,
                                   model_version,
                                   depth_prior=depth_prior_path,
                                   method='poisson',  # 快速泊松
                                #    method='cupy',  # GPU求解稀疏矩阵
                                #    method='spsolve',  # 求解库求解稀疏矩阵
                                   # method='sparseqr',  # 求解库求解稀疏矩阵
                                   beta=0.0,
                                   use_nir=use_nir_flag,
                                   enable_stage1_error_stats=enable_stage1_error_stats,
                                   enable_inpaint=False,  # 是否启用标记点预处理
                                   inpaint_radius=inpaint_radius,       
                                   inpaint_dilate_iter=inpaint_dilate_iter,   
                                   )
