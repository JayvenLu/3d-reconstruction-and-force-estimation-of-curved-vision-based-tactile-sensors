from __future__ import annotations

"""
作用:
- 深度重建数值核心实现。
- 冻结了法向 -> p/q -> 散度 -> 边界注入 -> DST/IDST -> 绝对深度 的唯一参考链路。
- 供离线脚本、回归测试和后续 FPGA/PS 对拍统一调用。

Frozen numeric chain for boundary-prior-guided fast Poisson depth reconstruction.

Version seal:
- entry: reconstruct_with_boundary_prior
- primary path: poisson_dirichlet_from_normals_rect_boundary
- input normal: float32, HWC/CHW accepted, no unit-length normalization in solver
- p = -nx / nz
- q =  ny / nz
- divergence: forward difference
- boundary injection: current rectangular Dirichlet injection
- output: absolute depth
- transform gold standard: scipy.fft.dst/idst(type=1, norm="ortho")
"""

import json
import math
import os
import subprocess
import time

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import torch
except Exception:
    torch = None


NUMERIC_CHAIN_VERSION = "v1_rect_boundary_dirichlet_fp32"
NUMERIC_CHAIN_TRANSFORM = 'scipy.fft.dst/idst(type=1, norm="ortho")'
NUMERIC_CHAIN_OUTPUT = "absolute_depth"
NUMERIC_CHAIN_P_FORMULA = "p = -nx / nz"
NUMERIC_CHAIN_Q_FORMULA = "q = ny / nz"
NUMERIC_CHAIN_DIFF = "forward_difference"
NUMERIC_CHAIN_BOUNDARY = "rectangular_dirichlet_injection"
NUMERIC_CHAIN_NORMAL_DTYPE = "float32"
NUMERIC_CHAIN_NORMAL_UNITIZE = False

_USE_SCIPY = False
_HAS_TORCH = torch is not None


def _init_transforms() -> None:
    global dst, idst, _USE_SCIPY

    try:
        from scipy.fft import dst as sp_dst, idst as sp_idst

        def dst(x, norm="ortho"):
            return sp_dst(x, type=1, norm=norm)

        def idst(x, norm="ortho"):
            return sp_idst(x, type=1, norm=norm)

        _USE_SCIPY = True
        return
    except Exception:
        _USE_SCIPY = False

    def dst(x, norm="ortho"):
        return _dst1_fft_backend_numpy(np.asarray(x), axis=-1, norm=norm)

    def idst(x, norm="ortho"):
        return _idst1_fft_backend_numpy(np.asarray(x), axis=-1, norm=norm)


_init_transforms()


def get_numeric_chain_info() -> dict[str, object]:
    return {
        "numeric_chain_version": NUMERIC_CHAIN_VERSION,
        "transform_backend": NUMERIC_CHAIN_TRANSFORM,
        "scipy_available": _USE_SCIPY,
        "torch_available": _HAS_TORCH,
        "cpu_transform_impl": get_cpu_transform_impl_name(),
        "gpu_transform_impl": get_gpu_transform_impl_name(),
        "normal_dtype": NUMERIC_CHAIN_NORMAL_DTYPE,
        "normal_unitize": NUMERIC_CHAIN_NORMAL_UNITIZE,
        "p_formula": NUMERIC_CHAIN_P_FORMULA,
        "q_formula": NUMERIC_CHAIN_Q_FORMULA,
        "divergence_scheme": NUMERIC_CHAIN_DIFF,
        "boundary_injection": NUMERIC_CHAIN_BOUNDARY,
        "output_mode": NUMERIC_CHAIN_OUTPUT,
    }


def get_cpu_transform_impl_name() -> str:
    if _USE_SCIPY:
        return "scipy_dst1_ortho"
    return "numpy_fft_odd_extension_dst1"


def get_gpu_transform_impl_name() -> str:
    if _HAS_TORCH:
        return "torch_fft_odd_extension_dst1"
    return "gpu_backend_unavailable"


def get_cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return out
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name", "/value"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if line.lower().startswith("name="):
                    name = line.split("=", 1)[1].strip()
                    if name:
                        return name
        except Exception:
            pass
    try:
        out = subprocess.check_output(["uname", "-p"], text=True).strip()
        if out:
            return out
    except Exception:
        pass
    return "unknown_cpu"


def get_gpu_runtime_info() -> dict[str, object]:
    info = {
        "torch_available": _HAS_TORCH,
        "cuda_available": False,
        "runtime_ok": False,
        "gpu_names": [],
        "error": None,
    }
    if torch is None:
        info["error"] = "torch_not_installed"
        return info
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if not info["cuda_available"]:
            info["error"] = "cuda_unavailable"
            return info
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        probe = torch.ones(8, dtype=torch.float32, device="cuda")
        _ = torch.fft.fft(probe)
        torch.cuda.synchronize()
        info["runtime_ok"] = True
        return info
    except Exception as exc:
        info["error"] = str(exc)
        return info


def save_numeric_chain_dump(
    dump_dir: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object] | None = None,
) -> None:
    os.makedirs(dump_dir, exist_ok=True)
    for name, value in arrays.items():
        arr = np.asarray(value, dtype=np.float32)
        np.save(os.path.join(dump_dir, f"{name}.npy"), arr)
        arr.tofile(os.path.join(dump_dir, f"{name}.bin"))

    info = get_numeric_chain_info()
    if metadata:
        info.update(metadata)
    with open(os.path.join(dump_dir, "numeric_chain_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def _normal_to_pq(normal: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    n = ensure_hwc_normal(normal).astype(np.float32)
    nx_, ny_, nz_ = n[..., 0], n[..., 1], np.clip(n[..., 2], eps, None)
    p = (-nx_ / nz_).astype(np.float32)
    q = (ny_ / nz_).astype(np.float32)
    return p, q


def _forward_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    dpdx = np.zeros_like(p, dtype=np.float32)
    dpdx[:, 1:] = p[:, 1:] - p[:, :-1]
    dqdy = np.zeros_like(q, dtype=np.float32)
    dqdy[1:, :] = q[1:, :] - q[:-1, :]
    return (dpdx + dqdy).astype(np.float32)


def _as_real_numpy(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise TypeError("DST-I only supports real-valued inputs")
    if arr.dtype.kind in "bui":
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.promote_types(arr.dtype, np.float32), copy=False)


def _dst1_fft_backend_numpy(x: np.ndarray, axis: int = -1, norm: str = "ortho") -> np.ndarray:
    if norm not in (None, "backward", "ortho"):
        raise ValueError('Only norm=None/"backward" or "ortho" is supported for DST-I')

    arr = _as_real_numpy(x)
    axis = axis % arr.ndim
    n = arr.shape[axis]
    if n < 1:
        raise ValueError("DST-I input length must be at least 1")

    arr = np.moveaxis(arr, axis, -1)
    ext_shape = arr.shape[:-1] + (2 * (n + 1),)
    ext = np.zeros(ext_shape, dtype=arr.dtype)
    ext[..., 1:n + 1] = arr
    ext[..., n + 2:] = -arr[..., ::-1]

    spectrum = np.fft.fft(ext, axis=-1)
    out = -spectrum.imag[..., 1:n + 1]
    if norm == "ortho":
        out = out / math.sqrt(2.0 * (n + 1))
    out = np.moveaxis(out, -1, axis)
    return out.astype(arr.dtype, copy=False)


def _idst1_fft_backend_numpy(x: np.ndarray, axis: int = -1, norm: str = "ortho") -> np.ndarray:
    if norm != "ortho":
        raise ValueError('This project only uses norm="ortho" for IDST-I')
    return _dst1_fft_backend_numpy(np.asarray(x), axis=axis, norm=norm)


def _torch_cuda_synchronize_if_needed(device: str | None) -> None:
    if torch is None or device is None:
        return
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _as_real_torch(x: np.ndarray | "torch.Tensor", device: str = "cuda") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch is not available")
    if torch.is_tensor(x):
        arr = x.to(device=device)
        if arr.is_complex():
            raise TypeError("DST-I only supports real-valued inputs")
        return arr.to(dtype=torch.float32)
    arr = torch.as_tensor(x, device=device)
    if arr.is_complex():
        raise TypeError("DST-I only supports real-valued inputs")
    return arr.to(dtype=torch.float32)


def _dst1_fft_backend_torch(
    x: np.ndarray | "torch.Tensor",
    axis: int = -1,
    norm: str = "ortho",
    device: str = "cuda",
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch is not available")
    if norm not in (None, "backward", "ortho"):
        raise ValueError('Only norm=None/"backward" or "ortho" is supported for DST-I')

    arr = _as_real_torch(x, device=device)
    axis = axis % arr.ndim
    n = arr.shape[axis]
    if n < 1:
        raise ValueError("DST-I input length must be at least 1")

    arr = torch.movedim(arr, axis, -1)
    ext_shape = tuple(arr.shape[:-1]) + (2 * (n + 1),)
    ext = torch.zeros(ext_shape, dtype=arr.dtype, device=arr.device)
    ext[..., 1:n + 1] = arr
    ext[..., n + 2:] = -torch.flip(arr, dims=[-1])

    spectrum = torch.fft.fft(ext, dim=-1)
    out = -spectrum.imag[..., 1:n + 1]
    if norm == "ortho":
        out = out / math.sqrt(2.0 * (n + 1))
    out = torch.movedim(out, -1, axis)
    return out


def _idst1_fft_backend_torch(
    x: np.ndarray | "torch.Tensor",
    axis: int = -1,
    norm: str = "ortho",
    device: str = "cuda",
) -> "torch.Tensor":
    if norm != "ortho":
        raise ValueError('This project only uses norm="ortho" for IDST-I')
    return _dst1_fft_backend_torch(x, axis=axis, norm=norm, device=device)


def _dst2_ortho(arr: np.ndarray) -> np.ndarray:
    return dst(dst(arr.astype(np.float32), norm="ortho").T, norm="ortho").T.astype(np.float32)


def _idst2_ortho(arr: np.ndarray) -> np.ndarray:
    return idst(idst(arr.astype(np.float32), norm="ortho").T, norm="ortho").T.astype(np.float32)


def _dst2_ortho_torch(arr: np.ndarray | "torch.Tensor", device: str = "cuda") -> "torch.Tensor":
    out = _dst1_fft_backend_torch(arr, axis=-1, norm="ortho", device=device)
    out = _dst1_fft_backend_torch(out, axis=0, norm="ortho", device=device)
    return out


def _idst2_ortho_torch(arr: np.ndarray | "torch.Tensor", device: str = "cuda") -> "torch.Tensor":
    out = _idst1_fft_backend_torch(arr, axis=-1, norm="ortho", device=device)
    out = _idst1_fft_backend_torch(out, axis=0, norm="ortho", device=device)
    return out


def _poisson_denom(n_h: int, n_w: int) -> np.ndarray:
    jj, ii = np.meshgrid(np.arange(1, n_w + 1), np.arange(1, n_h + 1), indexing="xy")
    denom = (2 * np.cos(np.pi * ii / (n_h + 1)) - 2) + (2 * np.cos(np.pi * jj / (n_w + 1)) - 2)
    denom = denom.astype(np.float32)
    denom[denom == 0] = np.inf
    return denom


def _poisson_denom_torch(n_h: int, n_w: int, device: str = "cuda") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch is not available")
    col_ids = torch.arange(1, n_w + 1, dtype=torch.float32, device=device)
    row_ids = torch.arange(1, n_h + 1, dtype=torch.float32, device=device)
    jj, ii = torch.meshgrid(col_ids, row_ids, indexing="xy")
    denom = (
        (2 * torch.cos(torch.pi * ii / (n_h + 1)) - 2)
        + (2 * torch.cos(torch.pi * jj / (n_w + 1)) - 2)
    ).to(dtype=torch.float32)
    denom = torch.where(denom == 0, torch.full_like(denom, float("inf")), denom)
    return denom


def _resize_nearest_numpy(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = shape_hw
    src_h, src_w = arr.shape
    y_idx = np.clip(np.rint(np.linspace(0, src_h - 1, dst_h)).astype(np.int32), 0, src_h - 1)
    x_idx = np.clip(np.rint(np.linspace(0, src_w - 1, dst_w)).astype(np.int32), 0, src_w - 1)
    return arr[y_idx][:, x_idx]


def _resize_linear_numpy(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = shape_hw
    src_h, src_w = arr.shape

    if Image is not None:
        pil_img = Image.fromarray(arr.astype(np.float32), mode="F")
        return np.array(pil_img.resize((dst_w, dst_h), resample=Image.BILINEAR), dtype=np.float32)

    x_src = np.arange(src_w, dtype=np.float32)
    x_dst = np.linspace(0, src_w - 1, dst_w, dtype=np.float32)
    tmp = np.stack([np.interp(x_dst, x_src, row).astype(np.float32) for row in arr], axis=0)

    y_src = np.arange(src_h, dtype=np.float32)
    y_dst = np.linspace(0, src_h - 1, dst_h, dtype=np.float32)
    out = np.stack([np.interp(y_dst, y_src, tmp[:, j]).astype(np.float32) for j in range(tmp.shape[1])], axis=1)
    return out


def _gaussian_kernel1d(size: int) -> np.ndarray:
    if size <= 1:
        return np.array([1.0], dtype=np.float32)
    radius = size // 2
    sigma = max(size / 6.0, 1e-6)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _gaussian_blur_numpy(arr: np.ndarray, size: int) -> np.ndarray:
    kernel = _gaussian_kernel1d(size)
    pad = size // 2

    tmp = np.pad(arr.astype(np.float32), ((0, 0), (pad, pad)), mode="edge")
    tmp = np.stack([np.convolve(row, kernel, mode="valid") for row in tmp], axis=0)

    out = np.pad(tmp, ((pad, pad), (0, 0)), mode="edge")
    out = np.stack([np.convolve(col, kernel, mode="valid") for col in out.T], axis=1)
    return out.astype(np.float32)


def ensure_hwc_normal(normal: np.ndarray) -> np.ndarray:
    if normal.ndim != 3:
        raise ValueError(f"normal must be 3D, got shape={normal.shape}")
    if normal.shape[-1] == 3:
        return normal.astype(np.float32)
    if normal.shape[0] == 3:
        return np.transpose(normal, (1, 2, 0)).astype(np.float32)
    raise ValueError(f"normal must be HWC or CHW with 3 channels, got shape={normal.shape}")


def normalize_normals(normal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = ensure_hwc_normal(normal)
    nlen = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.clip(nlen, eps, None)


def resize_depth_prior(depth_prior: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if depth_prior is None:
        return None
    h, w = shape_hw
    if depth_prior.shape == (h, w):
        return depth_prior.astype(np.float32)

    prior = depth_prior.astype(np.float32)
    if np.isnan(prior).any():
        # 对边界型先验优先保留 NaN 内部结构，避免线性插值把 NaN 扩散到边界附近。
        if cv2 is not None:
            return cv2.resize(prior, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        return _resize_nearest_numpy(prior, (h, w)).astype(np.float32)
    if cv2 is not None:
        return cv2.resize(prior, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return _resize_linear_numpy(prior, (h, w)).astype(np.float32)


def load_depth_prior(depth_prior_path: str | None, shape_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    if not depth_prior_path:
        return None
    prior = np.load(depth_prior_path).astype(np.float32)
    if shape_hw is not None:
        prior = resize_depth_prior(prior, shape_hw)
    return prior


def fast_poisson_from_normals(normal: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    p, q = _normal_to_pq(normal, eps=eps)
    h, w = p.shape
    f = _forward_divergence(p, q)

    inner = f[1:-1, 1:-1].copy()
    n_h = h - 2
    n_w = w - 2
    if n_h <= 0 or n_w <= 0:
        return np.zeros((h, w), dtype=np.float32)

    f_freq = _dst2_ortho(inner)
    denom = _poisson_denom(n_h, n_w)
    z_freq = f_freq / denom
    z_inner = _idst2_ortho(z_freq)

    z_rel = np.zeros((h, w), dtype=np.float32)
    z_rel[1:-1, 1:-1] = z_inner

    if mask is not None:
        valid = mask > 0
        if valid.any():
            z_rel -= z_rel[valid].mean()
        else:
            z_rel -= z_rel.mean()
    else:
        z_rel -= z_rel.mean()
    return z_rel


def surface_align_quadratic(
    z_rel: np.ndarray,
    depth_prior: np.ndarray | None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if depth_prior is None:
        return z_rel

    h, w = z_rel.shape
    if mask is not None:
        valid = (mask > 0) & np.isfinite(depth_prior)
    else:
        valid = np.isfinite(depth_prior)
    if valid.sum() < 20:
        return z_rel

    y_idx, x_idx = np.where(valid)
    z_diff = depth_prior[valid] - z_rel[valid]
    y_norm = (y_idx - h / 2) / (h / 2)
    x_norm = (x_idx - w / 2) / (w / 2)
    a = np.column_stack((
        np.ones_like(x_norm),
        x_norm,
        y_norm,
        x_norm ** 2,
        y_norm ** 2,
        x_norm * y_norm,
    ))
    coeff, _, _, _ = np.linalg.lstsq(a, z_diff, rcond=None)

    yy, xx = np.mgrid[0:h, 0:w]
    yy_norm = (yy - h / 2) / (h / 2)
    xx_norm = (xx - w / 2) / (w / 2)
    error_surface = (
        coeff[0]
        + coeff[1] * xx_norm
        + coeff[2] * yy_norm
        + coeff[3] * xx_norm ** 2
        + coeff[4] * yy_norm ** 2
        + coeff[5] * xx_norm * yy_norm
    )
    return z_rel + error_surface


def plane_align(z_rel: np.ndarray, depth_prior: np.ndarray | None, mask: np.ndarray | None = None) -> np.ndarray:
    if depth_prior is None:
        return z_rel

    h, w = z_rel.shape
    yy, xx = np.mgrid[0:h, 0:w]
    if mask is not None:
        valid = (mask > 0) & np.isfinite(depth_prior)
    else:
        valid = np.isfinite(depth_prior)
    if valid.sum() < 20:
        return z_rel

    a = np.stack([xx[valid], yy[valid], np.ones_like(xx[valid])], axis=1)
    b = (depth_prior[valid] - z_rel[valid]).astype(np.float32)
    theta, *_ = np.linalg.lstsq(a, b, rcond=None)
    ax, by, c = theta
    return z_rel + (ax * xx + by * yy + c)


def hybrid_reconstruct(
    normal: np.ndarray,
    depth_prior: np.ndarray | None,
    mask: np.ndarray | None = None,
    beta: float = 0.0,
) -> np.ndarray:
    z_rel = fast_poisson_from_normals(normal, mask=mask)
    if depth_prior is None:
        return z_rel

    z_align = plane_align(z_rel, depth_prior, mask=mask)
    if beta <= 0:
        return z_align

    k = max(5, int(0.1 * min(z_align.shape)))
    if k % 2 == 0:
        k += 1
    if cv2 is not None:
        lp_prior = cv2.GaussianBlur(depth_prior.astype(np.float32), (k, k), 0)
        lp_align = cv2.GaussianBlur(z_align.astype(np.float32), (k, k), 0)
    else:
        lp_prior = _gaussian_blur_numpy(depth_prior.astype(np.float32), k)
        lp_align = _gaussian_blur_numpy(z_align.astype(np.float32), k)
    high = z_align - lp_align
    z_mix = lp_prior + high
    return (1 - beta) * z_align + beta * z_mix


def poisson_dirichlet_from_normals_rect_boundary(
    normal, boundary_prior, eps=1e-6, transform_backend="cpu", torch_device="cuda", return_debug=False,
):
    """Historical normal-input API and frozen numeric chain, unchanged."""
    return _poisson_rect_boundary_impl(normal, boundary_prior, eps, transform_backend, torch_device, return_debug)


def reconstruct_from_gradients(gradients, boundary_prior, transform_backend="cpu", torch_device="cuda", return_debug=False):
    """Integrate HWC p/q directly using the same rectangular Dirichlet solver."""
    result = _poisson_rect_boundary_impl(None, boundary_prior, transform_backend=transform_backend,
        torch_device=torch_device, return_debug=return_debug, gradients=np.asarray(gradients, dtype=np.float32))
    if result is None:
        raise ValueError("Prior must contain a single interior rectangular NaN region")
    return result


def _poisson_rect_boundary_impl(
    normal: np.ndarray,
    boundary_prior: np.ndarray,
    eps: float = 1e-6,
    transform_backend: str = "cpu",
    torch_device: str = "cuda",
    return_debug: bool = False,
    gradients: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | dict[str, int]]] | None:
    """
    Solve the frozen rectangular Dirichlet Poisson problem from a full normal map.

    Inputs:
    - `normal` is still the full HWC normal image. The solver does not ask the
      caller to crop anything manually.
    - `boundary_prior` is a full-resolution depth matrix whose interior solve
      region is marked by a single rectangular NaN block. The four depth edges
      immediately surrounding that NaN block are the Dirichlet boundary values.

    Frozen numerical chain:
    1. Detect the rectangular NaN interior in `boundary_prior`.
    2. Expand by one pixel in each direction to extract the local normal ROI
       needed by forward differences.
    3. Convert ROI normals to slopes with p = -nx / nz, q = ny / nz.
    4. Build divergence with the frozen forward-difference scheme.
    5. Keep only the interior RHS and inject the four boundary edges.
    6. Solve the rectangular Poisson system with DST-I / IDST-I, norm="ortho".
    7. Write the solved interior back into a full-size absolute depth map.

    Return:
    - `None` when the boundary prior does not match the expected single
      rectangular NaN-interior format.
    - Otherwise the reconstructed absolute depth map.
    - When `return_debug=True`, also return all frozen intermediate tensors so
      offline regression and FPGA stage-by-stage comparison can reuse them.
    """
    if gradients is None:
        normal = ensure_hwc_normal(normal)
    elif gradients.shape != (*boundary_prior.shape, 2) or not np.isfinite(gradients).all():
        raise ValueError("Expected finite HWC two-channel gradients matching prior")
    h, w = boundary_prior.shape
    # The NaN block marks the unknown depth interior that should be reconstructed.
    # Everything around it must already contain valid boundary depth values.
    nan_mask = np.isnan(boundary_prior)
    if not nan_mask.any():
        return None

    ys, xs = np.where(nan_mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Freeze the ROI definition to a single filled rectangle of NaNs. If the NaN
    # area is irregular, or touches the image border, this solver refuses it
    # instead of silently changing the problem definition.
    if not np.all(nan_mask[y0:y1 + 1, x0:x1 + 1]):
        return None
    if y0 == 0 or x0 == 0 or y1 == h - 1 or x1 == w - 1:
        return None

    profile_ms: dict[str, float] = {
        "gpu_copy_to_ms": 0.0,
        "gpu_copy_from_ms": 0.0,
    }
    solver_t0 = time.perf_counter()

    # 取NaN区域紧邻的四条边作为Dirichlet边界注入值
    top_row = boundary_prior[y0 - 1, x0:x1 + 1]
    bottom_row = boundary_prior[y1 + 1, x0:x1 + 1]
    left_col = boundary_prior[y0:y1 + 1, x0 - 1]
    right_col = boundary_prior[y0:y1 + 1, x1 + 1]
    if any(np.isnan(v).any() for v in [top_row, bottom_row, left_col, right_col]):
        return None

    # Expand the interior by one pixel on each side before computing p/q and
    # divergence. This "sub" ROI is required because forward differences at the
    # unknown interior need one layer of neighboring slope samples around it.
    #
    # Shapes:
    # - interior solve region:  (n_h, n_w)     == (y1-y0+1, x1-x0+1)
    # - expanded normal ROI:    (n_h+2, n_w+2) == sub_normals.shape[:2]
    ny0, ny1 = y0 - 1, y1 + 1
    nx0, nx1 = x0 - 1, x1 + 1
    sub_normals = normal[ny0:ny1 + 1, nx0:nx1 + 1, :].astype(np.float32) if gradients is None else None

    # Frozen slope convention for the FPGA reference chain.
    step_t0 = time.perf_counter()
    if gradients is None:
        p, q = _normal_to_pq(sub_normals, eps=eps)
    else:
        sub = gradients[ny0:ny1 + 1, nx0:nx1 + 1].astype(np.float32)
        p, q = sub[..., 0], sub[..., 1]
    profile_ms["pq_ms"] = (time.perf_counter() - step_t0) * 1000.0

    # f is the full divergence over the expanded ROI. Its outermost one-pixel
    # ring exists only to support the inner region and will be discarded next.
    step_t0 = time.perf_counter()
    f = _forward_divergence(p, q)
    profile_ms["divergence_ms"] = (time.perf_counter() - step_t0) * 1000.0

    # Keep only the true unknown interior. This is the RHS before Dirichlet
    # injection, so its shape must exactly match the NaN rectangle.
    inner = f[1:-1, 1:-1].copy()
    n_h = y1 - y0 + 1
    n_w = x1 - x0 + 1
    if inner.shape != (n_h, n_w):
        return None
    if top_row.shape[0] != n_w or bottom_row.shape[0] != n_w:
        return None
    if left_col.shape[0] != n_h or right_col.shape[0] != n_h:
        return None

    # Dirichlet injection:
    # move the known boundary depths to the RHS of the discrete Poisson system.
    # Only the first/last row and first/last column of the interior are affected.
    step_t0 = time.perf_counter()
    inner[0, :] -= top_row
    inner[-1, :] -= bottom_row
    inner[:, 0] -= left_col
    inner[:, -1] -= right_col
    profile_ms["boundary_injection_ms"] = (time.perf_counter() - step_t0) * 1000.0

    # Solve Laplacian(z) = inner on a rectangle with zero-free orthogonal DST-I:
    # 1) transform RHS to the sine basis
    # 2) divide by the precomputed eigenvalues of the discrete Laplacian
    # 3) inverse transform back to spatial depth
    if transform_backend == "gpu":
        _torch_cuda_synchronize_if_needed(torch_device)
        step_t0 = time.perf_counter()
        inner_t = _as_real_torch(inner, device=torch_device)
        _torch_cuda_synchronize_if_needed(torch_device)
        profile_ms["gpu_copy_to_ms"] = (time.perf_counter() - step_t0) * 1000.0

        _torch_cuda_synchronize_if_needed(torch_device)
        step_t0 = time.perf_counter()
        f_freq_t = _dst2_ortho_torch(inner_t, device=torch_device)
        _torch_cuda_synchronize_if_needed(torch_device)
        profile_ms["dst2_ms"] = (time.perf_counter() - step_t0) * 1000.0

        _torch_cuda_synchronize_if_needed(torch_device)
        step_t0 = time.perf_counter()
        denom_t = _poisson_denom_torch(n_h, n_w, device=torch_device)
        _torch_cuda_synchronize_if_needed(torch_device)
        profile_ms["denom_ms"] = (time.perf_counter() - step_t0) * 1000.0

        _torch_cuda_synchronize_if_needed(torch_device)
        step_t0 = time.perf_counter()
        recip_denom_t = 1.0 / denom_t
        z_freq_t = f_freq_t * recip_denom_t
        _torch_cuda_synchronize_if_needed(torch_device)
        profile_ms["recip_denom_ms"] = (time.perf_counter() - step_t0) * 1000.0

        _torch_cuda_synchronize_if_needed(torch_device)
        step_t0 = time.perf_counter()
        z_inner_t = _idst2_ortho_torch(z_freq_t, device=torch_device)
        _torch_cuda_synchronize_if_needed(torch_device)
        profile_ms["idst2_ms"] = (time.perf_counter() - step_t0) * 1000.0

        _torch_cuda_synchronize_if_needed(torch_device)
        step_t0 = time.perf_counter()
        f_freq = f_freq_t.detach().cpu().numpy().astype(np.float32)
        denom = denom_t.detach().cpu().numpy().astype(np.float32)
        recip_denom = recip_denom_t.detach().cpu().numpy().astype(np.float32)
        z_inner = z_inner_t.detach().cpu().numpy().astype(np.float32)
        _torch_cuda_synchronize_if_needed(torch_device)
        profile_ms["gpu_copy_from_ms"] = (time.perf_counter() - step_t0) * 1000.0
    else:
        step_t0 = time.perf_counter()
        f_freq = _dst2_ortho(inner)
        profile_ms["dst2_ms"] = (time.perf_counter() - step_t0) * 1000.0

        step_t0 = time.perf_counter()
        denom = _poisson_denom(n_h, n_w)
        profile_ms["denom_ms"] = (time.perf_counter() - step_t0) * 1000.0

        step_t0 = time.perf_counter()
        recip_denom = 1.0 / denom
        z_freq = f_freq * recip_denom
        profile_ms["recip_denom_ms"] = (time.perf_counter() - step_t0) * 1000.0

        step_t0 = time.perf_counter()
        z_inner = _idst2_ortho(z_freq)
        profile_ms["idst2_ms"] = (time.perf_counter() - step_t0) * 1000.0

    # Write the solved interior back into the original full-size matrix so the
    # output remains an absolute depth image in the same coordinates as input.
    step_t0 = time.perf_counter()
    depth_out = boundary_prior.copy()
    depth_out[y0:y1 + 1, x0:x1 + 1] = z_inner
    depth_out = depth_out.astype(np.float32)
    profile_ms["writeback_ms"] = (time.perf_counter() - step_t0) * 1000.0
    profile_ms["solver_total_ms"] = (time.perf_counter() - solver_t0) * 1000.0

    if not return_debug:
        return depth_out

    # The debug payload is intentionally stage-aligned with software regression,
    # PS reference code, and future FPGA module bring-up:
    # - roi_normal / p / q / divergence / inner_rhs for Stage-1
    # - dst_freq / denom for spectral-domain checks
    # - depth_out for end-to-end validation
    debug = {
        "normal_in": normal.astype(np.float32) if normal is not None else None,
        "roi_normal": sub_normals.astype(np.float32) if sub_normals is not None else None,
        "boundary_top": top_row.astype(np.float32),
        "boundary_bottom": bottom_row.astype(np.float32),
        "boundary_left": left_col.astype(np.float32),
        "boundary_right": right_col.astype(np.float32),
        "p": p.astype(np.float32),
        "q": q.astype(np.float32),
        "divergence": f.astype(np.float32),
        "inner_rhs": inner.astype(np.float32),
        "dst_freq": f_freq.astype(np.float32),
        "denom": denom.astype(np.float32),
        "recip_denom": recip_denom.astype(np.float32),
        "depth_out": depth_out.astype(np.float32),
        "profile_ms": profile_ms,
        "transform_backend": transform_backend,
        "roi": {
            "y0": int(y0),
            "y1": int(y1),
            "x0": int(x0),
            "x1": int(x1),
            "ny0": int(ny0),
            "ny1": int(ny1),
            "nx0": int(nx0),
            "nx1": int(nx1),
        },
    }
    return depth_out, debug


def reconstruct_with_boundary_prior(
    normal: np.ndarray,
    depth_prior: np.ndarray | None,
    normal_gt: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    beta: float = 0.0,
    pixmm: float | None = None,
    transform_backend: str = "cpu",
    torch_device: str = "cuda",
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | dict[str, int]]]:
    del normal_gt, mask, beta, pixmm

    if depth_prior is None:
        raise ValueError("Frozen numeric chain requires a boundary prior matrix with NaN interior.")
    if not np.isnan(depth_prior).any():
        raise ValueError("Frozen numeric chain expects boundary prior with NaN interior and valid four-edge values.")

    depth_dir = poisson_dirichlet_from_normals_rect_boundary(
        normal,
        depth_prior,
        transform_backend=transform_backend,
        torch_device=torch_device,
        return_debug=return_debug,
    )
    if depth_dir is None:
        raise ValueError("Boundary prior does not describe a valid rectangular Dirichlet problem.")
    return depth_dir


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    valid = np.isfinite(d)
    if not np.any(valid):
        norm = np.zeros_like(d, dtype=np.float32)
    else:
        mn = float(np.nanmin(d[valid]))
        mx = float(np.nanmax(d[valid]))
        if mx <= mn:
            mx = mn + 1e-6
        norm = np.clip((d - mn) / (mx - mn), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    if cv2 is not None:
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    r = np.clip(1.5 * norm - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - 2.0 * np.abs(norm - 0.5), 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * norm, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32) * 255.0
