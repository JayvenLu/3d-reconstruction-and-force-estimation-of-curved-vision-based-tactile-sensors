import numpy as np
import cupy as cp
import cv2


# ==========================================
# 1. 核函数定义
# ==========================================

def tps_kernel(r: cp.ndarray) -> cp.ndarray:
    """薄板样条核函数 (TPS): r^2 * log(r)，避免 log(0)。"""
    r = cp.maximum(r, 1e-9)
    return (r ** 2) * cp.log(r)


def get_kernel_func(name='gaussian', epsilon=50.0):
    """
    返回指定的核函数。

    参数:
    epsilon (sigma): 对于 Gaussian/InvMultiquadric 是带宽参数，控制影响范围。
                     物理意义：大约是“影响半径”的大小 (像素)。
    """

    if name == 'tps':
        # 使用与 RBF_0 中一致的 TPS 核实现
        return tps_kernel
    elif name == 'gaussian':
        # Gaussian: exp(-r^2 / (2*sigma^2))
        # 特点：局部性强，随距离衰减。符合胶体按压的局部变形特性。
        # epsilon 在这里充当 sigma
        sigma = epsilon

        def kernel(r):
            return cp.exp(-(r ** 2) / (2 * (sigma ** 2)))

        return kernel

    elif name == 'inverse_multiquadric':
        # Inverse Multiquadric: 1 / sqrt(r^2 + c^2)
        # 特点：模拟点载荷在弹性半空间上的位移衰减 (类似 1/r)，物理上最接近。
        c = epsilon

        def kernel(r):
            return 1.0 / cp.sqrt(r ** 2 + c ** 2)

        return kernel

    elif name == 'linear':
        # Linear: -r
        # 特点：简单的锥形衰减
        def kernel(r):
            return -r

        return kernel

    else:
        raise ValueError(f"Unknown kernel: {name}")


# ==========================================
# 2. TPS 专用求解逻辑（与 RBF_0 保持一致）
# ==========================================

def solve_and_map_tps_gpu(control_points: cp.ndarray,
                          control_values: cp.ndarray,
                          grid_points: cp.ndarray,
                          regularization: float = 0.5) -> cp.ndarray:
    """使用 TPS 核进行线性求解，并在坐标归一化到 [0,1] 后施加正则，与 RBF_0 中实现保持一致。"""
    M = control_points.shape[0]
    N = grid_points.shape[0]

    if M < 3:
        return cp.zeros((N,), dtype=cp.float32)

    # 1) 坐标归一化：用全图网格的最大坐标作为尺度，使不同分辨率下 regularization 含义一致
    max_val = cp.max(grid_points)  # 例如 640.0
    c_pts_norm = control_points / max_val
    g_pts_norm = grid_points / max_val

    # 2) 构建系统矩阵 K (M, M)
    dists_cc = cp.linalg.norm(c_pts_norm[:, None, :] - c_pts_norm[None, :, :], axis=2)
    K_matrix = tps_kernel(dists_cc)

    # 3) 添加正则化
    K_matrix += cp.eye(M, dtype=cp.float32) * regularization

    # 4) 求解权重
    try:
        weights = cp.linalg.solve(K_matrix, control_values)
    except Exception as e:
        print(f"[RBF Error] TPS Singular matrix: {e}")
        return cp.zeros((N,), dtype=cp.float32)

    # 5) 映射到全图网格
    dists_gc = cp.linalg.norm(g_pts_norm[:, None, :] - c_pts_norm[None, :, :], axis=2)
    Phi = tps_kernel(dists_gc)
    interpolated = Phi @ weights
    return interpolated


# ==========================================
# 3. 通用 RBF 求解逻辑（Gaussian / Inverse MQ 等仍用原来的 solve_and_map_gpu）
# ==========================================

def solve_and_map_gpu(control_points, control_values, grid_points,
                      kernel_func, regularization=0.1):
    """
    通用 RBF 求解器
    """
    M = control_points.shape[0]
    N = grid_points.shape[0]

    if M < 3:
        return cp.zeros((N,), dtype=cp.float32)

    # --- 计算控制点之间的距离矩阵 ---
    # shape: (M, M)
    dists_cc = cp.linalg.norm(control_points[:, None, :] - control_points[None, :, :], axis=2)

    # 构建系统矩阵 K
    K_matrix = kernel_func(dists_cc)

    # --- 正则化 ---
    # TPS 的值域很大，regularization 通常较小 (0.01~0.5)
    # Gaussian 的值域是 [0, 1]，regularization 也应适配 (0.001 ~ 0.1)
    K_matrix += cp.eye(M, dtype=cp.float32) * regularization

    # --- 求解权重 weights (M,) ---
    try:
        weights = cp.linalg.solve(K_matrix, control_values)
    except Exception as e:
        print(f"[RBF Error] Singular matrix: {e}")
        return cp.zeros((N,), dtype=cp.float32)

    # --- 映射到全图网格 ---
    # 内存优化：如果 N 很大 (640*480=30万)，一次性计算 (N, M) 矩阵可能会爆显存
    # 这里我们分块计算
    batch_size = 10000  # 每次处理 10000 个像素
    interpolated = cp.zeros(N, dtype=cp.float32)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        # grid_batch: (batch, 2)
        grid_batch = grid_points[i:end]

        # dists_gc: (batch, M)
        dists_gc = cp.linalg.norm(grid_batch[:, None, :] - control_points[None, :, :], axis=2)

        Phi_batch = kernel_func(dists_gc)
        interpolated[i:end] = Phi_batch @ weights

    return interpolated


# ==========================================
# 3. 核心接口
# ==========================================

def compute_displacement_field_in_raw(src_points,
                                      dst_points,
                                      height,
                                      width,
                                      mask=None,
                                      epsilon=50.0,  # 这里的 epsilon 现在对于 Gaussian 来说是 sigma
                                      regularization=0.05,
                                      kernel_type='gaussian',  # 新增参数: 'tps', 'gaussian', 'inverse_multiquadric'
                                      show=False):
    """
    基于 RBF 的致密位移场生成。

    kernel_type 推荐:
      - 'gaussian': 最平滑且局部，适合大多数 Gel 触摸。epsilon 建议设为 30~80 (像素)。
      - 'inverse_multiquadric': 物理衰减更真实。
      - 'tps': 旧方法，适合大变形但不适合局部触摸。
    """
    # 1. 检查输入
    if src_points is None or dst_points is None or src_points.shape[0] < 3:
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32)

    # ---------------------------------------------------------
    # 边界锚点策略 (对于 Gaussian RBF，边界锚点其实不那么重要了，因为会自动衰减为0，但保留也无妨)
    # ---------------------------------------------------------
    margin = 0
    step = max(min(width, height) // 20, 8)

    anchors = []

    # if kernel_type == 'tps':
    #     # 仅 TPS 时添加四周边界锚点（与 RBF_0 保持一致）
    #     # 上边界 y = margin
    #     y_top = margin
    #     for x in range(margin, width - margin + 1, step):
    #         anchors.append([x, y_top])
    #     # 下边界 y = height - margin - 1
    #     y_bottom = height - margin - 1
    #     for x in range(margin, width - margin + 1, step):
    #         anchors.append([x, y_bottom])
    #     # 左右边界，避免重复角点
    #     x_left = margin
    #     x_right = width - margin - 1
    #     for y in range(margin + step, height - margin - step + 1, step):
    #         anchors.append([x_left, y])
    #         anchors.append([x_right, y])

    if len(anchors) > 0:
        anchors = np.array(anchors, dtype=np.float32)
        # 只有当使用 TPS 时，边界锚点才非常关键，用于防止发散。
        # 对于 Gaussian，如果 epsilon 设置合理，自然会衰减。但加上锚点更稳健。
        src_points_aug = np.vstack([src_points, anchors])
        dst_points_aug = np.vstack([dst_points, anchors])
    else:
        src_points_aug = src_points
        dst_points_aug = dst_points

    # ---------------------------------------------------------

    # 2. 准备数据 (CPU -> GPU)
    disp = dst_points_aug - src_points_aug
    pts_gpu = cp.asarray(src_points_aug, dtype=cp.float32)
    vals_dx = cp.asarray(disp[:, 0], dtype=cp.float32)
    vals_dy = cp.asarray(disp[:, 1], dtype=cp.float32)

    # 生成全图网格
    x_grid = cp.arange(width, dtype=cp.float32)
    y_grid = cp.arange(height, dtype=cp.float32)
    xx, yy = cp.meshgrid(x_grid, y_grid)
    grid_points = cp.vstack((xx.ravel(), yy.ravel())).T

    # 3. 根据 kernel_type 选择求解路径
    if kernel_type == 'tps':
        # 使用 TPS 专用求解（与 RBF_0 一致，忽略 epsilon，改用 regularization）
        du_gpu = solve_and_map_tps_gpu(pts_gpu, vals_dx, grid_points, regularization)
        dv_gpu = solve_and_map_tps_gpu(pts_gpu, vals_dy, grid_points, regularization)
    else:
        # Gaussian / inverse_multiquadric / linear 等仍用通用 RBF 求解
        kernel_func = get_kernel_func(kernel_type, epsilon)
        du_gpu = solve_and_map_gpu(pts_gpu, vals_dx, grid_points, kernel_func, regularization)
        dv_gpu = solve_and_map_gpu(pts_gpu, vals_dy, grid_points, kernel_func, regularization)

    # 5. 后处理
    du_field = du_gpu.reshape(height, width).get()
    dv_field = dv_gpu.reshape(height, width).get()

    # 可选：距离遮罩 (Distance Masking)
    # 如果像素距离最近的控制点太远，强制归零。这能消除远处的噪声。
    # 简单实现：利用前面的计算结果，或者简单的距离变换。
    # 这里如果不做，Gaussian RBF 也会自动衰减到接近 0。

    if mask is not None:
        mask_bool = (mask > 0)
        du_field[~mask_bool] = 0
        dv_field[~mask_bool] = 0

    if show:
        mag = np.sqrt(du_field ** 2 + dv_field ** 2)
        print(f"[RBF Viz] Kernel: {kernel_type}, Sigma: {epsilon}, Max Disp: {np.max(mag):.4f} px")
        vmax = np.percentile(mag, 99) + 1e-6
        vis = (np.clip(mag / vmax, 0, 1) * 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.imshow('RBF Magnitude', vis_color)

    return du_field, dv_field
