# 目前使用的版本
import os
import json
import csv
import random
import numpy as np
import cv2
import trimesh
import pyrender

# DATA_ROOT = 'data/20251123-095400/train'    # 三指 内窥镜 标记点1号
# DATA_ROOT = 'data/20250918-161136/test'    # 自研五指1号 大数据集
# DATA_ROOT = 'data/20251205_173723/train'    # 曲面三指1号
# DATA_ROOT = 'data/20260106_210820/test'    # 多光谱三指1号
# DATA_ROOT = 'data/20260114_120750/test'    # 多光谱三指3号 1920×1080
# DATA_ROOT = 'data/20260129_114800/train'    # 多光谱三指3号 1280×720
# DATA_ROOT = 'data/20260301_101000/test'    # 多光谱三指3号
# DATA_ROOT = 'data/20260306_193600/test'    # 多光谱三指3号 大球压头
# DATA_ROOT = 'data/20260310_151840/train'    # 多光谱三指3号 小球压头
# DATA_ROOT = 'data/20260421_094400/val'    # 多光谱三指3号 128*128
# DATA_ROOT = 'force_data/20260312_120630'    # 多光谱三指3号 小球压头
# DATA_ROOT = 'data/20260317_093500/val'    # 多光谱三指3号 小球压头
# DATA_ROOT = 'data/20260328_121407/train'    # 内窥镜三指1号
DATA_ROOT = 'data/20260404_103156/train'    # 内窥镜三指2号
# DATA_ROOT = 'data/20260417_112900/train'    # 内窥镜三指2号 128*128


IMAGE_DIR = os.path.join(DATA_ROOT, 'images')
CSV_PATH = os.path.join(DATA_ROOT, 'force_log.csv')
CALIB_FILE = os.path.join(DATA_ROOT, 'manual_calib.json')   # 删掉重新手动标定
NORMAL_DIR = os.path.join(DATA_ROOT, 'normal_gt')
DEPTH_DIR = os.path.join(DATA_ROOT, 'depth_gt')
MASK_DIR = os.path.join(DATA_ROOT, 'mask')
NORMAL_COMPLETE_DIR = os.path.join(DATA_ROOT, 'normal_complete_gt')
DH_DIR = os.path.join(DATA_ROOT, 'dh')  # 新增：位移场数据目录
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(NORMAL_COMPLETE_DIR, exist_ok=True)

# MESH_PATH = 'mesh/finger3_surface.STL'  # 平面三指
# MESH_PATH = 'mesh/finger_background.STL'    # 自研五指
MESH_PATH = 'mesh/sanzhi_bg.STL'    # 曲面三指
SAMPLE_COUNT = 200  # 手动标定数量
RANDOM_SEED = 42

# 交互控制与可视化参数
MANUAL_STEP_COARSE = 0.5   # 粗调步长（mm）
MANUAL_STEP_FINE = 0.1     # 细调步长（mm）
VISUALIZE_INTERVAL = 5    # 自动阶段每隔多少个样本可视化一次，0 表示不自动可视化
VISUALIZE_DELAY_MS = 400   # 自动可视化时每张图停留时间
VIS_WINDOW_HEIGHT = 1200    # 固定可视化窗口高度，宽度按比例自适应

# 是否在自动生成每个样本的小球位置后进行手动微调
ENABLE_PER_SAMPLE_MANUAL_FINE_TUNE = True  # 如不需要微调，可改为 False

# 手动模式下使用的固定旋转矩阵（机床坐标 -> 渲染坐标）
# 请根据需要修改此矩阵的数值
MANUAL_ROTATION_MATRIX = np.array([
    [0.5315, 0, 0.8470],
    [0.0, 1.0, 0.0],
    [-0.8470, 0.0, 0.5315],
], dtype=np.float64)

USE_MANUAL_ROT = None  # 占位，在 main 中由用户选择

# 新增：控制是否使用“链式微调”策略
USE_CHAIN_TUNING = True

# 新增：是否开启自动估计小球位置并批量生成
AUTO_ESTIMATE_SPHERE_POS = False

# 法线计算配置
# True: 先参考 dataset 的 render->pixel 旋转，将渲染 z 转到像素坐标系下，再计算法线
# False: 保持旧版，直接对渲染 depth 做中心差分
CONVERT_RENDER_Z_TO_PIXEL_FOR_GRADIENT = False

# 法向量可视化时使用的像素尺寸（mm/pixel）。
# 仅影响窗口显示，不影响保存到 normal*.npy 的原始法向量。
# NORMAL_VIS_PIXMM = 0.034    # 内窥镜三指
NORMAL_VIS_PIXMM = 0.1275    # 内窥镜三指 128*128

# # 三指 内窥镜 标记点 1号
# camera_pos = [13.94878765, 16.65199845, 10.36364977]  # x,y,z
# pitch = 152.14 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
# yaw = -0.92  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
# roll = -90.33  # 绕Z轴的旋转角度
# fx = 630.30
# fy = 618.78
# cx = 339.18
# cy = 281.96

# # 自研五指传感器相机参数2
# camera_pos = [11.999999999999973, 15.89999999999996, 11.899999999999974]  # x,y,z
# pitch = -35.90000000000024       # pitch,yaw,roll
# yaw = 175.00000000000028
# roll = 10.699999999999978
# fx = 1190
# fy = 1190
# cx = 320
# cy = 240

# # 曲面三指1号
# camera_pos = [11.03058961999993, 16.656628480000002, 2.657504859999958]  # x,y,z
# pitch = 154.72000000000136 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
# yaw = 3.6700000000000017  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
# roll = -93.55999999999987  # 绕Z轴的旋转角度
# fx = 964.87
# fy = 1010
# cx = 372.46
# cy = 293.78

# 多光谱三指1号
# camera_pos = [16.316706479999993, 16.18688785, 17.86220692]
# pitch = 135.55
# yaw = -0.94
# roll = -91.11
# fx = 566.1
# fy = 508.1
# cx = 945.5
# cy = 763.2

# 多光谱三指3号 1920×1080
# camera_pos = [15.61670647999999, 15.88688785, 18.462206920000007]
# pitch = 135.55
# yaw = -0.94
# roll = -90.61
# fx = 606.1
# fy = 548.1
# cx = 945.5
# cy = 763.2

# 多光谱三指3号 128*128
# camera_pos = [11.516706480000005, 16.286887850000003, 20.66220692000004]
# pitch = 135.55
# yaw = -0.94
# roll = -90.11
# fx = 81.8
# fy = 62.5
# cx = 60.0
# cy = 61.3

# 多光谱三指 1920×1080 20260316_rig
# camera_pos = [15.516706479999991, 17.086887850000014, 17.562206919999994]
# pitch = 133.05
# yaw = 0.06
# roll = -90.61
# fx = 616.1
# fy = 578.1
# cx = 940.5
# cy = 763.2

# 内窥镜三指1号相机参数 其他默认
# pyrender/OpenGL 默认相机朝向:相机默认朝向 -Z 轴,X 轴向右，Y 轴向上
# camera_pos = [3.399999999999999, 15.99999999999996, 1.6000000000000012]  # x,y,z
# pitch = -212.20 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
# yaw = 0.10  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
# roll = -90.20  # 绕Z轴的旋转角度
# fx = 1070.0
# fy = 930.0
# cx = 320
# cy = 230

# 内窥镜三指2号相机参数 其他默认
# pyrender/OpenGL 默认相机朝向:相机默认朝向 -Z 轴,X 轴向右，Y 轴向上
camera_pos = [4.6999999999999975, 15.89999999999996, 1.6000000000000012]  # x,y,z
pitch = -210.20 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
yaw = 2.60  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
roll = -86.70 # 绕Z轴的旋转角度
fx = 1100.0
fy = 860.0
cx = 320
cy = 230

# 内窥镜三指2号相机参数 128*128
# camera_pos = [4.6999999999999975, 15.89999999999996, 1.6000000000000012]
# pitch = -210.20
# yaw = 2.60
# roll = -86.70
# fx = 293.33
# fy = 229.33
# cx = 64.00
# cy = 61.33

# 多光谱三指3号 1280×720
# camera_pos = [16.86800519, 15.90078114, 19.79845209]
# pitch = 127.87
# yaw = -1.17
# roll = -89.27
# fx = 489.4
# fy = 426.3
# cx = 650.8
# cy = 513.8

def load_force_log(csv_path):
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['id'])
            # 读取机床坐标
            pos = np.array([float(row['x']), float(row['y']), float(row['z'])], dtype=np.float64)
            # 尝试读取力值，如果不存在则设为 0.0
            fx = 0.0
            fy = 0.0
            fz = 0.0
            
            # 尝试读取 fx
            if 'fx' in row:
                fx = float(row['fx'])
            elif 'Fx' in row:
                fx = float(row['Fx'])
            
            # 尝试读取 fy
            if 'fy' in row:
                fy = float(row['fy'])
            elif 'Fy' in row:
                fy = float(row['Fy'])
            
            # 尝试读取 fz
            if 'fz' in row:
                fz = float(row['fz'])
            elif 'Fz' in row:
                fz = float(row['Fz'])
            
            # 尝试读取 phase
            phase = row.get('phase', 'N/A')
            
            # 尝试读取 angle_deg
            angle_deg = 'N/A'
            if 'angle_deg' in row:
                try:
                    angle_deg = float(row['angle_deg'])
                except (ValueError, TypeError):
                    angle_deg = 'N/A'
            
            # 尝试读取 target
            target = row.get('target', 'N/A')

            # 将数据存为字典: {'pos': np.array(...), 'fx': float, 'fy': float, 'fz': float, 'phase': str, 'angle_deg': float or str, 'target': str}
            data[idx] = {
                'pos': pos, 
                'fx': fx, 
                'fy': fy, 
                'fz': fz,
                'phase': phase,
                'angle_deg': angle_deg,
                'target': target
            }
    return data


def get_vertical_ids(csv_path):
    ids = []
    if not os.path.exists(csv_path):
        return ids
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('phase') == 'vertical':
                try:
                    ids.append(int(row['id']))
                except ValueError:
                    continue
    return sorted(list(set(ids)))


def select_random_samples(ids, k):
    rng = random.Random(RANDOM_SEED)
    return rng.sample(ids, min(k, len(ids)))


def select_spread_samples(log_dict, k):
    """从所有样本 ID 中按等间距均匀选取 k 个样本。"""
    if not log_dict or k <= 0:
        return []
    ids_sorted = sorted(log_dict.keys())
    n = len(ids_sorted)
    if n <= k:
        return ids_sorted
    indices = np.linspace(0, n - 1, num=k, dtype=int)
    indices = sorted(set(int(i) for i in indices))
    if len(indices) < k:
        all_idx = list(range(n))
        remain = [i for i in all_idx if i not in indices]
        for i in remain:
            indices.append(i)
            if len(indices) >= k:
                break
        indices = sorted(indices)
    return [ids_sorted[i] for i in indices[:k]]


def collect_ids_with_dh(dh_dir):
    """在给定 dh 目录下，收集存在 x_dh.npy 的 x（int），返回排序好的 id 列表。"""
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
    print(f"[信息] 在 dh 目录 {dh_dir} 中共发现 {len(ids)} 个带位移场的样本 id。")
    return ids


def _rotation_matrix_x(deg):
    rad = np.deg2rad(deg)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float32)


def _rotation_matrix_y(deg):
    rad = np.deg2rad(deg)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float32)


def _rotation_matrix_z(deg):
    rad = np.deg2rad(deg)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


# 与 dataset.py 保持一致的 render -> pixel 坐标轴旋转
_RENDER_TO_PIXEL_AXES_ROT = _rotation_matrix_y(24.0) @ _rotation_matrix_z(-90.0)
RENDER_TO_PIXEL_ROT = _RENDER_TO_PIXEL_AXES_ROT.T


def get_camera_matrix(pos, pitch, yaw, roll):
    pitch_rad, yaw_rad, roll_rad = np.radians([pitch, yaw, roll])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                   [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
    Ry = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                   [0, 1, 0],
                   [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
    Rz = np.array([[np.cos(roll_rad), -np.sin(roll_rad), 0],
                   [np.sin(roll_rad), np.cos(roll_rad), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _convert_render_z_to_pixel_z(depth):
    depth = np.asarray(depth, dtype=np.float32)
    zeros = np.zeros_like(depth, dtype=np.float32)

    # 这里处理的是“渲染 z 标量场”本身，而不是 dataset 里的 dz，
    # 因此不额外取负号，直接按 render z 轴正方向旋到 pixel 坐标系。
    depth_render = np.stack([zeros, zeros, depth], axis=-1)
    depth_in_pixel = depth_render @ RENDER_TO_PIXEL_ROT.T
    return depth_in_pixel[..., 2]


def depth_to_normals2(depth, pixmm=None, convert_render_z_to_pixel=None):
    """按配置从渲染深度生成法线。"""
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)

    if convert_render_z_to_pixel is None:
        convert_render_z_to_pixel = CONVERT_RENDER_Z_TO_PIXEL_FOR_GRADIENT

    if convert_render_z_to_pixel:
        depth_for_gradient = _convert_render_z_to_pixel_z(depth)
    else:
        depth_for_gradient = depth

    dzdx = (depth_for_gradient[1:-1, 2:] - depth_for_gradient[1:-1, :-2]) / 2.0
    dzdy = (depth_for_gradient[2:, 1:-1] - depth_for_gradient[:-2, 1:-1]) / 2.0

    dzdx_full = np.zeros((h, w), dtype=np.float32)
    dzdy_full = np.zeros((h, w), dtype=np.float32)
    dzdx_full[1:-1, 1:-1] = dzdx
    dzdy_full[1:-1, 1:-1] = dzdy

    normals[:, :, 0] = -dzdx_full
    normals[:, :, 1] = -dzdy_full
    normals[:, :, 2] = 1.0

    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= np.clip(norms, 1e-6, None)
    normals[..., 1] *= -1
    return normals


def normal_to_vis_u8(normal_map, pixmm=None):
    """仅用于显示：先将 nx, ny 除以 pixmm，再重新归一化并映射到 uint8。"""
    n_vis = np.asarray(normal_map, dtype=np.float32).copy()

    if pixmm is None:
        pixmm = NORMAL_VIS_PIXMM
    if pixmm is not None and float(pixmm) > 1e-8:
        n_vis[..., 0] /= float(pixmm)
        n_vis[..., 1] /= float(pixmm)

    norms = np.linalg.norm(n_vis, axis=2, keepdims=True)
    n_vis /= np.clip(norms, 1e-6, None)

    vis = (n_vis[..., [2, 1, 0]] * 0.5 + 0.5)
    vis = np.clip(vis, 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def parse_rect_points(points_str):
    """Parse 4 corner points like "x1,y1;x2,y2;x3,y3;x4,y4"."""
    parts = [p.strip() for p in points_str.split(';') if p.strip()]
    if len(parts) != 4:
        raise ValueError("需要 4 个角点坐标: x1,y1;x2,y2;x3,y3;x4,y4")
    pts = []
    for p in parts:
        xy = [s.strip() for s in p.split(',')]
        if len(xy) != 2:
            raise ValueError(f"角点格式错误: {p}")
        pts.append((int(float(xy[0])), int(float(xy[1]))))
    return pts


def rect_from_points(points):
    """Compute axis-aligned rect (x,y,w,h) from 4 points (exclusive width/height)."""
    pts = np.array(points, dtype=np.int32)
    min_x = int(pts[:, 0].min())
    min_y = int(pts[:, 1].min())
    max_x = int(pts[:, 0].max())
    max_y = int(pts[:, 1].max())
    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)
    return min_x, min_y, w, h


def clamp_rect(x, y, w, h, img_w, img_h):
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def apply_rect_crop(arr, rect):
    """Crop a 2D/3D array with rect (x,y,w,h)."""
    if arr is None or rect is None:
        return arr
    x, y, w, h = rect
    img_h, img_w = arr.shape[:2]
    x, y, w, h = clamp_rect(x, y, w, h, img_w, img_h)
    return arr[y:y + h, x:x + w]


def build_renderer():
    """构建渲染器和场景（不放置小球），返回 renderer, scene。"""
    mesh = trimesh.load_mesh(MESH_PATH)
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    cam_matrix = get_camera_matrix(camera_pos, pitch, yaw, roll)
    camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.5, zfar=1000.0)
    cam_node = pyrender.Node(camera=camera, matrix=cam_matrix)
    scene.add_node(cam_node)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add_node(pyrender.Node(light=light), parent_node=cam_node)

    # 用真实图像尺寸初始化 renderer
    first_img = cv2.imread(os.path.join(IMAGE_DIR, '1_rgb.png'))
    if first_img is None:
        raise FileNotFoundError('无法读取渲染参考尺寸图像: 1_rgb.png')
    h, w = first_img.shape[:2]
    renderer = pyrender.OffscreenRenderer(w, h)
    return renderer, scene


def resize_for_display_height(image, target_height=VIS_WINDOW_HEIGHT):
    """按固定高度缩放显示图像，宽度按比例自适应。"""
    if image is None:
        return image
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    if target_height <= 0:
        return image
    scale = float(target_height) / float(h)
    target_width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, int(target_height)), interpolation=interpolation)


def imshow_fixed_height(window_name, image, target_height=VIS_WINDOW_HEIGHT):
    """以固定高度显示窗口，保证不同输入尺寸下都便于观察。"""
    display = resize_for_display_height(image, target_height=target_height)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display)


def create_sphere_node():
    """创建默认小球节点，返回 sphere_node 和初始平移。"""
    sphere = trimesh.primitives.Sphere(radius=2.5, subdivisions=3)  # 球形压头
    # sphere = trimesh.primitives.Sphere(radius=3.5, subdivisions=3)  # 圆弧压头
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.5,
        baseColorFactor=[0.2, 0.2, 0.8, 1.0]
    )
    sphere_translation = np.array([17.6, 13.0, 29.5], dtype=np.float32)
    mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
    sphere_node = pyrender.Node(mesh=mesh, translation=sphere_translation.copy())
    return sphere_node, sphere_translation


def interactive_calibrate_one(renderer, base_scene, img_path, initial_translation=None):
    """
    类似于 2_normal_generate 中的交互方式，对单张图片进行小球位置标定。
    - renderer: OffscreenRenderer
    - base_scene: 不包含小球的场景
    - img_path: 实际图像路径
    - initial_translation: 初始小球平移 (3,)
    返回: 渲染坐标 sphere_translation (np.ndarray shape=(3,)) 或 None(用户取消)
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法读取实际图像 {img_path}, 跳过此样本")
        return None

    # 在场景中添加一个可交互控制的小球
    scene = base_scene
    sphere_node, sphere_translation_ori = create_sphere_node()
    if initial_translation is not None:
        sphere_translation = np.array(initial_translation, dtype=np.float32)
    else:
        sphere_translation = sphere_translation_ori.copy()
    sphere_node.translation = sphere_translation.copy()
    scene.add_node(sphere_node)

    step = MANUAL_STEP_FINE
    print("--- 交互标定 ---")
    print("W/S: X- / X+ | A/D: Y+ / Y- | Q/E: Z+ / Z-")
    print("J: 切换步长 (粗/细), P: 打印坐标, R: 记录并退出, Space: 跳过, ESC: 退出")

    while True:
        color, depth = renderer.render(scene)
        # 生成法线和遮罩，用于辅助观察
        normal_map = depth_to_normals2(depth)
        h_img, w_img = img.shape[:2]

        normal_vis_u8 = normal_to_vis_u8(normal_map)
        normal_vis_resized = cv2.resize(normal_vis_u8, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

        # 重叠真实图像与渲染图像
        color_resized = cv2.resize(color.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        img_u8 = img.astype(np.uint8)
        overlap_u8 = cv2.addWeighted(color_resized, 0.5, img_u8, 0.5, 0)

        # 放大2倍显示
        # overlap_u8_large = cv2.resize(overlap_u8, (w_img * 2, h_img * 2), interpolation=cv2.INTER_LINEAR)
        imshow_fixed_height("Normal Map (calib)", normal_vis_resized)
        imshow_fixed_height("Overlap (calib)", overlap_u8)
        imshow_fixed_height("Original Image", img_u8)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):  # y+
            sphere_translation[1] += step
        elif key == ord('d'):  # y-
            sphere_translation[1] -= step
        elif key == ord('s'):  # x+
            sphere_translation[0] += step
        elif key == ord('w'):  # x-
            sphere_translation[0] -= step
        elif key == ord('q'):  # z+
            sphere_translation[2] += step
        elif key == ord('e'):  # z-
            sphere_translation[2] -= step
        elif key == ord('j'):
            # 切换步长
            if abs(step - MANUAL_STEP_COARSE) < 1e-6:
                step = MANUAL_STEP_FINE
                print(f"切换到细步长: {step} mm")
            else:
                step = MANUAL_STEP_COARSE
                print(f"切换到粗步长: {step} mm")
        elif key == ord('o'):
            sphere_translation = sphere_translation_ori.copy()
            print("Sphere position reset to origin.")
        elif key == ord('p'):
            print(f"Sphere Position: x={sphere_translation[0]:.3f}, y={sphere_translation[1]:.3f}, z={sphere_translation[2]:.3f}")
        elif key == ord('r'):
            print("记录当前小球渲染坐标并退出此样本标定。")
            break
        elif key == 32:
            print("取消此样本标定。")
            sphere_translation = None
            break
        elif key == 27:  # ESC
            print("按下ESC，取消此样本标定。")
            sphere_translation = None
            break
        elif key == ord('b') or key == ord('B'):
            print("返回上一张图像。")
            sphere_translation = 'BACK'
            break

        sphere_node.translation = sphere_translation.copy()

    # 从场景移除小球，避免污染后续渲染
    scene.remove_node(sphere_node)
    cv2.destroyWindow("Normal Map (calib)")
    cv2.destroyWindow("Overlap (calib)")
    cv2.destroyWindow("Original Image")
    return None if sphere_translation is None else sphere_translation.copy()


def prompt_manual_calibration(sample_ids, log_dict):
    """对随机挑选的样本，通过交互方式标定小球渲染坐标。"""
    records = []
    renderer, base_scene = build_renderer()

    for idx in sample_ids:
        img_path = os.path.join(IMAGE_DIR, f'{idx}_rgb.png')
        if not os.path.exists(img_path):
            print(f'skip id {idx}, image missing')
            continue
        print(f"\n手动标定图像: {img_path}")
        render_pos = interactive_calibrate_one(renderer, base_scene, img_path)
        if render_pos is None:
            print(f"样本 {idx} 未完成标定，跳过。")
            continue
        records.append({
            'id': idx,
            'machine': log_dict[idx].tolist(),
            'render': render_pos.tolist()
        })

    with open(CALIB_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"已将 {len(records)} 条标定记录写入 {CALIB_FILE}")
    return records


def load_or_create_calibration(log_dict):
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r', encoding='utf-8') as f:
            recs = json.load(f)
        print("\n已从 manual_calib.json 读取标定数据，共 {} 条:".format(len(recs)))
        for rec in recs:
            mid = rec['id']
            mpos = rec['machine']
            rpos = rec['render']
            print(f"  id={mid}: machine = [{mpos[0]:.3f}, {mpos[1]:.3f}, {mpos[2]:.3f}], render = [{rpos[0]:.3f}, {rpos[1]:.3f}, {rpos[2]:.3f}]")
        return recs
    # 无现成标定数据，按空间分布挑选手动标定样本
    sample_ids = select_spread_samples(log_dict, SAMPLE_COUNT)
    recs = prompt_manual_calibration(sample_ids, log_dict)
    print("\n手动标定完成，共 {} 条: ".format(len(recs)))
    for rec in recs:
        mid = rec['id']
        mpos = rec['machine']
        rpos = rec['render']
        print(f"  id={mid}: machine = [{mpos[0]:.3f}, {mpos[1]:.3f}, {mpos[2]:.3f}], render = [{rpos[0]:.3f}, {rpos[1]:.3f}, {rpos[2]:.3f}]")
    return recs


def compute_rotation_matrix(machine_pts, render_pts):
    machine_center = np.mean(machine_pts, axis=0)
    render_center = np.mean(render_pts, axis=0)
    H = (machine_pts - machine_center).T @ (render_pts - render_center)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = render_center - R @ machine_center

    # 打印变换矩阵信息
    print("\n=== 机床坐标 -> 渲染坐标 拟合结果 ===")
    print("R (3x3 旋转矩阵):")
    for row in R:
        print("  [{: .6f} {: .6f} {: .6f}]".format(row[0], row[1], row[2]))
    print("t (平移向量): [{:.6f}, {:.6f}, {:.6f}]".format(t[0], t[1], t[2]))
    print("det(R) = {:.6f}".format(np.linalg.det(R)))

    return R, t


def visualize_projection(img_path, color_render):
    """将渲染结果与真实图像重叠显示，用于自动阶段抽查。"""
    img = cv2.imread(img_path)
    if img is None:
        return
    h_img, w_img = img.shape[:2]
    try:
        color_resized = cv2.resize(color_render.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    except Exception:
        color_resized = cv2.resize((color_render * 255.0).astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    img_u8 = img.astype(np.uint8)
    overlap_u8 = cv2.addWeighted(color_resized, 0.5, img_u8, 0.5, 0)
    # cv2.imshow("render", color_resized)
    imshow_fixed_height("AutoCheck Overlap", overlap_u8)
    cv2.waitKey(VISUALIZE_DELAY_MS)


def ask_manual_rotation_and_ref(log_dict):
    """手动旋转模式下，返回预设好的旋转矩阵 R 以及一个参考机床/渲染坐标对。

    R 来自代码顶部的 MANUAL_ROTATION_MATRIX，不再每次运行时从终端输入；
    参考点优先从 manual_calib.json 中选择一个已标定样本（用户只需选 id），
    若 manual_calib.json 不存在，则退化为从 log_dict 选 id 并手动输入该 id 的渲染坐标。
    """
    # 直接使用预先写好的旋转矩阵
    R_manual = MANUAL_ROTATION_MATRIX.copy()
    print("\n=== 手动旋转模式：使用预设旋转矩阵 R ===")
    print("R =")
    for r in R_manual:
        print("  [{: .6f} {: .6f} {: .6f}]".format(r[0], r[1], r[2]))
    print("det(R) = {:.6f}".format(np.linalg.det(R_manual)))

    # 从 manual_calib.json 中读取可用的标定点，供选择参考 id
    calib_recs = None
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, 'r', encoding='utf-8') as f:
                calib_recs = json.load(f)
        except Exception as e:
            print(f"读取 {CALIB_FILE} 失败: {e}")
            calib_recs = None

    ref_machine = None
    ref_render = None
    ref_id = None

    if calib_recs:
        print("\n在 manual_calib.json 中发现以下已标定样本，可作为参考点:")
        for rec in calib_recs:
            mid = rec.get('id')
            mpos = rec.get('machine')
            rpos = rec.get('render')
            print(f"  id={mid}: machine = [{mpos[0]:.3f}, {mpos[1]:.3f}, {mpos[2]:.3f}], "
                  f"render = [{rpos[0]:.3f}, {rpos[1]:.3f}, {rpos[2]:.3f}]")

        calib_ids = {rec['id'] for rec in calib_recs}
        print("\n请选择一个参考样本 id（必须在上述已标定 id 中）：")
        while True:
            try:
                ref_id = int(input("参考样本 id > "))
            except ValueError:
                print("请输入整数 id。")
                continue
            if ref_id not in calib_ids:
                print("该 id 不在 manual_calib.json 中，请重新选择。")
                continue
            break

        # 在 calib_recs 中找到对应记录
        for rec in calib_recs:
            if rec['id'] == ref_id:
                ref_machine = np.array(rec['machine'], dtype=np.float64)
                ref_render = np.array(rec['render'], dtype=np.float64)
                break

        print(f"参考样本 id={ref_id}：")
        print(f"  ref_machine = [{ref_machine[0]:.3f}, {ref_machine[1]:.3f}, {ref_machine[2]:.3f}]")
        print(f"  ref_render  = [{ref_render[0]:.3f}, {ref_render[1]:.3f}, {ref_render[2]:.3f}]")
    else:
        # 没有 manual_calib.json 的退化情况：只能让用户输入一个 ref_id，并从 log_dict 取机床坐标，
        # 渲染坐标仍然需要人工输入
        print("\n未找到 manual_calib.json，无法自动提供参考渲染坐标，将退化为手动输入模式。")
        example_ids = list(log_dict.keys())[:10]
        print("可用 id 示例:", example_ids)

        while True:
            try:
                ref_id = int(input("参考样本 id > "))
            except ValueError:
                print("请输入整数 id。")
                continue
            if ref_id not in log_dict:
                print("该 id 不在力日志中，请重新选择。")
                continue
            break

        ref_machine = log_dict[ref_id].astype(np.float64)
        print(f"参考样本 id={ref_id} 的机床坐标: "
              f"[{ref_machine[0]:.3f}, {ref_machine[1]:.3f}, {ref_machine[2]:.3f}]")

        print("请输出该样本对应的渲染器中小球坐标 (x y z)：")
        while True:
            line = input("ref_render (x y z) > ")
            vals = [v for v in line.replace(',', ' ').split() if v]
            if len(vals) != 3:
                print("必须输入 3 个数，请重新输入。")
                continue
            try:
                ref_render = np.array([float(v) for v in vals], dtype=np.float64)
            except ValueError:
                print("包含非数字字符，请重新输入。")
                continue
            break

        print("\n手动输入模式配置：")
        print(f"  参考 id = {ref_id}")
        print(f"  ref_machine = [{ref_machine[0]:.3f}, {ref_machine[1]:.3f}, {ref_machine[2]:.3f}]")
        print(f"  ref_render  = [{ref_render[0]:.3f}, {ref_render[1]:.3f}, {ref_render[2]:.3f}]")

    return R_manual, ref_machine, ref_render


def fine_tune_sphere_position(renderer, scene, sphere_node, img_path, init_pos, allow_skip=False):
    """在自动计算出的小球位置基础上做一次手动微调。
    返回: (final_pos, is_skipped)
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法读取实际图像 {img_path}, 跳过微调")
        # 如果读图失败，视为未跳过但使用初始值，或者你也可以根据逻辑改为返回 (None, True)
        return init_pos, False

    # 当前微调起点 = 自动计算出的 init_pos
    cur_pos = np.array(init_pos, dtype=np.float32)
    # 记录进入微调前的球位置，如果后面要恢复
    orig_pos = np.array(init_pos, dtype=np.float32)
    sphere_node.translation = cur_pos.copy()

    step = MANUAL_STEP_FINE
    print("\n--- 当前样本手动微调 ---")
    print("W/S: X- / X+ | A/D: Y+ / Y- | Q/E: Z+ / Z-")
    print("J: 切换步长 (粗/细), P: 打印坐标, R: 确认并继续下一张, C: 放弃微调")
    if allow_skip:
        print("X: 跳过当前样本")

    h_img, w_img = img.shape[:2]

    # 增加一个标志位，判断是否用户主动跳过
    is_skipped = False

    while True:
        color, depth = renderer.render(scene)
        normal_map = depth_to_normals2(depth)

        normal_vis_u8 = normal_to_vis_u8(normal_map)
        normal_vis_resized = cv2.resize(normal_vis_u8, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

        color_resized = cv2.resize(color.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        img_u8 = img.astype(np.uint8)
        overlap_u8 = cv2.addWeighted(color_resized, 0.5, img_u8, 0.5, 0)
        imshow_fixed_height("Normal Map (fine)", normal_vis_resized)
        imshow_fixed_height("Overlap (fine)", overlap_u8)
        imshow_fixed_height("Original Image", img_u8)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):  # y+
            cur_pos[1] += step
        elif key == ord('d'):  # y-
            cur_pos[1] -= step
        elif key == ord('s'):  # x+
            cur_pos[0] += step
        elif key == ord('w'):  # x-
            cur_pos[0] -= step
        elif key == ord('q'):  # z+
            cur_pos[2] += step
        elif key == ord('e'):  # z-
            cur_pos[2] -= step
        elif key == ord('j'):
            if abs(step - MANUAL_STEP_COARSE) < 1e-6:
                step = MANUAL_STEP_FINE
                print(f"切换到细步长: {step} mm")
            else:
                step = MANUAL_STEP_COARSE
                print(f"切换到粗步长: {step} mm")
        elif key == ord('p'):
            print(f"当前 Sphere Position: x={cur_pos[0]:.3f}, y={cur_pos[1]:.3f}, z={cur_pos[2]:.3f}")
        elif key == ord('r'):
            print("确认当前微调结果，进入下一张。")
            break
        elif key == ord('c') or key == 27:
            print("放弃微调，使用自动计算位置。")
            cur_pos = orig_pos.copy()
            sphere_node.translation = cur_pos.copy()
            break
        elif allow_skip and key == ord('x'):
            print("跳过当前样本。")
            cur_pos = None
            is_skipped = True  # 标记为跳过
            break

        sphere_node.translation = cur_pos.copy()

    cv2.destroyWindow("Normal Map (fine)")
    cv2.destroyWindow("Overlap (fine)")
    cv2.destroyWindow("Original Image")

    # --- 修复部分 ---
    if is_skipped:
        return None, True
    else:
        return cur_pos.astype(np.float64), False


def main():
    log_dict = load_force_log(CSV_PATH)

    renderer, scene = build_renderer()

    if not AUTO_ESTIMATE_SPHERE_POS:
        # 手动模式也渲染一次背景，用于构造 mask
        color_bg_manual, depth_bg_manual = renderer.render(scene)

        # 新增：手动模式也保存背景 normal / normal_complete / depth / mask（索引0），与自动模式一致
        normal_bg_manual = depth_to_normals2(depth_bg_manual)
        np.save(os.path.join(NORMAL_COMPLETE_DIR, 'normal_complete0.npy'), normal_bg_manual)
        np.save(os.path.join(NORMAL_DIR, 'normal0.npy'), normal_bg_manual)
        np.save(os.path.join(DEPTH_DIR, 'depth0.npy'), depth_bg_manual)
        np.save(os.path.join(MASK_DIR, 'mask0.npy'), np.ones(depth_bg_manual.shape, dtype=np.uint8))
        print(f"[手动模式] 已保存背景数据: {os.path.join(NORMAL_COMPLETE_DIR, 'normal_complete0.npy')}, {os.path.join(NORMAL_DIR, 'normal0.npy')}, {os.path.join(DEPTH_DIR, 'depth0.npy')}, {os.path.join(MASK_DIR, 'mask0.npy')}")

        def run_manual_per_image(renderer_local, base_scene, log_dict_local, depth_bg_local):
            """纯手动模式：给定初始小球位置，每一张选中的图像都手动调整一次小球位置。

            修改：
            1）小球初始位置使用上一张图像标定结果，实现“链式”连续微调；
            2）可选择仅标定 force_log 中 phase=vertical 的样本，或标定全部样本；
            3）对所有符合条件的图像都进行标定，不再只选取 SAMPLE_COUNT 张。
            4）按下 b 键可回到上一张图像。
            """
            # 选择标定范围：vertical 或 all（默认 vertical）
            print("\n[手动模式] 请选择标定样本范围：")
            print("  1: 仅 phase=vertical 样本")
            print("  2: 全部样本")
            scope_choice = input("请输入 1 或 2（默认 1）> ").strip()

            if scope_choice == '2':
                candidate_ids = sorted(log_dict_local.keys())
                scope_desc = "全部样本"
            else:
                candidate_ids = get_vertical_ids(CSV_PATH)
                scope_desc = "phase=vertical 样本"

            # 检查是否有对应图像
            valid_ids = []
            for idx in candidate_ids:
                img_path = os.path.join(IMAGE_DIR, f"{idx}_rgb.png")
                if os.path.exists(img_path):
                    valid_ids.append(idx)

            if not valid_ids:
                print(f"[手动模式] 没有找到符合条件的样本（{scope_desc} 且 images 中有对应文件）。")
                return

            valid_ids = sorted(list(set(valid_ids).intersection(log_dict_local.keys())))
            sample_ids = valid_ids  # 不再采样，全部标定
            print(f"\n[手动模式] 当前范围：{scope_desc}")
            print(f"[手动模式] 将对所有符合条件的 {len(sample_ids)} 个样本进行小球位置标定: {sample_ids}")

            _, sphere_init = create_sphere_node()
            scene_local = base_scene

            last_pos = None  # 记录上一张图像的小球位置

            # 改为索引循环，支持返回上一张
            i = 0
            while i < len(sample_ids):
                idx = sample_ids[i]
                img_path = os.path.join(IMAGE_DIR, f"{idx}_rgb.png")
                if not os.path.exists(img_path):
                    print(f"样本 {idx} 图像不存在，跳过。")
                    i += 1
                    continue
                print(f"\n[手动模式] 标定样本 id={idx} 对应图像: {img_path}")

                init_translation = sphere_init if last_pos is None else last_pos

                # pos = interactive_calibrate_one(
                #     renderer_local,
                #     scene_local,
                #     img_path,
                #     initial_translation=init_translation
                # )

                # 内联定义带 Overlay 的交互标定函数，以便传入 idx 和 log_dict
                def interactive_calibrate_with_overlay(renderer, base_scene, img_path, initial_translation, cur_idx, cur_log, total_count):
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"警告: 无法读取实际图像 {img_path}, 跳过此样本")
                        return None

                    # 在场景中添加一个可交互控制的小球
                    scene = base_scene
                    sphere_node, sphere_translation_ori = create_sphere_node()
                    if initial_translation is not None:
                        sphere_translation = np.array(initial_translation, dtype=np.float32)
                    else:
                        sphere_translation = sphere_translation_ori.copy()
                    sphere_node.translation = sphere_translation.copy()
                    scene.add_node(sphere_node)

                    step = MANUAL_STEP_FINE
                    print("--- 交互标定 ---")
                    print("W/S: X- / X+ | A/D: Y+ / Y- | Q/E: Z+ / Z-")
                    print("J: 切换步长 (粗/细), P: 打印坐标, R: 记录并退出, Space: 跳过")
                    print("B: 返回上一张, ESC: 退出整个标定")

                    h_img, w_img = img.shape[:2]

                    # 读取 Force X, Y, Z 和其他字段
                    fx_val = None
                    fy_val = None
                    fz_val = None
                    phase_val = 'N/A'
                    angle_deg_val = 'N/A'
                    target_val = 'N/A'
                    
                    if cur_idx in cur_log:
                        try:
                            fx_val = float(cur_log[cur_idx]['fx'])
                            fy_val = float(cur_log[cur_idx]['fy'])
                            fz_val = float(cur_log[cur_idx]['fz'])
                            phase_val = cur_log[cur_idx].get('phase', 'N/A')
                            angle_deg_val = cur_log[cur_idx].get('angle_deg', 'N/A')
                            target_val = cur_log[cur_idx].get('target', 'N/A')
                        except Exception:
                            fx_val = None
                            fy_val = None
                            fz_val = None

                    info_text_id = f"ID: {cur_idx} / {total_count}"
                    if fx_val is not None:
                        info_text_fx = f"Fx: {fx_val:.3f}"
                    else:
                        info_text_fx = "Fx: N/A"

                    if fy_val is not None:
                        info_text_fy = f"Fy: {fy_val:.3f}"
                    else:
                        info_text_fy = "Fy: N/A"

                    if fz_val is not None:
                        info_text_fz = f"Fz: {fz_val:.3f}"
                    else:
                        info_text_fz = "Fz: N/A"

                    if isinstance(angle_deg_val, float):
                        info_text_angle = f"angle_deg: {angle_deg_val:.1f}"
                    else:
                        info_text_angle = f"angle_deg: {angle_deg_val}"

                    info_text_phase = f"phase: {phase_val}"
                    info_text_target = f"target: {target_val}"

                    print(
                        "\n[当前样本信息]\n"
                        f"  {info_text_id}\n"
                        f"  {info_text_fx}\n"
                        f"  {info_text_fy}\n"
                        f"  {info_text_fz}\n"
                        f"  {info_text_phase}\n"
                        f"  {info_text_angle}\n"
                        f"  {info_text_target}"
                    )

                    while True:
                        color, depth = renderer.render(scene)
                        # 生成法线和遮罩，用于辅助观察
                        normal_map = depth_to_normals2(depth)

                        normal_vis_u8 = normal_to_vis_u8(normal_map)
                        normal_vis_resized = cv2.resize(normal_vis_u8, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

                        # 重叠真实图像与渲染图像
                        try:
                            color_resized = cv2.resize(color.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                        except:
                            color_resized = cv2.resize((color * 255.0).astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_LINEAR)

                        img_u8 = img.astype(np.uint8)
                        overlap_u8 = cv2.addWeighted(color_resized, 0.3, img_u8, 0.7, 0)

                        # 放大2倍显示 (可选)
                        # overlap_u8_large = cv2.resize(overlap_u8, (w_img * 2, h_img * 2), interpolation=cv2.INTER_LINEAR)
                        imshow_fixed_height("Normal Map (calib)", normal_vis_resized)
                        imshow_fixed_height("Overlap (calib)", overlap_u8)
                        imshow_fixed_height("Original Image", img_u8)

                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('a'):  # y+
                            sphere_translation[1] += step
                        elif key == ord('d'):  # y-
                            sphere_translation[1] -= step
                        elif key == ord('s'):  # x+
                            sphere_translation[0] += step
                        elif key == ord('w'):  # x-
                            sphere_translation[0] -= step
                        elif key == ord('q'):  # z+
                            sphere_translation[2] += step
                        elif key == ord('e'):  # z-
                            sphere_translation[2] -= step
                        elif key == ord('j'):
                            # 切换步长
                            if abs(step - MANUAL_STEP_COARSE) < 1e-6:
                                step = MANUAL_STEP_FINE
                                print(f"切换到细步长: {step} mm")
                            else:
                                step = MANUAL_STEP_COARSE
                                print(f"切换到粗步长: {step} mm")
                        elif key == ord('o'):
                            sphere_translation = sphere_translation_ori.copy()
                            print("Sphere position reset to origin.")
                        elif key == ord('p'):
                            print(f"Sphere Position: x={sphere_translation[0]:.3f}, y={sphere_translation[1]:.3f}, z={sphere_translation[2]:.3f}")
                        elif key == ord('r'):
                            print("记录当前小球渲染坐标并退出此样本标定。")
                            break
                        elif key == 32:
                            print("取消此样本标定。")
                            sphere_translation = None
                            break
                        elif key == 27:  # ESC
                            print("按下ESC，退出整个标定流程。")
                            sphere_translation = 'EXIT'
                            break
                        elif key == ord('b') or key == ord('B'):
                            print("返回上一张图像。")
                            sphere_translation = 'BACK'
                            break

                        sphere_node.translation = sphere_translation.copy() if isinstance(sphere_translation, (list, np.ndarray)) else sphere_node.translation

                    # 从场景移除小球，避免污染后续渲染
                    scene.remove_node(sphere_node)
                    # cv2.destroyWindow("Normal Map (calib)")
                    # cv2.destroyWindow("Overlap (calib)")
                    return None if sphere_translation is None else sphere_translation

                pos = interactive_calibrate_with_overlay(
                    renderer_local,
                    scene_local,
                    img_path,
                    initial_translation=init_translation,
                    cur_idx=idx,
                    cur_log=log_dict_local,
                    total_count=len(sample_ids)
                )

                # 处理返回上一张
                if isinstance(pos, str) and pos == 'BACK':
                    i = max(0, i - 1)
                    # 回退时不更新 last_pos
                    print(f"回到上一张，当前索引 -> {i}")
                    continue

                # 处理ESC退出
                if isinstance(pos, str) and pos == 'EXIT':
                    print("用户按下ESC，退出整个标定流程。")
                    break

                if pos is None:
                    print(f"样本 {idx} 未完成标定，跳过。")
                    i += 1
                    continue

                # 记录本次结果，供下一张图像使用
                last_pos = np.array(pos, dtype=np.float32).copy()

                print(f"样本 {idx} 最终小球位置: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

                temp_sphere_node, _ = create_sphere_node()
                temp_sphere_node.translation = np.array(pos, dtype=np.float32)
                scene_local.add_node(temp_sphere_node)

                _, depth = renderer_local.render(scene_local)

                scene_local.remove_node(temp_sphere_node)

                normal_map = depth_to_normals2(depth)
                mask = ((np.abs(depth - depth_bg_local) > 0.1) * 255).astype(np.uint8)

                np.save(os.path.join(NORMAL_COMPLETE_DIR, f'normal_complete{idx}.npy'), normal_map)
                masked_normal = normal_map.copy()
                masked_normal[mask == 0] = 0.0
                np.save(os.path.join(NORMAL_DIR, f'normal{idx}.npy'), masked_normal)
                np.save(os.path.join(DEPTH_DIR, f'depth{idx}.npy'), depth)
                np.save(os.path.join(MASK_DIR, f'mask{idx}.npy'), mask)
                print(f"[手动模式] 已保存 normal_complete{idx}.npy, normal{idx}.npy, depth{idx}.npy, mask{idx}.npy")

                # 前进到下一张
                i += 1

            print("\n[手动模式] 选定样本的小球位置标定及 normal/depth/mask 保存完成。")

        # 调用带背景深度的手动标定函数
        run_manual_per_image(renderer, scene, log_dict, depth_bg_manual)
        cv2.destroyAllWindows()
        return

    # ================= 自动模式：沿用原有逻辑 =================
    print("\n[自动模式] 将自动估计小球位置，并在整个数据集上生成 normal/depth/mask。")

    # 询问用户是否使用手动输入旋转矩阵
    print("\n请选择旋转矩阵来源：")
    print("  1: 手动输入旋转矩阵 R，并指定一个参考机床/渲染坐标对")
    print("  2: 使用手动标定点，通过 SVD 自动拟合 R 和 t")
    choice = input("请输入 1 或 2 > ").strip()

    if choice == '1':
        use_manual_rot = True
    else:
        use_manual_rot = False

    if use_manual_rot:
        R, ref_machine, ref_render = ask_manual_rotation_and_ref(log_dict)
        if ref_machine is None or ref_render is None:
            raise RuntimeError("手动旋转模式下未能获得有效的参考点，请检查 manual_calib.json 或输入。")
        t = None
        calibration = None
    else:
        calibration = load_or_create_calibration(log_dict)
        if len(calibration) < 3:
            raise RuntimeError('标定点不足 3 个，无法拟合旋转矩阵')
        machine_pts = np.array([rec['machine'] for rec in calibration])
        render_pts = np.array([rec['render'] for rec in calibration])
        R, t = compute_rotation_matrix(machine_pts, render_pts)
        reference = calibration[0]
        ref_machine = np.array(reference['machine'])
        ref_render = np.array(reference['render'])

    # 背景深度/法线，参考 2_normal_generate 的保存方式
    color_bg, depth_bg = renderer.render(scene)
    normal_bg = depth_to_normals2(depth_bg)
    np.save(os.path.join(NORMAL_COMPLETE_DIR, 'normal_complete0.npy'), normal_bg)
    np.save(os.path.join(NORMAL_DIR, 'normal0.npy'), normal_bg)
    np.save(os.path.join(DEPTH_DIR, 'depth0.npy'), depth_bg)
    np.save(os.path.join(MASK_DIR, 'mask0.npy'), np.ones(depth_bg.shape, dtype=np.uint8))
    print(f"[自动模式] 已保存背景数据: {os.path.join(NORMAL_COMPLETE_DIR, 'normal_complete0.npy')}, {os.path.join(NORMAL_DIR, 'normal0.npy')}, {os.path.join(DEPTH_DIR, 'depth0.npy')}, {os.path.join(MASK_DIR, 'mask0.npy')}")

    # 准备一个小球节点用于自动阶段
    sphere_node, _ = create_sphere_node()
    scene.add_node(sphere_node)

    print("\n=== 自动法向量生成阶段 ===")
    last_confirmed_pos = None

    for idx in sorted(log_dict.keys()):
        machine_pos = log_dict[idx]
        img_path = os.path.join(IMAGE_DIR, f'{idx}_rgb.png')
        if not os.path.exists(img_path):
            continue

        machine_pos = machine_pos.astype(np.float64)
        if use_manual_rot:
            delta_machine = machine_pos - ref_machine
            delta_render = R @ delta_machine
            base_pos = ref_render + delta_render
        else:
            base_pos = R @ machine_pos + t

        if USE_CHAIN_TUNING and last_confirmed_pos is not None:
            sphere_pos = last_confirmed_pos.copy()
        else:
            sphere_pos = base_pos

        if ENABLE_PER_SAMPLE_MANUAL_FINE_TUNE:
            sphere_pos, skipped = fine_tune_sphere_position(renderer, scene, sphere_node, img_path, sphere_pos, allow_skip=True)
            if skipped:
                print(f"样本 {idx} 被 'X' 跳过，不生成法向量。")
                continue

        sphere_node.translation = sphere_pos.astype(np.float32)
        last_confirmed_pos = sphere_pos.copy()

        color, depth = renderer.render(scene)
        # 使用 depth_to_normals2 生成完整法线
        normal_map = depth_to_normals2(depth)
        # 根据与背景深度差构造掩膜（与 2_normal_generate 类似）
        mask = ((np.abs(depth - depth_bg) > 0.1) * 255).astype(np.uint8)

        # 保存 normal_complete（不加掩膜）
        np.save(os.path.join(NORMAL_COMPLETE_DIR, f'normal_complete{idx}.npy'), normal_map)
        # 保存 normal（掩膜外清零）
        masked_normal = normal_map.copy()
        masked_normal[mask == 0] = 0.0
        np.save(os.path.join(NORMAL_DIR, f'normal{idx}.npy'), masked_normal)
        # 保存 depth（保留完整深度）
        np.save(os.path.join(DEPTH_DIR, f'depth{idx}.npy'), depth)
        # 保存 mask
        np.save(os.path.join(MASK_DIR, f'mask{idx}.npy'), mask)
        print(f'generated normals for id {idx}')

        if VISUALIZE_INTERVAL > 0 and (idx % VISUALIZE_INTERVAL == 0):
            visualize_projection(img_path, color)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
