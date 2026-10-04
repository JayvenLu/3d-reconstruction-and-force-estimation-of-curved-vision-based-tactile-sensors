import argparse
import datetime

import trimesh
import pyrender
import cv2
import numpy as np
import open3d as o3d

DEFAULT_CONFIG = {
    "mesh_path": "mesh/sanzhi_bg.STL",
    "render_width": 128,
    "render_height": 128,
    "inner_width": 126,
    "inner_height": 126,
    "depth_output_path": "dep_inner126x126_20260421.npy",
    "prior_output_path": "dep_p_inner126x126_20260421.npy",
    "show_point_cloud": True,
}

# 定义相机矩阵
def get_camera_matrix(pos=(0, 0, 10), pitch=0.0, yaw=0.0, roll=0.0):
    """
    全自由度相机外参矩阵生成
    参数:
        pos: 相机位置 (x,y,z)
        pitch/yaw/roll: 绕X/Y/Z轴的旋转角度（度）
    """
    # 角度转弧度
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)

    # 构造旋转矩阵（ZYX顺序）
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                   [0, np.sin(pitch_rad), np.cos(pitch_rad)]])

    Ry = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                   [0, 1, 0],
                   [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])

    Rz = np.array([[np.cos(roll_rad), -np.sin(roll_rad), 0],
                   [np.sin(roll_rad), np.cos(roll_rad), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx  # 组合旋转

    # 构造变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos # 平移分量

    # 添加OpenGL到CV的坐标系转换
    cv_to_gl = np.diag([1, -1, -1, 1])  # 翻转Y/Z轴方向
    # return T @ cv_to_gl
    return T

# # 自研五指相机参数
# camera_pos = [11.999999999999973, 15.89999999999996, 11.899999999999974]  # x,y,z
# pitch = -35.90000000000024       # pitch,yaw,roll
# yaw = 175.00000000000028
# roll = 10.699999999999978
# fx = 1190
# fy = 1190

# # 拾玥食指1号相机参数
# camera_pos = [9.099999999999996, 11.499999999999998, 16.900000000000002]  # x,y,z
# pitch = -201.60000000000036  # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
# yaw = 182.19999999999987   # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
# roll = 90.49999999999997  # 绕Z轴的旋转角度
# fx = 540
# fy = 540

# # 三指 1号
# camera_pos = [12.999999999999968, 16.399999999999963, -3.8000000000000016]  # x,y,z
# pitch = -193.1999999999933 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
# yaw = 0.0999999999999992  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
# roll = -90.19999999999915  # 绕Z轴的旋转角度
# fx = 1260
# fy = 1260

# # 第二个传感器相机参数
# pos = [0, -3.5000000000000013, 9.000000000000004]
# pitch = -15.89999999999996
# yaw = 1.9000000000000006
# roll = 0
# fx=fy=680.8

# # 三指 内窥镜 2号
# camera_pos = [12.70319275, 16.58702443,  8.24603313]  # x,y,z
# pitch = 156.49 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
# yaw = -0.95  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
# roll = -89.19  # 绕Z轴的旋转角度
# fx = 1167.43
# fy = 1122.59
# cx = 655.72
# cy = 329.43

# # 三指 内窥镜 标记点 1号
# camera_pos = [13.94878765, 16.65199845, 10.36364977]  # x,y,z
# pitch = 152.14 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
# yaw = -0.92  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
# roll = -90.33  # 绕Z轴的旋转角度
# fx = 630.30
# fy = 618.78
# cx = 339.18
# cy = 281.96

# 曲面三指1号
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

# 多光谱三指3号
# camera_pos = [15.61670647999999, 15.88688785, 18.462206920000007]
# pitch = 135.55
# yaw = -0.94
# roll = -90.61
# fx = 606.1
# fy = 548.1
# cx = 945.5
# cy = 763.2

# 内窥镜三指2号相机参数 128*128
# camera_pos = [4.6999999999999975, 15.89999999999996, 1.6000000000000012]
# pitch = -210.20
# yaw = 2.60
# roll = -86.70
# fx = 293.33
# fy = 229.33
# cx = 64.00
# cy = 61.33

# 多光谱三指3号 128*128
camera_pos = [11.516706480000005, 16.286887850000003, 20.66220692000004]
pitch = 135.55
yaw = -0.94
roll = -90.11
fx = 81.8
fy = 62.5
cx = 60.0
cy = 61.3

# 多光谱三指 1920×1080 无标记点
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
# camera_pos = [4.6999999999999975, 15.89999999999996, 1.6000000000000012]  # x,y,z
# pitch = -210.20 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
# yaw = 2.60  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
# roll = -86.70 # 绕Z轴的旋转角度
# fx = 1100.0
# fy = 860.0
# cx = 320
# cy = 230

# 计算点云
def depth_to_point_cloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    points = []

    # 将每个像素的深度值转换为 3D 点
    for y in range(h):
        for x in range(w):
            Z = depth[y, x]
            if Z > 0:  # 只考虑深度大于0的有效点
                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                points.append([X, Y, Z])

    return np.array(points)


def create_depth_prior(depth, inner_width, inner_height):
    h, w = depth.shape
    if inner_width <= 0 or inner_height <= 0:
        raise ValueError(f"inner size must be positive, got {inner_width}x{inner_height}")
    if inner_width > w or inner_height > h:
        raise ValueError(
            f"inner size {inner_width}x{inner_height} exceeds rendered depth size {w}x{h}"
        )

    margin_x = w - inner_width
    margin_y = h - inner_height
    left = margin_x // 2
    right = margin_x - left
    top = margin_y // 2
    bottom = margin_y - top

    depth_prior = depth.copy()
    inner_y_end = h - bottom
    inner_x_end = w - right
    depth_prior[top:inner_y_end, left:inner_x_end] = np.nan
    return depth_prior, {"left": left, "right": right, "top": top, "bottom": bottom}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Render depth prior with a configurable inner reconstruction region."
    )
    parser.add_argument("--inner-width", type=int, default=DEFAULT_CONFIG["inner_width"])
    parser.add_argument("--inner-height", type=int, default=DEFAULT_CONFIG["inner_height"])
    parser.add_argument("--render-width", type=int, default=DEFAULT_CONFIG["render_width"])
    parser.add_argument("--render-height", type=int, default=DEFAULT_CONFIG["render_height"])
    parser.add_argument("--mesh-path", default=DEFAULT_CONFIG["mesh_path"])
    parser.add_argument("--depth-output-path", default=DEFAULT_CONFIG["depth_output_path"])
    parser.add_argument("--prior-output-path", default=DEFAULT_CONFIG["prior_output_path"])
    parser.add_argument(
        "--no-show-point-cloud",
        action="store_true",
        help="Disable Open3D point cloud visualization.",
    )
    return parser


def main(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    mesh = trimesh.load_mesh(cfg["mesh_path"])
    scene = pyrender.Scene()
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pyrender_mesh)
    renderer = pyrender.OffscreenRenderer(cfg["render_width"], cfg["render_height"])

    camera_matrix = get_camera_matrix(camera_pos, pitch, yaw, roll)
    camera = pyrender.camera.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=0.5,
        zfar=1000.0,
        name='rgb'
    )
    camera_node = pyrender.Node(camera=camera, matrix=camera_matrix)
    scene.add_node(camera_node)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    direct_light = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(direct_light, parent_node=camera_node)

    color, depth = renderer.render(scene)
    np.save(cfg["depth_output_path"], depth)

    depth_p, margins = create_depth_prior(depth, cfg["inner_width"], cfg["inner_height"])

    npy_filename = cfg["prior_output_path"]
    np.save(npy_filename, depth_p)
    print("深度先验已保存为:", npy_filename)
    print(
        "inner reconstruction size:",
        f'{cfg["inner_width"]}x{cfg["inner_height"]}',
        "margins:",
        margins,
    )

    if cfg["show_point_cloud"]:
        h, w = depth_p.shape
        vis_cx, vis_cy = w / 2, h / 2
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_p.flatten()
        x = (u.flatten() - vis_cx) * z / fx
        y = (v.flatten() - vis_cy) * z / fy
        valid = ~np.isnan(z)
        points = np.vstack((x[valid], y[valid], z[valid])).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(
        {
            "mesh_path": args.mesh_path,
            "render_width": args.render_width,
            "render_height": args.render_height,
            "inner_width": args.inner_width,
            "inner_height": args.inner_height,
            "depth_output_path": args.depth_output_path,
            "prior_output_path": args.prior_output_path,
            "show_point_cloud": not args.no_show_point_cloud,
        }
    )
