import trimesh
import pyrender
import cv2
import numpy as np

# --- 1. 手动标记 2D 特征点 (与之前类似) ---

# 全局变量，用于存储手动标记的点
manual_points = []
target_img_for_drawing = None


def select_points_callback(event, x, y, flags, param):
    """鼠标回调函数，用于手动标记特征点"""
    global manual_points, target_img_for_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_points.append((x, y))
        cv2.circle(target_img_for_drawing, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(target_img_for_drawing, str(len(manual_points)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.imshow("Select Target Points - Press 'c' to continue", target_img_for_drawing)


def get_camera_transform_from_rvec_tvec(rvec, tvec):
    """将OpenCV的旋转/平移向量转换为4x4变换矩阵"""
    R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def get_camera_matrix(pos=(0, 0, 10), pitch=0, yaw=0, roll=0):
    """
    全自由度相机外参矩阵生成 (用于手动微调)
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
    T[:3, 3] = pos  # 平移分量

    # 添加OpenGL到CV的坐标系转换 (如果 auto 脚本里已经乘了 fix_matrix，这里就不需要了，或者保持一致)
    # 在 1_get_cam_param.py 里 cv_to_gl 也是被注释掉的，返回的就是 T
    return T


# --- 2. 主程序流程 ---
if __name__ == '__main__':
    # 是否开启自动标定 (True: 手动打点标定 -> 微调; False: 直接使用预设参数 -> 微调)
    ENABLE_AUTO_CALIBRATION = False

    # 加载目标图像
    # target_img = cv2.imread("processed_4.png")  # 多光谱三指1号
    # target_img = cv2.imread("20250715-092715/0.png")  # 自研五指1号
    # target_img = cv2.imread("20250726-153852/0.png")  # 拾玥食指
    # target_img = cv2.imread("20251118-171914/0_rgb.png")    # 内窥镜三指2号
    # target_img = cv2.imread("20251118-153419/0_rgb.png")    # 内窥镜三指 标记点1号
    # target_img = cv2.imread("20251206-161215/0_rgb.png")  # 曲面三指 标记点 1号
    # target_img = cv2.imread("data/20260106_210820/processed_93.png")  # 多光谱三指1号
    # target_img = cv2.imread("data/20260114_120750/processed_109.png")  # 多光谱三指3号
    # target_img = cv2.imread("data/20260129_114800/processed_109.png")  # 多光谱三指3号 1280×720
    # target_img = cv2.imread("data/20260421_094400/val/images_removed_cropped/20260421_calib.png")  # 多光谱三指3号 128*128
    # target_img = cv2.imread("20260316_rig.png")  
    # target_img = cv2.imread("20260328-204221/0_rgb.png")  # 内窥镜三指1号
    # target_img = cv2.imread("20260404-183844/0_rgb.png")  # 内窥镜三指2号
    # target_img = cv2.imread("data/20260417_112900/images_removed_cropped/cam_calib.png")  # 内窥镜三指2号 128*128
    target_img = cv2.imread("cam_calib/cam_calib.png")  # 曲面三指2号

    assert target_img is not None, "图像读取失败，请检查文件路径"
    h, w = target_img.shape[:2]
    target_img_for_drawing = target_img.copy()

    # 初始化参数变量（后续自动标定会覆盖，或手动模式作为默认值）
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

    # 多光谱三指3号 1280×720
    # camera_pos = [16.86800519, 15.90078114, 19.79845209]
    # pitch = 127.87
    # yaw = -1.17
    # roll = -89.27
    # fx = 489.4
    # fy = 426.3
    # cx = 650.8
    # cy = 513.8

    # 内窥镜三指1号相机参数 其他默认
    # # pyrender/OpenGL 默认相机朝向:相机默认朝向 -Z 轴,X 轴向右，Y 轴向上
    # camera_pos = [3.399999999999999, 15.99999999999996, 1.6000000000000012]  # x,y,z
    # pitch = -212.20 # 绕X轴的旋转角度,但在渲染器中是绕Y轴的旋转角度
    # yaw = 0.10  # 绕Y轴的旋转角度,但在渲染器中是绕X轴的旋转角度;相机初始方向朝z轴负方向,这里转180度使相机能看到标定物
    # roll = -90.20  # 绕Z轴的旋转角度
    # fx = 1070.0
    # fy = 930.0
    # cx = 320
    # cy = 230

    # 内窥镜三指2号相机参数
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



    cam_to_world_matrix = np.eye(4)

    if ENABLE_AUTO_CALIBRATION:
        # --- 手动标记2D特征点 ---
        window_name = "Select Target Points - Press 'c' to continue"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_points_callback)
        cv2.imshow(window_name, target_img_for_drawing)

        print("请在弹出的窗口中，用鼠标左键按既定顺序点击所有特征点。")
        print("标记完成后，按 'c' 键开始计算。")

        if cv2.waitKey(0) & 0xFF != ord('c'):
            print("操作取消。")
            exit()

        if len(manual_points) < 6:  # PnP至少需要4个点，但越多越稳定
            print(f"错误：需要至少6个特征点，当前标记了 {len(manual_points)} 个。")
            exit()

        # 按标记顺序直接使用，确保与3D点顺序一致
        image_points_2d = np.array(manual_points, dtype=np.float32)
        print(f"成功标记了 {len(image_points_2d)} 个特征点。")
        cv2.destroyWindow(window_name)

        # --- 定义3D世界坐标 ---
        # !!! 关键步骤: 您必须用从STL模型中测量的精确值替换这些坐标 !!!
        # 顺序必须与您手动点击2D点的顺序完全一致

        # # 平面三指标定物特征点的示例坐标
        # object_points_3d = np.array([
        #     # 示例坐标，请务必替换
        #     [16.06044, 22.01535, 31.2185], [16.06045, 17.01535, 31.2185], [16.06046, 12.01535, 31.2185],
        #     [18.71841, 22.01536, 26.9835], [18.71842, 17.01536, 26.9835], [18.71843, 12.01536, 26.9835],
        #     [21.37638, 22.01536, 22.7485], [21.37638, 17.01536, 22.7485], [21.37639, 12.01536, 22.7485],
        # ], dtype=np.float32)

        # # 自研五指的示例坐标
        # object_points_3d = np.array([
        #     # 示例坐标，请务必替换
        #     [15.4385, 4.26546, 32.88853], [12.40409, 4.38635, 34.64524], [9.38486, 4.26548, 32.89235],
        #     [9.40465, 4.02173, 29.3686], [12.40376, 3.87399, 27.62066], [15.4449, 4.02453, 29.36709],
        #     [12.40385, 4.11423, 31.13535], [18.57917, 15.47647, 9.40852]
        # ], dtype=np.float32)

        # # 多光谱三指1号的示例坐标
        # object_points_3d = np.array([
        #     # 示例坐标，请务必替换
        #     [22.23559, 19.57933, 22.59226],
        #     [21.85072,17.84749,25.53882],
        #     [21.44351,17.84763,27.45853],
        #     [21.30611,21.21343,27.40156],
        #     [21.18311,16.11566,28.40404],
        #     [21.44345,14.38353,27.45876],
        #     [21.30632,11.0176,27.40238],
        #     [22.23554,12.65112,22.59275],
        #     [22.32305,14.3831,21.59656],
        #     [22.32307,17.8472,21.59631],
        #     [22.00454,16.11537,24.56509]
        # ], dtype=np.float32)

        # 多光谱三指3号的示例坐标
        object_points_3d = np.array([
            # 示例坐标，请务必替换
            [22.23559, 19.57933, 22.59226],
            [21.85072,17.84749,25.53882],
            [21.44351,17.84763,27.45853],
            [21.30611,21.21343,27.40156],
            [21.18311,16.11566,28.40404],
            [21.44345,14.38353,27.45876],
            [21.30632,11.0176,27.40238],
            [22.23554,12.65112,22.59275],
            [22.32305,14.3831,21.59656],
            [22.32307,17.8472,21.59631],
            [22.00454,16.11537,24.56509]
        ], dtype=np.float32)

        # # 拾玥食指的示例坐标
        # object_points_3d = np.array([
        #     # 示例坐标，请务必替换
        #     [7.85787, 15.51974, 6.16439],
        #     [7.85647, 13.54145, 6.15877],
        #     [6.61154, 14.54145, 7.99571],
        #     [7.12844, 11.8094, 7.17162],
        #     [5.93814, 12.8094, 9.24351],
        #     [5.93814, 10.8094, 9.24351],
        #     [7.85647, 10.07735, 6.15877],
        #     [6.61154, 9.07735, 7.99571],
        #     [7.85787, 8.09906, 6.16439],
        #     [7.46076, 7.15787, 6.9323],
        #     [6.20441, 8.06389, 8.73548],
        #     [8.5692, 6.92313, 5.66903],
        #     [5.5806, 9.18285, 10.01731],
        #     [5.16575, 11.06223, 11.05188],
        #     [5.55354, 14.42828, 10.07997],
        #     [5.16575, 12.55657, 11.05188],
        #     [8.62364, 11.80913, 5.3321]
        # ], dtype=np.float32)

        # 确保3D点和2D点数量一致
        if len(object_points_3d) != len(image_points_2d):
            print(f"错误：3D点数量({len(object_points_3d)})与2D点数量({len(image_points_2d)})不匹配！")
            exit()

        print("\n开始使用PnP算法计算相机参数...")

        # --- 使用 cv2.calibrateCamera 计算内外参 ---
        # 将3D点和2D点包装在列表中
        objpoints = [object_points_3d]
        imgpoints = [image_points_2d]

        # --- 新增代码：为非平面标定创建初始内参矩阵 ---
        # 对于非平面点，需要提供一个内参的初始猜测值
        # 一个常见的做法是假设主点在图像中心，焦距约等于图像宽度
        initial_camera_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # 设置标定标志，告诉函数使用我们提供的初始内参
        calibration_flags = cv2.CALIB_USE_INTRINSIC_GUESS

        # (类型, 最大迭代次数, 精度)
        # 这里我们将最大迭代次数设置为 100 次
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)


        # 执行相机标定 (修改了最后两个参数)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), initial_camera_matrix, None,
            flags=calibration_flags,
            criteria=criteria
        )

        if not ret:
            print("相机标定失败！请检查输入的3D/2D点是否准确。")
            exit()

        # --- 提取并打印结果 ---
        fx, fy = mtx[0, 0], mtx[1, 1]
        cx, cy = mtx[0, 2], mtx[1, 2]
        rvec, tvec = rvecs[0], tvecs[0]  # 旋转和平移向量

        # 将旋转向量转换为欧拉角（度）以便查看
        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0
        pitch_deg, yaw_deg, roll_deg = np.degrees([pitch, yaw, roll])

        print("\n计算成功！")
        print("=" * 30)
        print("相机内参矩阵:")
        print(mtx)
        print(f"\nfx = {fx:.2f}, fy = {fy:.2f}")
        print(f"cx = {cx:.2f}, cy = {cy:.2f}")
        print("=" * 30)

        # OpenCV的(rvec, tvec)描述了世界坐标系到相机坐标系的变换
        # Pyrender的相机pose描述了相机在世界坐标系中的位置和姿态，是前者的逆
        cam_to_world_matrix = np.linalg.inv(get_camera_transform_from_rvec_tvec(rvec, tvec))

        # 这是因为OpenCV和Pyrender/OpenGL的相机坐标系定义不同
        # (Y轴和Z轴方向相反)。这个修正矩阵将OpenCV相机转换为OpenGL相机。
        fix_matrix = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ])
        cam_to_world_matrix = cam_to_world_matrix @ fix_matrix

        # 从变换矩阵中提取平移（位置）和旋转
        t_world = cam_to_world_matrix[:3, 3]
        R_world = cam_to_world_matrix[:3, :3]

        # 将旋转矩阵转换为欧拉角（度）以便查看
        sy_world = np.sqrt(R_world[0, 0] ** 2 + R_world[1, 0] ** 2)
        singular_world = sy_world < 1e-6
        if not singular_world:
            pitch_world = np.arctan2(R_world[2, 1], R_world[2, 2])
            yaw_world = np.arctan2(-R_world[2, 0], sy_world)
            roll_world = np.arctan2(R_world[1, 0], R_world[0, 0])
        else:
            pitch_world = np.arctan2(-R_world[1, 2], R_world[1, 1])
            yaw_world = np.arctan2(-R_world[2, 0], sy_world)
            roll_world = 0
        roll_world_deg, pitch_world_deg, yaw_world_deg = np.degrees([roll_world, pitch_world, yaw_world])

        # 更新全局变量，用于微调初始值
        camera_pos = t_world
        pitch = pitch_world_deg
        yaw = yaw_world_deg
        roll = roll_world_deg

        print("\n相机在世界坐标系中的位姿 (Camera Pose in World Frame):")
        print(f"位置 (x, y, z): {t_world}")
        print(f"姿态 (Roll, Pitch, Yaw) in degrees: [{roll_world_deg:.2f}, {pitch_world_deg:.2f}, {yaw_world_deg:.2f}]")
        print("=" * 30)

    else:
        # 手动模式，直接使用预设值构建 cam_to_world_matrix
        print(f"跳过自动标定，使用预设参数: pos={camera_pos}, pitch={pitch}, yaw={yaw}, roll={roll}")
        cam_to_world_matrix = get_camera_matrix(pos=camera_pos, pitch=pitch, yaw=yaw, roll=roll)

    # --- (可选) 使用pyrender验证结果 + 手动微调 ---
    print("\n正在使用计算出的参数进行渲染以供验证...")
    # mesh = trimesh.load_mesh('mesh/finger_3.STL')   # 平面三指
    # mesh = trimesh.load_mesh('mesh/finger_03_05.STL')  # 自研五指
    # mesh = trimesh.load_mesh('mesh/shiyue_finger.STL')  # 拾玥食指
    mesh = trimesh.load_mesh('mesh/sanzhi_curved.STL')  # 曲面三指
    # scene = pyrender.Scene(bg_color=[0.3, 0.3, 0.3, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene = pyrender.Scene()
    # material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.1, roughnessFactor=0.7)
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pyrender_mesh)

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

    camera_node = pyrender.Node(camera=camera, matrix=cam_to_world_matrix)
    scene.add_node(camera_node)

    # scene.add(camera, pose=cam_to_world_matrix) # 用 Node 方式添加更方便修改
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), parent_node=camera_node)

    renderer = pyrender.OffscreenRenderer(w, h)

    print("\n进入手动微调模式:")
    print("  [W/A/S/D/Z/X] - 调整相机位置 (x, y, z)")
    print("  [I/K/J/L/U/O] - 调整相机角度 (Pitch, Yaw, Roll)")
    print("  [4/1] - fx +/- 10")
    print("  [5/2] - fy +/- 10")
    print("  [6/3] - cx +/- 5")
    print("  [8/7] - cy +/- 5")
    print("  [p]   - 打印当前参数")
    print("  [q]   - 退出")

    cv2.namedWindow("Target Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Final Alignment Verification (Overlap)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Render Only", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while True:
        # 渲染
        color, _ = renderer.render(scene)

        # 显示
        final_overlap = cv2.addWeighted(color, 0.6, target_img, 0.4, 0)

        # 为了方便观察，可以拼图显示或者只显示重叠图
        # cv2.imshow("Rendered Image", cv2.resize(color, (w, h)))
        cv2.imshow("Target Image", cv2.resize(target_img, (w, h)))
        cv2.imshow("Final Alignment Verification (Overlap)", cv2.resize(final_overlap, (w, h)))
        cv2.imshow("Render Only", cv2.resize(color, (w, h)))

        key = cv2.waitKey(30) & 0xFF

        # Allow quit via 'q' or ESC during manual calibration
        if key == 27:
            print("User requested exit (ESC). Exiting manual adjustment.")
            break

        # 位置调整
        step_pos = 0.1
        if key == ord('a'): camera_pos[1] -= step_pos
        elif key == ord('d'): camera_pos[1] += step_pos
        elif key == ord('s'): camera_pos[0] -= step_pos
        elif key == ord('w'): camera_pos[0] += step_pos
        elif key == ord('z'): camera_pos[2] += step_pos
        elif key == ord('x'): camera_pos[2] -= step_pos

        # 角度调整
        step_angle = 0.5
        if key == ord('i'): pitch -= step_angle
        elif key == ord('k'): pitch += step_angle
        elif key == ord('j'): yaw += step_angle
        elif key == ord('l'): yaw -= step_angle
        elif key == ord('u'): roll -= step_angle
        elif key == ord('o'): roll += step_angle

        # 内参调整
        if key == ord('4'): fx += 10
        elif key == ord('1'): fx -= 10
        if key == ord('5'): fy += 10
        elif key == ord('2'): fy -= 10

        if key == ord('6'): cx += 5
        elif key == ord('3'): cx -= 5
        if key == ord('8'): cy += 5
        elif key == ord('7'): cy -= 5

        # 仅在有按键操作时打印（简单的检测按键是否被按下比较麻烦，这里暂时每次循环都打印或者只在调整后打印，
        # 为了简化，如果是调整键才打印）
        if key != 255: # 有按键按下
             if key == ord('p'):
                 print(f"pos:{camera_pos}, fx:{fx:.1f}, fy:{fy:.1f}, cx:{cx:.1f}, cy:{cy:.1f}, pitch:{pitch:.2f}, yaw:{yaw:.2f}, roll:{roll:.2f}")

             # 更新相机节点
             new_camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
             camera_node.camera = new_camera

             # 更新位姿矩阵
             new_matrix = get_camera_matrix(pos=camera_pos, pitch=pitch, yaw=yaw, roll=roll)
             camera_node.matrix = new_matrix

    cv2.destroyAllWindows()
