# 该脚本用于离线标定网格点位置，生成预设文件供后续使用

import cv2
import numpy as np
import os
import setting

# 加载配置
setting.init()
N = setting.N_  # 行数
M = setting.M_  # 列数

# 多光谱三指3号 小球形压头
# IMG_PATH = r"20260120-165420/0_rgb_perspective.png"
# PRESET_PATH = r"20260120-165420/grid_preset_rgb_perspective.npy"

# 多光谱三指3号 小球形压头
# IMG_PATH = r"data/20260114_120750/train/images/0_rgb_perspective.png"
# PRESET_PATH = r"data/20260114_120750/train/grid_preset_rgb_perspective.npy"

# 内窥镜三指1号
# IMG_PATH = r"data/20260328_121407/train/images/0_rgb_perspective.png"  
# PRESET_PATH = r"data/20260328_121407/train/grid_preset_rgb_perspective.npy"

# 内窥镜三指2号
# IMG_PATH = r"data/20260404_103156/train/images/0_rgb_perspective.png"  
# PRESET_PATH = r"data/20260404_103156/train/images/grid_preset_rgb_perspective.npy"

# 内窥镜三指2号 20260827
# IMG_PATH = r"cam/20260827-145539/0_rgb_perspective.png"  
# PRESET_PATH = r"cam/20260827-145539/grid_preset_rgb_perspective.npy"

# 华威科样品1号 20260827
IMG_PATH = r"cam/20260827-150753/0_rgb_perspective.png"  
PRESET_PATH = r"cam/20260827-150753/grid_preset_rgb_perspective.npy"

# 多光谱三指3号 大球形压头
# IMG_PATH = r"data/20260302_115200/test/images/0_rgb_perspective.png"
# PRESET_PATH = r"data/20260302_115200/grid_preset_rgb_perspective.npy"

# 默认检测参数
# DEFAULT_BS = 255
# DEFAULT_C = 17

# 三指1号
DEFAULT_BS = 151
DEFAULT_C = 30

# 多光谱三指3号
# DEFAULT_BS = 153
# DEFAULT_C = 19

# 交互状态
detected_points = []   # 当前参数下自动检测到的点（无序）
sorted_points = []     # 自动检测与手工修正合并后的点（网格顺序）
point_sources = {}     # {(x, y): "auto" | "manual"}
undo_stack = []        # 手工添加/删除前的状态快照
image_width = 0
image_height = 0

CALIBRATION_WINDOW = "Calibration"
ADD_MIN_DISTANCE = 10
DELETE_MAX_DISTANCE = 20


def calculate_V_mark(frame, block_size, c_const):
    """基于局部亮度阈值生成 Mask"""
    # 确保 block_size 是奇数且 >= 3
    if block_size < 3: block_size = 3
    if block_size % 2 == 0: block_size += 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    v_blur = cv2.GaussianBlur(v, (35, 35), 0)

    mask = cv2.adaptiveThreshold(
        src=v_blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c_const,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask_dilate = cv2.dilate(mask_closed, kernel, iterations=1)
    mask_erode = cv2.erode(mask_dilate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask_erode


def detect_markers(mask):
    """从 Mask 检测标记点重心"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for cnt in contours:
        M_moments = cv2.moments(cnt)
        if M_moments["m00"] > 0:
            cx = int(M_moments["m10"] / M_moments["m00"])
            cy = int(M_moments["m01"] / M_moments["m00"])
            markers.append((cx, cy))
    return markers


def group_points_grid(points, rows):
    """
    将任意数量的点尽可能分成从上到下的网格行。

    先按 y 排序，再选择最大的若干个相邻 y 间隔作为行边界；这样即使
    存在少量缺失点或误检点，仍可给出稳定、可视化友好的部分排序。
    """
    unique_points = list(dict.fromkeys((int(p[0]), int(p[1])) for p in points))
    if not unique_points:
        return []

    pts_y_sorted = sorted(unique_points, key=lambda p: (p[1], p[0]))
    row_count = min(max(1, int(rows)), len(pts_y_sorted))

    if row_count == 1:
        return [sorted(pts_y_sorted, key=lambda p: (p[0], p[1]))]

    y_gaps = [
        (pts_y_sorted[i + 1][1] - pts_y_sorted[i][1], i + 1)
        for i in range(len(pts_y_sorted) - 1)
    ]
    # 间隔相同时优先选择靠前的边界，保证排序结果确定。
    split_positions = sorted(
        position
        for _, position in sorted(y_gaps, key=lambda item: (-item[0], item[1]))[:row_count - 1]
    )

    row_groups = []
    start = 0
    for stop in split_positions + [len(pts_y_sorted)]:
        row = sorted(pts_y_sorted[start:stop], key=lambda p: (p[0], p[1]))
        if row:
            row_groups.append(row)
        start = stop

    row_groups.sort(key=lambda row: float(np.mean([p[1] for p in row])))
    return row_groups


def sort_points_grid(points, rows, cols):
    """按从上到下、每行从左到右的顺序，对完整或不完整点集排序。"""
    del cols  # 保留原函数接口；列数仅在完整性校验时使用。
    return [pt for row in group_points_grid(points, rows) for pt in row]


def validate_grid(points, rows, cols):
    """返回网格是否可保存、排序后的点以及各行点数。"""
    row_groups = group_points_grid(points, rows)
    row_counts = [len(row) for row in row_groups]
    valid = (
        len(points) == rows * cols
        and len(row_groups) == rows
        and all(count == cols for count in row_counts)
    )
    ordered = [pt for row in row_groups for pt in row]
    return valid, ordered, row_counts


def compute_mean_grid_spacing(points, rows, cols):
    """计算排序后网格点阵的平均横向/纵向间距。"""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) != rows * cols:
        return None, None

    pts_grid = pts.reshape(rows, cols, 2)
    dx_vals = []
    dy_vals = []

    if cols > 1:
        dx = pts_grid[:, 1:, 0] - pts_grid[:, :-1, 0]
        dx_vals = dx.reshape(-1)

    if rows > 1:
        dy = pts_grid[1:, :, 1] - pts_grid[:-1, :, 1]
        dy_vals = dy.reshape(-1)

    mean_dx = float(np.mean(dx_vals)) if len(dx_vals) > 0 else None
    mean_dy = float(np.mean(dy_vals)) if len(dy_vals) > 0 else None
    return mean_dx, mean_dy


def _window_to_image_coords(x, y):
    """使用 HighGUI 已映射到原图的鼠标坐标，并裁剪到图像范围。"""
    if image_width <= 0 or image_height <= 0:
        return None

    # Windows HighGUI 在触发鼠标回调前已经按照显示窗口大小将坐标
    # 映射到了原始图像坐标系。这里不能再根据窗口尺寸进行二次缩放。
    img_x = int(round(x))
    img_y = int(round(y))
    img_x = int(np.clip(img_x, 0, image_width - 1))
    img_y = int(np.clip(img_y, 0, image_height - 1))
    return img_x, img_y


def _push_undo_state():
    undo_stack.append((list(sorted_points), dict(point_sources)))


def _resort_current_points():
    global sorted_points
    sorted_points = sort_points_grid(sorted_points, N, M)


def mouse_callback(event, x, y, flags, param):
    del flags, param
    global sorted_points, point_sources

    if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        return

    click_pt = _window_to_image_coords(x, y)
    if click_pt is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist_sq = min(
            ((pt[0] - click_pt[0]) ** 2 + (pt[1] - click_pt[1]) ** 2 for pt in sorted_points),
            default=float("inf"),
        )
        if min_dist_sq < ADD_MIN_DISTANCE ** 2:
            print(f"忽略重复添加：{click_pt} 附近已有标记点")
            return

        _push_undo_state()
        sorted_points.append(click_pt)
        point_sources[click_pt] = "manual"
        _resort_current_points()
        print(f"手工添加: {click_pt}, 当前点数: {len(sorted_points)}")
        return

    if not sorted_points:
        return

    closest_pt = min(
        sorted_points,
        key=lambda pt: (pt[0] - click_pt[0]) ** 2 + (pt[1] - click_pt[1]) ** 2,
    )
    min_dist_sq = (closest_pt[0] - click_pt[0]) ** 2 + (closest_pt[1] - click_pt[1]) ** 2
    if min_dist_sq > DELETE_MAX_DISTANCE ** 2:
        print(f"未删除：点击位置 {click_pt} 的 {DELETE_MAX_DISTANCE} 像素内没有标记点")
        return

    _push_undo_state()
    sorted_points.remove(closest_pt)
    point_sources.pop(closest_pt, None)
    _resort_current_points()
    print(f"手工删除: {closest_pt}, 当前点数: {len(sorted_points)}")


def _nothing(x):
    pass


def main():
    global detected_points, sorted_points, point_sources
    global image_width, image_height

    img_raw = cv2.imread(IMG_PATH)
    if img_raw is None:
        print(f"无法读取图片: {IMG_PATH}")
        return

    print(f"目标网格: {N}行 x {M}列 = {N*M}点")
    print("------------------------------------------------")
    print("1. 调整 'BlockSize' 和 'C'，尽可能多地自动检测标记点。")
    print("2. 鼠标左键直接添加缺失点，右键删除附近的误检点。")
    print("3. 每次增删后自动按从上到下、从左到右重新排序。")
    print("快捷键：Z 撤销，R 按当前参数重新检测，Enter 保存，Esc 退出。")
    print("注意：移动任一滑条会清空手工修正并重新自动检测。")
    print("------------------------------------------------")

    cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("BlockSize", "Controls", DEFAULT_BS, 255, _nothing)
    cv2.createTrackbar("C", "Controls", DEFAULT_C, 100, _nothing)
    cv2.setMouseCallback(CALIBRATION_WINDOW, mouse_callback)

    # 同样显示 Mask 窗口方便调试
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

    # 获取图片原始宽高
    h, w = img_raw.shape[:2]
    image_width = w
    image_height = h

    # 设定一个目标显示高度（例如屏幕高度的 80%，或者固定 800 像素）
    # 这样可以保证既能显示完整，又保持比例
    target_height = 800
    scale_ratio = target_height / h
    target_width = int(w * scale_ratio)

    cv2.resizeWindow(CALIBRATION_WINDOW, target_width, target_height)
    cv2.resizeWindow("Mask", target_width, target_height)

    last_params = None
    mask = None

    while True:
        vis = img_raw.copy()

        bs = cv2.getTrackbarPos("BlockSize", "Controls")
        c_val = cv2.getTrackbarPos("C", "Controls")
        current_params = (bs, c_val)

        # 仅在启动、滑条变化或按 R 时重新检测。重新检测会清空手工修正。
        if current_params != last_params:
            mask = calculate_V_mark(img_raw, bs, c_val)
            detected_points = detect_markers(mask)
            sorted_points = sort_points_grid(detected_points, N, M)
            point_sources = {pt: "auto" for pt in sorted_points}
            undo_stack.clear()
            last_params = current_params
            print(
                f">> 自动检测刷新: BlockSize={bs}, C={c_val}, "
                f"检测到 {len(detected_points)}/{N*M} 点"
            )

        cv2.imshow("Mask", mask)

        grid_valid, sorted_points, row_counts = validate_grid(sorted_points, N, M)
        status_color = (0, 255, 0) if grid_valid else (0, 0, 255)
        manual_count = sum(point_sources.get(pt) == "manual" for pt in sorted_points)
        msg = (
            f"Points {len(sorted_points)}/{N*M} | Auto {len(sorted_points)-manual_count} "
            f"| Manual {manual_count} | Rows {row_counts}"
        )

        # 只有完整且每行列数正确时才绘制完整网格连线。
        if grid_valid:
            for i, pt in enumerate(sorted_points):
                if i > 0 and i % M != 0:
                    cv2.line(vis, sorted_points[i - 1], pt, (255, 255, 0), 1)
                if i >= M:
                    cv2.line(vis, sorted_points[i - M], pt, (0, 255, 255), 1)

        for i, pt in enumerate(sorted_points):
            source = point_sources.get(pt, "auto")
            color = (0, 165, 255) if source == "manual" else (0, 255, 0)
            cv2.circle(vis, pt, 5 if source == "manual" else 4, color, -1)
            cv2.putText(
                vis,
                str(i),
                (pt[0] + 6, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        cv2.putText(
            vis,
            "L-click add | R-click delete | Z undo | R redetect | Enter save | Esc quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        cv2.imshow(CALIBRATION_WINDOW, vis)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Esc
            print("退出未保存")
            break
        elif key in (10, 13):  # Enter
            grid_valid, ordered_points, row_counts = validate_grid(sorted_points, N, M)
            if grid_valid:
                sorted_points = ordered_points
                arr = np.asarray(sorted_points, dtype=np.float32)
                first_pt = arr[0]
                mean_dx, mean_dy = compute_mean_grid_spacing(arr, N, M)
                print(f"第一个标记点坐标: ({first_pt[0]:.2f}, {first_pt[1]:.2f})")
                print(
                    "标记点阵列平均间距: "
                    f"dx={mean_dx:.2f}" if mean_dx is not None else "标记点阵列平均间距: dx=N/A",
                    end="",
                )
                print(f", dy={mean_dy:.2f}" if mean_dy is not None else ", dy=N/A")
                np.save(PRESET_PATH, arr)
                print(f"标定完成！已保存为 {PRESET_PATH}")
                break
            else:
                print(
                    f"无法保存：当前 {len(sorted_points)}/{N*M} 点，"
                    f"各行点数为 {row_counts}；要求 {N} 行且每行 {M} 点。"
                )
        elif key in (ord("z"), ord("Z")):
            if undo_stack:
                previous_points, previous_sources = undo_stack.pop()
                sorted_points = previous_points
                point_sources = previous_sources
                _resort_current_points()
                print(f">> 已撤销，当前点数: {len(sorted_points)}")
            else:
                print(">> 没有可撤销的手工操作")
        elif key in (ord("r"), ord("R")):
            last_params = None
            print(">> 将按当前参数重新自动检测并清空手工修正")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
