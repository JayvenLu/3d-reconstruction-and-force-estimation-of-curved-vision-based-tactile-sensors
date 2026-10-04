# 定义计算水平位移所需进行透视变换的 4 个角点（离线工具）
# 通过鼠标拖动调整角点位置，按 p 打印当前坐标


import cv2
import numpy as np
import os

# -------------------------------------
# 配置：输入 / 输出
# -------------------------------------
IMAGE_PATH = r"cam/20260827-150753/0_rgb.png"   # 修改为你的图片路径
WINDOW_NAME = "Define Boundary (Drag Corners)"

# 鼠标拖动相关全局状态
_dragging = False
_drag_index = -1
_drag_radius = 15
pts_o = None  # 原图上的 4 个角点
_display_scale = 0.5  # 窗口显示的缩放比例（相对于原图），保持宽高比


def order_points(pts: np.ndarray) -> np.ndarray:
    """将 4 个点按顺时针排序: [左上, 右上, 右下, 左下]."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("order_points 需要形状为 (4,2) 的数组")

    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def mouse_callback(event, x, y, flags, param):
    """鼠标回调：拖动 4 个角点（窗口坐标 -> 原图坐标）。"""
    global _dragging, _drag_index, pts_o
    if pts_o is None:
        return

    # 将当前鼠标位置从“显示图坐标”映射回“原图坐标”
    sx = x / _display_scale
    sy = y / _display_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (px, py) in enumerate(pts_o):
            if (sx - px) ** 2 + (sy - py) ** 2 <= _drag_radius ** 2:
                _dragging = True
                _drag_index = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and _dragging and _drag_index >= 0:
        pts_o[_drag_index] = [sx, sy]
    elif event == cv2.EVENT_LBUTTONUP:
        _dragging = False
        _drag_index = -1


def print_points():
    ordered = order_points(pts_o)
    points = [[int(p[0]), int(p[1])] for p in ordered]
    print(f"当前顶点(顺时针): {points}")


def perspective_transform(img, dst_size=None):
    """根据当前角点对输入图像做透视变换。

    dst_size: (W, H)，如果为 None，则使用输入图像的分辨率。
    """
    h, w = img.shape[:2]
    if dst_size is None:
        dst_w, dst_h = w, h
    else:
        dst_w, dst_h = dst_size

    ordered = order_points(pts_o)
    pts_d = np.float32([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]])
    M = cv2.getPerspectiveTransform(ordered, pts_d)
    return cv2.warpPerspective(img, M, (dst_w, dst_h))


def draw_points(img, points):
    """在图像上绘制当前 4 个角点及连线。"""
    overlay = img.copy()
    ordered = order_points(points)

    # 绘制顶点
    for i, (x, y) in enumerate(ordered):
        cv2.circle(overlay, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(overlay, str(i), (int(x) + 10, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 绘制连线
    for i in range(4):
        x1, y1 = ordered[i]
        x2, y2 = ordered[(i + 1) % 4]
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return overlay


def main():
    global pts_o

    if not os.path.isfile(IMAGE_PATH):
        print(f"找不到输入图像: {IMAGE_PATH}")
        return

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        print(f"读取图像失败: {IMAGE_PATH}")
        return

    # 自动读取图像分辨率
    h, w = img.shape[:2]
    print(f"加载图像大小: {w}x{h}")

    # 初始化 4 个角点：使用图像四个角的内缩一点的位置
    margin = 50
    pts_o = np.float32([
        [margin, margin],           # 左上
        [w - margin, margin],       # 右上
        [w - margin, h - margin],   # 右下
        [margin, h - margin],       # 左下
    ])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("操作说明：")
    print("  - 鼠标左键拖动红点调整区域角点位置")
    print("  - 按 p 打印当前角点坐标（顺时针）")
    print("  - 按 s 保存透视变换后的图像")
    print("  - 按 ESC 退出")

    out_path = os.path.splitext(IMAGE_PATH)[0] + "_perspective.png"

    while True:
        # 在原图坐标系上绘制角点和连线
        frame_with_points = draw_points(img, pts_o)
        transformed = perspective_transform(img, dst_size=(w, h))

        # 仅改变显示大小：按比例缩小显示，保持宽高比
        display_frame = cv2.resize(
            frame_with_points,
            None,
            fx=_display_scale,
            fy=_display_scale,
            interpolation=cv2.INTER_AREA,
        )

        display_transformed = cv2.resize(
            transformed,
            None,
            fx=_display_scale,
            fy=_display_scale,
            interpolation=cv2.INTER_AREA,
        )

        cv2.imshow(WINDOW_NAME, display_frame)
        cv2.imshow("Transformed Result", display_transformed)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            print_points()
        elif key == ord('s'):
            cv2.imwrite(out_path, transformed)
            print(f"已保存透视变换结果到: {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

