import cv2
import numpy as np
import setting
from scipy.interpolate import Rbf
import cupy as cp


def init(frame):
    RESCALE = setting.RESCALE
    return cv2.resize(frame, (0, 0), fx=1.0 / RESCALE, fy=1.0 / RESCALE)


def find_marker(frame):
    RESCALE = setting.RESCALE
    # Blur image to remove noise
    blur = cv2.GaussianBlur(frame, (int(63 / RESCALE), int(63 / RESCALE)), 0)

    # subtract the surrounding pixels to magnify difference between markers and background
    diff = frame.astype(np.float32) - blur

    diff *= 4.0
    diff[diff < 0.] = 0.
    diff[diff > 255.] = 255.
    diff = cv2.GaussianBlur(diff, (int(63 / RESCALE), int(63 / RESCALE)), 0)

    # Switch image from BGR colorspace to HSV
    hsv = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2HSV)

    # yellow range in HSV color space
    yellowMin = (0, 0, 32)
    yellowMax = (100, 255, 255)

    # Sets pixels to white if in yellow range, else will be set to black
    mask = cv2.inRange(hsv, yellowMin, yellowMax)

    return mask


def marker_center(mask, frame):
    RESCALE = setting.RESCALE

    areaThresh1 = 90 / RESCALE ** 2
    areaThresh2 = 1920 / RESCALE ** 2
    MarkerCenter = []

    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours[0]) < 25:  # if too little markers, then give up
        print("Too less markers detected: ", len(contours))
        return MarkerCenter

    for contour in contours[0]:
        x, y, w, h = cv2.boundingRect(contour)
        AreaCount = cv2.contourArea(contour)
        # print(AreaCount)
        if AreaCount > areaThresh1 and AreaCount < areaThresh2 and abs(np.max([w, h]) * 1.0 / np.min([w, h]) - 1) < 1:
            t = cv2.moments(contour)
            # print("moments", t)
            # MarkerCenter=np.append(MarkerCenter,[[t['m10']/t['m00'], t['m01']/t['m00'], AreaCount]],axis=0)
            mc = [t['m10'] / t['m00'], t['m01'] / t['m00']]
            # if t['mu11'] < -100: continue
            MarkerCenter.append(mc)
            # print(mc)
            # cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, ( 0, 0, 255 ), 2, 6);

    # 0:x 1:y
    return MarkerCenter


def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occupied, dx, dy, distance = flow
    K = 0
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
            color = (0, 0, 255)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)  # 画箭头

            # 标注位移
            base_x = pt2[0] - 30
            base_y = pt2[1] + 10
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.3
            font_color = (255, 255, 255)
            thickness = 1
            #
            # # 第一行：（Ox,Oy）
            # cv2.putText(
            #     frame,
            #     f"OxOy: {pt1}",
            #     (base_x, base_y),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     font_size,
            #     font_color,
            #     thickness)
            # 第二行：显示 dx（x方向偏移）
            cv2.putText(
                frame,
                f"dx: {dx[i][j]:.2f}",
                (base_x, base_y + 10),  # 第一行位置
                font,
                font_size,
                font_color,
                thickness,
                cv2.LINE_AA
            )

            # 第三行：显示 dy（y方向偏移），y坐标比第一行低15像素（上下间距）
            cv2.putText(
                frame,
                f"dy: {dy[i][j]:.2f}",
                (base_x, base_y + 25),  # 第二行位置（垂直偏移15像素）
                font,
                font_size,
                font_color,
                thickness,
                cv2.LINE_AA
            )

#
# def draw_displacement_field(flow):
#     """绘制位移场"""
#     Ox, Oy, Cx, Cy, Occupied, dx, dy, distance = flow
#     # 数据格式转换
#     Ox_flat = np.array(Ox).flatten()
#     Oy_flat = np.array(Oy).flatten()
#     dx_flat = np.array(dx).flatten()
#     dy_flat = np.array(dy).flatten()
#     # 生成网格
#     x_grid = np.arange(640)
#     y_grid = np.arange(480)
#     xx, yy = np.meshgrid(x_grid, y_grid)
#     # 构建RBF插值器
#     rbf_dx_final = Rbf(Ox_flat, Oy_flat, dx_flat, kernel='multiquadric', epsilon=1.0)
#     rbf_dy_final = Rbf(Ox_flat, Oy_flat, dy_flat, kernel='multiquadric', epsilon=1.0)
#     # 计算位移场
#     dx_field = rbf_dx_final(xx, yy)  # (480, 640)
#     dy_field = rbf_dy_final(xx, yy)  # (480, 640)
#
#     # ---------------------- 可视化 ----------------------
#     dx_min, dx_max = -40, 40  # X方向位移阈值
#     dy_min, dy_max = -40, 40  # Y方向位移阈值
#     # 阈值裁剪
#     dx_clamped_gpu = cp.clip(dx_field, dx_min, dx_max)
#     dy_clamped_gpu = cp.clip(dy_field, dy_min, dy_max)
#     # 归一化
#     dx_denominator = dx_max - dx_min
#     dy_denominator = dy_max - dy_min
#     dx_norm = ((dx_clamped_gpu - dx_min) / dx_denominator * 255).astype(cp.uint8)
#     dy_norm = ((dy_clamped_gpu - dy_min) / dy_denominator * 255).astype(cp.uint8)
#     # 映射
#     dx_color = cv2.applyColorMap(dx_norm, cv2.COLORMAP_VIRIDIS)
#     dy_color = cv2.applyColorMap(dy_norm, cv2.COLORMAP_VIRIDIS)
#     # 显示图像
#     cv2.imshow('X_displacement', dx_color)
#     cv2.imshow('Y_displacement', dy_color)



def warp_perspective(img):
    TOPLEFT = (175, 230)
    TOPRIGHT = (380, 225)
    BOTTOMLEFT = (10, 410)
    BOTTOMRIGHT = (530, 400)

    WARP_W = 215
    WARP_H = 215

    points1 = np.float32([TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT])
    points2 = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])

    matrix = cv2.getPerspectiveTransform(points1, points2)

    result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))

    return result


def init_HSR(img):
    DIM = (640, 480)
    img = cv2.resize(img, DIM)

    K = np.array(
        [[225.57469247811056, 0.0, 280.0069549918857], [0.0, 221.40607131318117, 294.82435570493794], [0.0, 0.0, 1.0]])
    D = np.array([[0.7302503082668154], [-0.18910060205317372], [-0.23997727800712282], [0.13938490908400802]])
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return warp_perspective(undistorted_img)
