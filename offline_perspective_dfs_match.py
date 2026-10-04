# 当前使用的版本

"""Offline DFS matching in perspective-warped coordinates.

Workflow:
1) Warp raw frame into perspective view.
2) Detect markers in warped view and run DFS matching.
3) Backwarp matched grid points to raw coordinates.
4) Interpolate displacement field in raw coordinates.
"""

import os
import glob
import importlib.util
import cv2
import numpy as np
import sys
import time
from typing import List, Tuple

import marker_detection
import find_marker
import setting
import RBF


# ====================== Global config ======================

# Toggle preset grid usage
USE_PRESET_GRID = True

# 多光谱三指3号 小球压头
# PRESET_GRID_PATH = "data/20260114_120750/train/grid_preset_rgb_perspective.npy"
# PTS_SRC = np.float32([[504, 344], [1410, 368], [1530, 1064], [220, 1036]])

# PRESET_GRID_PATH = "data/20260310_151840/grid_preset_rgb_perspective.npy"

# 多光谱三指3号 圆弧形压头
# PRESET_GRID_PATH = "data/20260306_193600/grid_preset_rgb_perspective.npy"

# 曲面三指1号
# PRESET_GRID_PATH = "20260120-165420/grid_preset_rgb_perspective.npy"

# 多光谱三指3号 大球压头
# PTS_SRC = np.float32([[462, 334], [1470, 350], [1602, 1028], [164, 1036]])

# 曲面三指1号
# PTS_SRC = np.float32([[166, 144], [520, 138], [584, 438], [98, 456]])
# PTS_SRC = np.float32([[122, 102], [520, 116], [584, 460], [24, 454]])
# PRESET_GRID_PATH = "data/20260328_121407/train/grid_preset_rgb_perspective.npy"

# 曲面三指2号
PTS_SRC = np.float32([[122, 96], [528, 94], [584, 434], [48, 444]])
PRESET_GRID_PATH = "data/20260404_103156/train/grid_preset_rgb_perspective.npy"



# 多光谱三指3号
# V_BLOCKSIZE = 199  # must be odd
# V_C = 14

# 曲面三指
# V_BLOCKSIZE = 151  # must be odd
# V_C = 30

# 曲面三指2号 推荐参数
V_BLOCKSIZE = 151  # adaptiveThreshold 的 blockSize (必须为奇数)
V_C = 30      # adaptiveThreshold 的 C 参数

# CIRCULARITY_MIN = 0.61
CIRCULARITY_MIN = 0.5

# RBF parameters
CURRENT_KERNEL_TYPE = "inverse_multiquadric"
CURRENT_EPSILON = 80.0
CURRENT_REGULARIZATION = 0.02

WARPED_WINDOW_NAME = "Warped"
RAW_WINDOW_NAME = "Raw"
MASK_WINDOW_NAME = "Mask"



# ===========================================================  


def get_perspective_matrix(pts_src: np.ndarray, pts_dst: np.ndarray):
    return cv2.getPerspectiveTransform(pts_src, pts_dst)


def perspective_transform(img, M, size):
    return cv2.warpPerspective(img, M, size)


def backwarp_points_to_raw(points: List[Tuple[float, float]], M_inv) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    pts_raw = cv2.perspectiveTransform(pts, M_inv)
    pts_raw = pts_raw.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in pts_raw]


def get_timeout_ids_module_path(image_folder: str) -> str:
    return os.path.join(image_folder, f"timeout_ids_b{V_BLOCKSIZE}_c{V_C}.py")


def save_timeout_ids_module(image_folder: str, timeout_frame_ids) -> str:
    module_path = get_timeout_ids_module_path(image_folder)
    timeout_value_repr = "None"
    if timeout_frame_ids is not None:
        timeout_ids_sorted = sorted({int(frame_id) for frame_id in timeout_frame_ids})
        timeout_value_repr = repr(timeout_ids_sorted) if timeout_ids_sorted else "None"

    with open(module_path, "w", encoding="utf-8") as module_file:
        module_file.write(f"BLOCK_SIZE = {V_BLOCKSIZE}\n")
        module_file.write(f"C_VALUE = {V_C}\n")
        module_file.write(f"TIMEOUT_FRAME_IDS = {timeout_value_repr}\n")

    return module_path


def load_timeout_ids_module(image_folder: str):
    module_path = get_timeout_ids_module_path(image_folder)
    if not os.path.exists(module_path):
        return None

    try:
        module_name = f"timeout_ids_b{V_BLOCKSIZE}_c{V_C}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        timeout_frame_ids = getattr(module, "TIMEOUT_FRAME_IDS", None)
        if timeout_frame_ids is None:
            return None
        timeout_frame_ids = [int(frame_id) for frame_id in timeout_frame_ids]
        return timeout_frame_ids if timeout_frame_ids else None
    except Exception as exc:
        print(f"[WARN] Failed to load timeout ids module {module_path}: {exc}")
        return None


def apply_manual_marker_edits(centers, edit_state, remove_radius=18):
    edited_centers = [(int(x), int(y)) for x, y in centers]

    for rx, ry in edit_state.get("remove", []):
        if not edited_centers:
            break
        nearest_idx = -1
        nearest_dist = None
        for idx, (cx, cy) in enumerate(edited_centers):
            dist_sq = (cx - rx) ** 2 + (cy - ry) ** 2
            if nearest_dist is None or dist_sq < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist_sq
        if nearest_idx >= 0 and nearest_dist is not None and nearest_dist <= remove_radius ** 2:
            edited_centers.pop(nearest_idx)

    for ax, ay in edit_state.get("add", []):
        duplicate = any((cx - ax) ** 2 + (cy - ay) ** 2 <= 4 for cx, cy in edited_centers)
        if not duplicate:
            edited_centers.append((int(ax), int(ay)))

    return edited_centers


def warped_mouse_callback(event, x, y, flags, state):
    if state is None:
        return
    if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        return
    scale_x = state.get("scale_x")
    scale_y = state.get("scale_y")
    frame_id = state.get("frame_id")
    if scale_x is None or scale_y is None or frame_id is None:
        return

    wx = int(round(x * scale_x))
    wy = int(round(y * scale_y))
    edits = state["frame_edits"].setdefault(frame_id, {"add": [], "remove": []})
    remove_radius = int(state.get("remove_radius", 18))

    if event == cv2.EVENT_LBUTTONDOWN:
        edits["add"].append((wx, wy))
        state["needs_reprocess"] = True
        print(f"[Manual] Added marker at warped ({wx}, {wy})")
        return

    current_centers = state.get("current_centers", [])
    if not current_centers:
        return

    nearest_center = None
    nearest_dist = None
    for cx, cy in current_centers:
        dist_sq = (cx - wx) ** 2 + (cy - wy) ** 2
        if nearest_dist is None or dist_sq < nearest_dist:
            nearest_center = (int(cx), int(cy))
            nearest_dist = dist_sq

    if nearest_center is None or nearest_dist is None or nearest_dist > remove_radius ** 2:
        return

    removed_from_add = False
    for idx in range(len(edits["add"]) - 1, -1, -1):
        ax, ay = edits["add"][idx]
        if (ax - nearest_center[0]) ** 2 + (ay - nearest_center[1]) ** 2 <= remove_radius ** 2:
            edits["add"].pop(idx)
            removed_from_add = True
            break

    if not removed_from_add:
        edits["remove"].append(nearest_center)

    state["needs_reprocess"] = True
    print(f"[Manual] Removed marker near warped {nearest_center}")


def calculate_V_mark(frame, debug=False, debug_prefix="Vmark"):
    """HSV-V adaptive thresholding on warped frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    if debug:
        cv2.imshow(f"{debug_prefix}_hsv_v", v)

    v_blur = cv2.GaussianBlur(v, (35, 35), 0)

    mask = cv2.adaptiveThreshold(
        src=v_blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=V_BLOCKSIZE,
        C=V_C,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask_dilate = cv2.dilate(mask_closed, kernel, iterations=1)
    mask_erode = cv2.erode(mask_dilate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask_erode


def detect_markers(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 0.0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)

        if circularity < CIRCULARITY_MIN:
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            markers.append((cx, cy))
    return markers


def draw_markers(img, markers, color=(0, 255, 0), size=5):
    for (x, y) in markers:
        cv2.circle(img, (int(x), int(y)), size, color, -1)
    return img


def resize_to_height(img, target_height):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_height / h
    new_w = int(w * scale)
    interpolation = cv2.INTER_NEAREST if len(img.shape) == 2 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, target_height), interpolation=interpolation)


def wrap_text_to_width(text, font, font_scale, thickness, max_width):
    words = text.split()
    if not words:
        return [text]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        (candidate_w, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if candidate_w <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_text_block(
    img,
    lines,
    origin=(10, 10),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    thickness=2,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    bg_alpha=0.45,
    line_gap=8,
    padding=10,
    max_width=None,
):
    if not lines:
        return img

    if max_width is None:
        max_width = img.shape[1] - origin[0] - 10

    wrapped_lines = []
    for entry in lines:
        wrapped_lines.extend(wrap_text_to_width(entry, font, font_scale, thickness, max_width))

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in wrapped_lines]
    text_heights = [size[1] for size in text_sizes]
    box_w = max(size[0] for size in text_sizes) + 2 * padding
    box_h = sum(text_heights) + max(0, len(wrapped_lines) - 1) * line_gap + 2 * padding

    x0, y0 = origin
    x1 = min(img.shape[1], x0 + box_w)
    y1 = min(img.shape[0], y0 + box_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1.0 - bg_alpha, 0.0, img)

    y = y0 + padding
    for line, (_, text_h) in zip(wrapped_lines, text_sizes):
        y += text_h
        cv2.putText(img, line, (x0 + padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_gap
    return img


def make_text_panel(width, height, lines):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    draw_text_block(
        panel,
        lines,
        origin=(10, 10),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.7,
        thickness=2,
        text_color=(220, 240, 255),
        bg_color=(0, 0, 0),
        bg_alpha=1.0,
        line_gap=8,
        padding=10,
        max_width=width - 20,
    )
    return panel


def disp_to_pseudocolor(arr: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a float displacement array to a BGR pseudocolor image.

    Maps zero-centered values to the jet colormap using symmetric range based on
    the maximum absolute value in the array. NaNs are rendered black.
    """
    if arr is None:
        return None
    d = arr.astype(np.float32)
    mask_nan = ~np.isfinite(d)
    if np.all(mask_nan):
        h, w = d.shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    abs_max = float(np.nanmax(np.abs(d)))
    if abs_max <= 0:
        abs_max = 1.0
    vmin = -abs_max
    vmax = abs_max
    norm = (d - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    if mask_nan.any():
        img[mask_nan] = (0, 0, 0)
    return img


def visualize_dense_field_arrows(bg_img: np.ndarray, du: np.ndarray, dv: np.ndarray, step=25, scale=5.0) -> np.ndarray:
    if du is None or dv is None:
        return bg_img
    vis = bg_img.copy()
    H, W = du.shape

    mag = np.sqrt(du ** 2 + dv ** 2)
    max_mag = np.max(mag) if mag.size > 0 else 1.0
    if max_mag < 1e-6:
        return vis

    thr = max_mag * 0.05
    for v in range(step // 2, H, step):
        for u in range(step // 2, W, step):
            dx_val, dy_val = float(du[v, u]), float(dv[v, u])
            m_ = np.sqrt(dx_val ** 2 + dy_val ** 2)
            if m_ < thr:
                continue
            p0 = (u, v)
            p1 = (int(u + dx_val * scale), int(v + dy_val * scale))
            rel = np.clip(m_ / max_mag, 0.0, 1.0)
            color = (0, int(255 * (1 - rel)), int(255 * rel))
            cv2.arrowedLine(vis, p0, p1, color, 2, tipLength=0.3)
    return vis


def draw_flow_arrows(img, Ox, Oy, Cx, Cy, Occupied, M_inv, color=(0, 255, 0), scale=1.0):
    vis = img.copy()
    rows = len(Ox)
    cols = len(Ox[0]) if rows > 0 else 0
    active = 0

    for i in range(rows):
        for j in range(cols):
            # Backwarp reference point
            pt_ref_warp = np.array([[[Ox[i][j], Oy[i][j]]]], dtype=np.float32)
            pt_ref_raw = cv2.perspectiveTransform(pt_ref_warp, M_inv)[0, 0]
            pt_ref = (int(pt_ref_raw[0]), int(pt_ref_raw[1]))
            
            # Draw grid point (gray)
            cv2.circle(vis, pt_ref, 2, (100, 100, 100), -1)

            if Occupied[i][j] > -1:
                active += 1
                # Backwarp current point
                pt_cur_warp = np.array([[[Cx[i][j], Cy[i][j]]]], dtype=np.float32)
                pt_cur_raw = cv2.perspectiveTransform(pt_cur_warp, M_inv)[0, 0]
                
                # Calculate displacement in raw coords
                dx = pt_cur_raw[0] - pt_ref_raw[0]
                dy = pt_cur_raw[1] - pt_ref_raw[1]
                
                p_end = (int(pt_ref[0] + dx * scale), int(pt_ref[1] + dy * scale))
                cv2.arrowedLine(vis, pt_ref, p_end, color, 3, tipLength=0.3)

    return vis, active


def draw_reference_grid_points(img, Ox, Oy, M_inv, color=(160, 160, 160), size=3):
    vis = img.copy()
    rows = len(Ox)
    cols = len(Ox[0]) if rows > 0 else 0

    half = int(max(1, size))
    thickness = 2
    for i in range(rows):
        for j in range(cols):
            pt_ref_warp = np.array([[[Ox[i][j], Oy[i][j]]]], dtype=np.float32)
            pt_ref_raw = cv2.perspectiveTransform(pt_ref_warp, M_inv)[0, 0]
            x = int(round(pt_ref_raw[0]))
            y = int(round(pt_ref_raw[1]))
            # draw horizontal and vertical lines to form a cross
            cv2.line(vis, (x - half, y), (x + half, y), color, thickness)
            cv2.line(vis, (x, y - half), (x, y + half), color, thickness)

    return vis


def draw_reference_grid_points_warped(img, Ox, Oy, color=(160, 160, 160), size=3):
    vis = img.copy()
    rows = len(Ox)
    cols = len(Ox[0]) if rows > 0 else 0

    half = int(max(1, size))
    thickness = 2
    for i in range(rows):
        for j in range(cols):
            x = int(round(Ox[i][j]))
            y = int(round(Oy[i][j]))
            cv2.line(vis, (x - half, y), (x + half, y), color, thickness)
            cv2.line(vis, (x, y - half), (x, y + half), color, thickness)

    return vis


def diagnose_unmatched_point(pt, matcher, Occupied, Cx, Cy):
    px, py = float(pt[0]), float(pt[1])
    reasons = []

    best_row = 0
    best_col = 0
    best_dist_sq = None
    for row in range(matcher.N):
        for col in range(matcher.M):
            ox = float(matcher.O[row][col].x)
            oy = float(matcher.O[row][col].y)
            dist_sq = (px - ox) ** 2 + (py - oy) ** 2
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_row = row
                best_col = col

    nearest_ox = float(matcher.O[best_row][best_col].x)
    nearest_oy = float(matcher.O[best_row][best_col].y)
    nearest_dist = float(np.sqrt(best_dist_sq)) if best_dist_sq is not None else float("inf")

    if nearest_dist > float(matcher.moving_max):
        reasons.append(
            f"超出最大位移约束: dist={nearest_dist:.2f} > moving_max={matcher.moving_max:.2f}"
        )

    if Occupied[best_row][best_col] > -1:
        reasons.append(
            f"最近网格({best_row},{best_col})已被占用: matched=({Cx[best_row][best_col]:.1f}, {Cy[best_row][best_col]:.1f})"
        )

    if best_row > 0 and Occupied[best_row - 1][best_col] > -1 and py <= Cy[best_row - 1][best_col]:
        reasons.append(
            f"与上方同列点顺序冲突: y={py:.1f} <= upper_y={Cy[best_row - 1][best_col]:.1f}"
        )
    if best_row < matcher.N - 1 and Occupied[best_row + 1][best_col] > -1 and py >= Cy[best_row + 1][best_col]:
        reasons.append(
            f"与下方同列点顺序冲突: y={py:.1f} >= lower_y={Cy[best_row + 1][best_col]:.1f}"
        )

    flow_x = px - nearest_ox
    flow_y = py - nearest_oy
    for row_offset, col_offset, label in ((0, -1, "左邻"), (-1, 0, "上邻"), (1, 0, "下邻"), (0, 1, "右邻")):
        nr = best_row + row_offset
        nc = best_col + col_offset
        if nr < 0 or nr >= matcher.N or nc < 0 or nc >= matcher.M:
            continue
        if Occupied[nr][nc] <= -1:
            continue

        neighbor_flow_x = float(Cx[nr][nc]) - float(matcher.O[nr][nc].x)
        neighbor_flow_y = float(Cy[nr][nc]) - float(matcher.O[nr][nc].y)
        flow_gap = float(np.sqrt((flow_x - neighbor_flow_x) ** 2 + (flow_y - neighbor_flow_y) ** 2))
        if flow_gap >= float(matcher.flow_difference_threshold):
            reasons.append(
                f"与{label}流场差过大: gap={flow_gap:.2f} >= threshold={matcher.flow_difference_threshold:.2f}"
            )

    if best_col > 0 and Occupied[best_row][best_col - 1] > -1:
        left_x = float(Cx[best_row][best_col - 1])
        left_y = float(Cy[best_row][best_col - 1])
        dx = px - left_x
        dy = py - left_y
        dist = float(np.sqrt(dx * dx + dy * dy))
        angle_deg = float(np.degrees(np.arcsin(abs(dy) / dist))) if dist > 1e-6 else 0.0
        if dist * dist < float(matcher.dmin) or dist * dist > float(matcher.dmax):
            reasons.append(
                f"与左邻前驱距离不合法: dist={dist:.2f}, allowed=[{np.sqrt(matcher.dmin):.2f}, {np.sqrt(matcher.dmax):.2f}]"
            )
        if angle_deg > float(matcher.theta):
            reasons.append(
                f"与左邻前驱角度过大: angle={angle_deg:.2f} > theta={matcher.theta:.2f}"
            )

    if matcher.did_current_run_timeout():
        reasons.append("DFS时间剪枝触发，完整最优解可能未搜索完")

    if not reasons:
        reasons.append("未违反单一硬约束，但在全局代价最优搜索中被舍弃为 spare/missing")

    return reasons, (best_row, best_col), (nearest_ox, nearest_oy), nearest_dist


def print_unmatched_point_diagnostics(frame_id, unmatched_pts, matcher, Occupied, Cx, Cy):
    if not unmatched_pts:
        print(f"[MATCH] Frame {frame_id}: all detected markers were matched.")
        return

    print(f"[MATCH] Frame {frame_id}: {len(unmatched_pts)} unmatched detected markers")
    for idx, pt in enumerate(unmatched_pts, start=1):
        reasons, nearest_grid, nearest_ref, nearest_dist = diagnose_unmatched_point(pt, matcher, Occupied, Cx, Cy)
        reason_text = " | ".join(reasons)
        print(
            f"  - unmatched[{idx}] warped=({pt[0]:.1f}, {pt[1]:.1f}), "
            f"nearest_grid={nearest_grid}, ref=({nearest_ref[0]:.1f}, {nearest_ref[1]:.1f}), "
            f"dist={nearest_dist:.2f} -> {reason_text}"
        )


def build_rbf_from_flow(
    Ox, Oy, Cx, Cy, Occupied, M_inv, H_raw, W_raw
) -> Tuple[np.ndarray, np.ndarray, int]:
    grid_src_raw = []
    grid_dst_raw = []
    active = 0

    for i in range(len(Ox)):
        for j in range(len(Ox[0])):
            if Occupied[i][j] <= -1:
                continue

            pt_ref = (Ox[i][j], Oy[i][j])
            pt_cur = (Cx[i][j], Cy[i][j])
            src_raw = backwarp_points_to_raw([pt_ref], M_inv)[0]
            dst_raw = backwarp_points_to_raw([pt_cur], M_inv)[0]
            grid_src_raw.append([src_raw[0], src_raw[1]])
            grid_dst_raw.append([dst_raw[0], dst_raw[1]])
            active += 1

    du = np.zeros((H_raw, W_raw), dtype=np.float32)
    dv = np.zeros((H_raw, W_raw), dtype=np.float32)

    if len(grid_src_raw) >= 4:
        du, dv = RBF.compute_displacement_field_in_raw(
            src_points=np.array(grid_src_raw, dtype=np.float32),
            dst_points=np.array(grid_dst_raw, dtype=np.float32),
            height=H_raw,
            width=W_raw,
            kernel_type=CURRENT_KERNEL_TYPE,
            epsilon=CURRENT_EPSILON,
            regularization=CURRENT_REGULARIZATION,
            mask=None,
            show=False,
        )

    return du, dv, active


def offline_main(image_folder, manual=True, target_frame_ids=None):
    global V_BLOCKSIZE, V_C, CIRCULARITY_MIN
    setting.init()

    frame_budget_ms = float("inf") if setting.fps_ <= 0 else 1000.0 / setting.fps_
    manual_marker_edits = {}
    mouse_state = {
        "frame_edits": manual_marker_edits,
        "frame_id": None,
        "scale_x": None,
        "scale_y": None,
        "current_centers": [],
        "needs_reprocess": False,
        "remove_radius": 18,
    }

    def save_dh(frame_id, du, dv):
        save_path = os.path.join(dh_dir, f"{frame_id}_dh.npy")
        np.save(save_path, np.dstack((du, dv)).astype(np.float32))
        return save_path

    preset_path = PRESET_GRID_PATH if USE_PRESET_GRID else None
    matcher = find_marker.Matching(
        N_=setting.N_,
        M_=setting.M_,
        fps_=setting.fps_,
        x0_=setting.x0_,
        y0_=setting.y0_,
        dx_=setting.dx_,
        dy_=setting.dy_,
        preset_grid_path=preset_path
    )

    files = sorted(
        glob.glob(os.path.join(image_folder, "*_rgb.png")),
        key=lambda f: int(os.path.basename(f).split("_")[0]),
    )

    if target_frame_ids is not None:
        target_frame_ids = set(int(frame_id) for frame_id in target_frame_ids)
        files = [
            file_path for file_path in files
            if int(os.path.basename(file_path).split("_")[0]) in target_frame_ids
        ]

    if not files:
        print(f"No images found: {image_folder}")
        return

    dh_dir = os.path.join(os.path.dirname(image_folder.rstrip(os.path.sep)), "dh")
    os.makedirs(dh_dir, exist_ok=True)

    idx = 0
    while idx < len(files):
        img_path = files[idx]
        frame_id = int(os.path.basename(img_path).split("_")[0])
        raw_frame = cv2.imread(img_path)
        if raw_frame is None:
            idx += 1
            continue

        H_raw, W_raw = raw_frame.shape[:2]
        dst = np.float32([[0, 0], [W_raw, 0], [W_raw, H_raw], [0, H_raw]])
        M = get_perspective_matrix(PTS_SRC, dst)
        M_inv = np.linalg.inv(M)

        need_process = True
        t_det = 0.0
        t_match = 0.0
        t_interp = 0.0
        t_total = 0.0

        while True:
            if need_process:
                t_frame_start = time.perf_counter()
                warped = perspective_transform(raw_frame, M, (W_raw, H_raw))
                warped = marker_detection.init(warped)

                t0 = time.perf_counter()
                mask = calculate_V_mark(warped)
                centers = detect_markers(mask)
                if manual:
                    frame_edit_state = manual_marker_edits.setdefault(frame_id, {"add": [], "remove": []})
                    centers = apply_manual_marker_edits(centers, frame_edit_state, remove_radius=mouse_state["remove_radius"])
                t_det = (time.perf_counter() - t0) * 1000.0

                if frame_id == 0 and centers:
                    matcher.init(centers)
                    matcher.set_current_frame_id(frame_id)
                    t0 = time.perf_counter()
                    matcher.run()
                    t_match = (time.perf_counter() - t0) * 1000.0

                    Ox, Oy, Cx, Cy, Occupied, _, _, _ = matcher.get_flow()

                    t0 = time.perf_counter()
                    du, dv, active = build_rbf_from_flow(Ox, Oy, Cx, Cy, Occupied, M_inv, H_raw, W_raw)
                    t_interp = (time.perf_counter() - t0) * 1000.0

                    du = np.zeros((H_raw, W_raw), dtype=np.float32)
                    dv = np.zeros((H_raw, W_raw), dtype=np.float32)
     
   
                else:
                    matcher.init(centers)
                    matcher.set_current_frame_id(frame_id)
                    t0 = time.perf_counter()
                    matcher.run()
                    t_match = (time.perf_counter() - t0) * 1000.0

                    Ox, Oy, Cx, Cy, Occupied, _, _, _ = matcher.get_flow()
                    
                    t0 = time.perf_counter()
                    du, dv, active = build_rbf_from_flow(Ox, Oy, Cx, Cy, Occupied, M_inv, H_raw, W_raw)
                    t_interp = (time.perf_counter() - t0) * 1000.0

                t_total = (time.perf_counter() - t_frame_start) * 1000.0

                if not manual:
                    save_path = save_dh(frame_id, du, dv)
                    print(f"[AUTO] Saved {frame_id}_dh.npy | active={active}")

                if manual:
                    vis_raw = raw_frame.copy()
                    warped_vis = warped.copy()

                    # Separate matched and unmatched points
                    matched_set = set()
                    if Occupied is not None:
                        rows = len(Occupied)
                        cols = len(Occupied[0]) if rows > 0 else 0
                        for r in range(rows):
                            for c in range(cols):
                                if Occupied[r][c] > -1:
                                    matched_set.add((int(Cx[r][c]), int(Cy[r][c])))

                    matched_pts = []
                    unmatched_pts = []
                    for pt in centers:
                        if pt in matched_set:
                            matched_pts.append(pt)
                        else:
                            unmatched_pts.append(pt)

                    matched_raw = backwarp_points_to_raw(matched_pts, M_inv)
                    unmatched_raw = backwarp_points_to_raw(unmatched_pts, M_inv)

                    # print_unmatched_point_diagnostics(frame_id, unmatched_pts, matcher, Occupied, Cx, Cy)

                    vis_raw = draw_reference_grid_points(vis_raw, Ox, Oy, M_inv, color=(255,0,0), size=4)

                    draw_markers(vis_raw, unmatched_raw, color=(0, 165, 255), size=4)
                    draw_markers(vis_raw, matched_raw, color=(0, 255, 0), size=4)

                    warped_vis = draw_reference_grid_points_warped(warped_vis, Ox, Oy, color=(150, 150, 150), size=10)
                    draw_markers(warped_vis, unmatched_pts, color=(0, 165, 255), size=4)
                    draw_markers(warped_vis, matched_pts, color=(0, 255, 0), size=4)

                    vis_raw = visualize_dense_field_arrows(vis_raw, du, dv, step=70, scale=2.0)

                    if Occupied is not None:
                        vis_raw, _ = draw_flow_arrows(vis_raw, Ox, Oy, Cx, Cy, Occupied, M_inv, color=(0, 255, 255), scale=1.0)

                    vis_raw_copy = vis_raw.copy()

                    vis_raw_show = resize_to_height(vis_raw, 480)
                    warped_show = resize_to_height(warped_vis, 480)
                    mask_show = resize_to_height(mask, 480)

                    du_color = disp_to_pseudocolor(du)
                    dv_color = disp_to_pseudocolor(dv)
                    du_show = resize_to_height(du_color, 480)
                    dv_show = resize_to_height(dv_color, 480)

                    raw_info_panel = make_text_panel(
                        width=520,
                        height=vis_raw_show.shape[0],
                        lines=[
                            f"ID: {frame_id} | Active: {active}",
                            f"BlockSize: {V_BLOCKSIZE} | C: {V_C} | Circ: {CIRCULARITY_MIN:.2f}",
                            f"Det: {t_det:.1f}ms | Match: {t_match:.1f}ms | Interp: {t_interp:.1f}ms | Total: {t_total:.1f}ms",
                            "Right: interpolated displacement over raw image",
                            "Keys: Z/X BlockSize -/+",
                            "Keys: C/V Threshold -/+",
                            "Keys: N/M Circularity +/-",
                            "Keys: B Prev frame",
                            "Enter Save dh only",
                            "S Save visuals",
                            "Space Next frame",
                            "Esc Exit",
                            "Mouse on Warped:",
                            "Left add marker",
                            "Right remove nearest marker",
                        ],
                    )
                    raw_combined_show = np.hstack([raw_info_panel, vis_raw_show])

                    mouse_state["frame_id"] = frame_id
                    mouse_state["current_centers"] = [(int(x), int(y)) for x, y in centers]
                    mouse_state["scale_x"] = warped.shape[1] / warped_show.shape[1] if warped_show is not None else 1.0
                    mouse_state["scale_y"] = warped.shape[0] / warped_show.shape[0] if warped_show is not None else 1.0

                    cv2.imshow(RAW_WINDOW_NAME, raw_combined_show)
                    cv2.imshow(WARPED_WINDOW_NAME, warped_show)
                    cv2.setMouseCallback(WARPED_WINDOW_NAME, warped_mouse_callback, mouse_state)
                    cv2.imshow(MASK_WINDOW_NAME, mask_show)
                    if du_show is not None:
                        cv2.imshow("du", du_show)
                    if dv_show is not None:
                        cv2.imshow("dv", dv_show)

                need_process = False

            # Manual mode: blocking interaction and parameter tuning.
            # Auto mode: refresh visualization briefly and move to next frame automatically.
            if manual:
                while True:
                    key = cv2.waitKey(20) & 0xFF
                    if mouse_state["needs_reprocess"]:
                        mouse_state["needs_reprocess"] = False
                        need_process = True
                        break
                    if key == 255:
                        continue
                    break

                if need_process:
                    continue

                if key == 27:
                    return

                is_param_key = False
                if key == ord('z'):
                    V_BLOCKSIZE = max(3, V_BLOCKSIZE - 2)
                    if V_BLOCKSIZE % 2 == 0:
                        V_BLOCKSIZE -= 1
                    is_param_key = True
                elif key == ord('x'):
                    V_BLOCKSIZE = min(501, V_BLOCKSIZE + 2)
                    if V_BLOCKSIZE % 2 == 0:
                        V_BLOCKSIZE += 1
                    is_param_key = True
                elif key == ord('c'):
                    V_C = max(0, V_C - 1)
                    is_param_key = True
                elif key == ord('v'):
                    V_C = min(50, V_C + 1)
                    is_param_key = True
                elif key == ord('n'):
                    CIRCULARITY_MIN = min(1.0, CIRCULARITY_MIN + 0.01)
                    is_param_key = True
                elif key == ord('m'):
                    CIRCULARITY_MIN = max(0.0, CIRCULARITY_MIN - 0.01)
                    is_param_key = True

                if is_param_key:
                    need_process = True
                elif key == ord('b'):
                    idx = max(0, idx - 1)
                    break
                elif key in (13, 10):
                    # Enter pressed: save only dh.npy and advance to next frame.
                    try:
                        save_dh_path = save_dh(frame_id, du, dv)
                        print(f"[SAVED] {save_dh_path}")
                    except Exception as e:
                        print(f"Failed to save dh for frame {frame_id}: {e}")
                    idx += 1
                    break
                elif key == 32:
                    idx += 1
                    break
                elif key == ord('s'):
                    # 's' pressed: save visualization images (mask, warped, raw, du, dv) without advancing frame.
                    try:
                        save_mask_path = os.path.join(image_folder, f"{frame_id}_mask.png")
                        save_warped_path = os.path.join(image_folder, f"{frame_id}_warped.png")
                        save_raw_path = os.path.join(image_folder, f"{frame_id}_dense.png")
                        save_du_img = os.path.join(image_folder, f"{frame_id}_du.png")
                        save_dv_img = os.path.join(image_folder, f"{frame_id}_dv.png")

                        # Ensure folder exists
                        os.makedirs(image_folder, exist_ok=True)

                        # Write mask (single-channel), warped (BGR) and vis_raw_copy (BGR)
                        cv2.imwrite(save_mask_path, mask)
                        cv2.imwrite(save_warped_path, warped)
                        cv2.imwrite(save_raw_path, vis_raw_copy)

                        # Save du/dv images and raw arrays
                        du_color = disp_to_pseudocolor(du)
                        dv_color = disp_to_pseudocolor(dv)
                        if du_color is not None:
                            cv2.imwrite(save_du_img, du_color)
                        if dv_color is not None:
                            cv2.imwrite(save_dv_img, dv_color)

                        print(f"[SAVED_VIS] {save_mask_path}, {save_warped_path}, {save_raw_path}, {save_du_img}, {save_dv_img}")
                    except Exception as e:
                        print(f"Failed to save visualizations for frame {frame_id}: {e}")
                    # remain on current frame
                    continue
                else:
                    continue
            else:
                break

        if not manual:
            idx += 1
            continue

    if not manual:
        timeout_frame_ids = matcher.get_timeout_frame_ids()
        timeout_module_path = save_timeout_ids_module(image_folder, timeout_frame_ids if timeout_frame_ids else None)
        print(f"[AUTO] Saved timeout id file: {timeout_module_path}")
        if timeout_frame_ids:
            timeout_ids_text = ", ".join(str(frame_id) for frame_id in timeout_frame_ids)
            print(f"[AUTO] Timeout frames (DFS pruned at budget {frame_budget_ms:.1f} ms): {timeout_ids_text}")
            print("[AUTO] Switching to manual review for timeout frames...")
            cv2.destroyAllWindows()
            offline_main(image_folder, manual=True, target_frame_ids=timeout_frame_ids)
            return
        else:
            print(f"[AUTO] No DFS timeout frames (budget {frame_budget_ms:.1f} ms).")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    manual_mode = True
    target_frame_ids = None

    # 多光谱三指3号 小球压头
    # img_dir = r"data/20260114_120750/val/images"

    # 多光谱三指3号 小球压头
    # img_dir = r"data/20260310_151840/train/images"

    # 内窥镜三指1号
    # img_dir = r"data/20260328_121407/train/images"

    # 内窥镜三指2号
    img_dir = r"data/20260404_103156/train/images"


    # 多光谱三指3号 大球压头
    # img_dir = r"data/20260306_193600/train/images"

    # 多光谱三指3号 位移检测测试
    # img_dir = r"data/20260114_120750/train/vis/images"

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--auto":
            manual_mode = False
        elif argv[i] == "--image-dir" and i + 1 < len(argv):
            img_dir = argv[i + 1]
            i += 1
        elif argv[i] == "--no-preset":
            USE_PRESET_GRID = False
        elif argv[i] == "--preset" and i + 1 < len(argv):
            USE_PRESET_GRID = True
            PRESET_GRID_PATH = argv[i + 1]
            i += 1
        i += 1

    loaded_timeout_frame_ids = load_timeout_ids_module(img_dir)
    # if loaded_timeout_frame_ids is not None:
    #     manual_mode = True
    #     target_frame_ids = loaded_timeout_frame_ids
    #     print(
    #         f"[MAIN] Loaded timeout frame ids from {get_timeout_ids_module_path(img_dir)}: "
    #         f"{loaded_timeout_frame_ids}"
    #     )

    offline_main(img_dir, manual=manual_mode, target_frame_ids=target_frame_ids)
