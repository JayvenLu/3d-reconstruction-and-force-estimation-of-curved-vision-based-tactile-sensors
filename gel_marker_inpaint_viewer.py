"""Marker mask detection/editor. Image completion lives in geometry_processing.

Historical detector/inpaint imports remain compatible for existing realtime callers.
"""
import os
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from geometry_processing import detect_markers, default_config, legacy_inpaint, save_json, file_hash

BLOCK_SIZE = 151
C_VALUE = 26
marker_centers = []
deleted_markers = set()
deleted_markers_history = []
mask_for_interaction = None
current_interaction_frame = None
raw_mask_scale_x = raw_mask_scale_y = 1.0

def _contour_center(contour):
    """计算单个轮廓的中心点，失败时返回 None。"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def get_marker_contours_and_centers(mask):
    """返回 [(contour, (cx, cy)), ...]，用于稳定地按坐标匹配标记点。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pairs = []
    for contour in contours:
        center = _contour_center(contour)
        if center is not None:
            pairs.append((contour, center))
    return pairs


def find_marker_centers(mask):
    """
    从Mask中找到所有标记点的中心。
    返回: [(cx, cy), ...] 列表
    """
    return [center for _, center in get_marker_contours_and_centers(mask)]

def remove_marker_from_mask(mask, marker_idx):
    """
    从Mask中删除指定索引的标记点。
    返回修改后的Mask。
    """
    result_mask = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if 0 <= marker_idx < len(contours):
        cv2.drawContours(result_mask, [contours[marker_idx]], 0, 0, -1)
    return result_mask


def remove_markers_from_mask_by_center(mask, deleted_centers):
    """根据中心点集合删除标记点，避免轮廓索引在不同 findContours 调用间漂移。"""
    result_mask = mask.copy()
    if not deleted_centers:
        return result_mask

    for contour, center in get_marker_contours_and_centers(mask):
        if center in deleted_centers:
            cv2.drawContours(result_mask, [contour], 0, 0, -1)
    return result_mask

def mouse_callback_on_mask(event, x, y, flags, param):
    """
    处理Mask窗口的鼠标点击事件。
    点击Mask中的标记点可将其删除。
    """
    global marker_centers, deleted_markers, deleted_markers_history, mask_for_interaction, raw_mask_scale_x, raw_mask_scale_y
    
    if event == cv2.EVENT_LBUTTONDOWN and mask_for_interaction is not None and len(marker_centers) > 0:
        # 将鼠标点击坐标从显示坐标变换回原始mask坐标
        # （因为Raw Mask窗口可能被缩放，而marker_centers基于原始坐标）
        x_orig = int(x * raw_mask_scale_x)
        y_orig = int(y * raw_mask_scale_y)
        
        # 查找最近的标记点
        min_dist = float('inf')
        closest_center = None
        for cx, cy in marker_centers:
            center = (cx, cy)
            if center not in deleted_markers:
                dist = np.sqrt((x_orig - cx)**2 + (y_orig - cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_center = center
        
        # 如果点击距离标记点足够近（50像素内），则删除该标记点
        if closest_center is not None and min_dist < 50:
            deleted_markers.add(closest_center)
            deleted_markers_history.append(closest_center)
            print(f"已删除标记点 位置({closest_center[0]}, {closest_center[1]})")

# ===========================================

def generate_marker_mask(frame, block_size=None, c=None):
    params = default_config()["detector"]
    params.update(block_size=BLOCK_SIZE if block_size is None else block_size,
                  c=C_VALUE if c is None else c)
    return detect_markers(frame, params)


def inpaint_markers(frame, mask, method="telea", radius=3, dilate_iter=1):
    return legacy_inpaint(frame, mask, method, radius, dilate_iter)


def inpaint_markers_with_deleted(frame, original_mask, deleted_indices, method="telea", radius=3, dilate_iter=1):
    mask = remove_markers_from_mask_by_center(original_mask, deleted_indices)
    return inpaint_markers(frame, mask, method if method in ("telea", "ns") else "telea", radius, dilate_iter)


def iter_ids_from_normal(normal_dir):
    ids = set()
    if not os.path.isdir(normal_dir):
        return []
    for name in os.listdir(normal_dir):
        if not name.startswith("normal") or not name.endswith(".npy"):
            continue
        core = name[len("normal"): -len(".npy")]
        if not core.isdigit():
            continue
        idx = int(core)
        ids.add(idx)
    return sorted(ids)


def iter_ids_from_images(images_dir):
    ids = set()
    if not os.path.isdir(images_dir):
        return []
    for name in os.listdir(images_dir):
        if not (name.endswith("_rgb.png") or name.endswith("_nir.png")):
            continue
        if name.endswith("_rgb.png"):
            core = name[:-len("_rgb.png")]
        else:
            core = name[:-len("_nir.png")]
        if not core.isdigit():
            continue
        idx = int(core)
        ids.add(idx)
    return sorted(ids)


def resolve_sample_stem(images_dir, idx):
    """
    Return the actual numeric filename stem for a sample ID.

    Datasets in this repo may use either plain IDs (78_rgb.png) or zero-padded
    IDs (0078_rgb.png). Processing logic keeps IDs as integers for sorting and
    filtering, but file IO must use the real stem on disk.
    """
    exact_stem = str(idx)
    for suffix in ("rgb", "nir"):
        if os.path.exists(os.path.join(images_dir, f"{exact_stem}_{suffix}.png")):
            return exact_stem

    candidates = set()
    for suffix in ("rgb", "nir"):
        pattern = os.path.join(images_dir, f"*_{suffix}.png")
        suffix_text = f"_{suffix}.png"
        for path in glob.glob(pattern):
            name = os.path.basename(path)
            stem = name[:-len(suffix_text)]
            if stem.isdigit() and int(stem) == idx:
                candidates.add(stem)

    if not candidates:
        return None

    # Prefer the shortest padded representation if multiple padded widths exist.
    return sorted(candidates, key=lambda stem: (len(stem), stem))[0]


def iter_ids_from_force_log(data_root):
    """
    从 force_log.csv 中读取 phase 为 vertical 的样本 ID。
    """
    csv_path = os.path.join(data_root, 'force_log.csv')
    if not os.path.exists(csv_path):
        print(f"[警告] {csv_path} 不存在，无法通过 force_log 筛选 ID。")
        return None  # 返回 None 表示未找到 CSV

    try:
        df = pd.read_csv(csv_path)
        # 清理列名空格
        df.columns = [c.strip() for c in df.columns]

        if 'phase' not in df.columns:
            print("[警告] force_log.csv 中没有 'phase' 列。")
            return None

        if 'id' not in df.columns:
            print("[警告] force_log.csv 中没有 'id' 列。")
            return None

        # 筛选 phase == 'vertical'
        # 兼容可能的空格或大小写
        df['phase'] = df['phase'].astype(str).str.strip().str.lower()
        vertical_df = df[df['phase'] == 'vertical']
        
        ids = vertical_df['id'].unique().tolist()
        # 转换为 int
        ids = [int(i) for i in ids]
        return sorted(list(set(ids)))

    except Exception as e:
        print(f"[错误] 读取 force_log.csv 出错: {e}")
        return None


def collect_target_ids(data_root, process_scope="all"):
    """
    根据处理范围收集样本 ID。

    process_scope:
    - "all": 使用 images 目录中的全部图像 ID
    - "vertical": 使用 force_log.csv 中 phase=vertical 的 ID，并与 images 目录取交集
    """
    images_dir = os.path.join(data_root, "images")
    image_ids = set(iter_ids_from_images(images_dir))

    if not image_ids:
        print(f"[警告] 在 {images_dir} 中未找到任何 *_rgb.png / *_nir.png。")
        return []

    scope = str(process_scope).strip().lower()
    if scope == "all":
        ids = sorted(image_ids)
        print(f"[信息] 处理范围=all，从 images 中获取到 {len(ids)} 个样本。")
        return ids

    if scope == "vertical":
        vertical_ids = iter_ids_from_force_log(data_root)
        if vertical_ids is None:
            print("[警告] 无法从 force_log.csv 获取 vertical 样本，回退为处理全部图像。")
            ids = sorted(image_ids)
            print(f"[信息] 回退后从 images 中获取到 {len(ids)} 个样本。")
            return ids

        ids = sorted(image_ids.intersection(vertical_ids))
        print(f"[信息] 处理范围=vertical，force_log 中有 {len(vertical_ids)} 个 vertical 样本，与 images 交集后保留 {len(ids)} 个。")
        return ids

    raise ValueError(f"不支持的 process_scope: {process_scope}，仅支持 'all' 或 'vertical'")


def apply_start_sample_id(ids, start_sample_id=None):
    """
    根据起始样本 ID 裁剪待处理列表。

    start_sample_id:
    - None: 不裁剪，保持原顺序
    - int: 从第一个 >= start_sample_id 的样本开始处理
    """
    if start_sample_id is None:
        return ids

    filtered_ids = [idx for idx in ids if idx >= start_sample_id]
    if not filtered_ids:
        print(f"[警告] 未找到 ID >= {start_sample_id} 的样本。")
        return []

    if filtered_ids[0] != start_sample_id:
        print(f"[信息] 未找到样本 ID {start_sample_id}，将从最近的后续样本 {filtered_ids[0]} 开始。")
    else:
        print(f"[信息] 将从样本 ID {start_sample_id} 开始处理。")

    return filtered_ids


def process_dataset(data_root, inpaint_method="telea", inpaint_radius=3, dilate_iter=2, visualize=False,
                    scale=1.0, process_scope="all", start_sample_id=None):
    """Save masks only. Deprecated inpaint arguments are accepted but never applied."""
    import json
    global BLOCK_SIZE, C_VALUE, deleted_markers, deleted_markers_history
    global marker_centers, mask_for_interaction, raw_mask_scale_x, raw_mask_scale_y
    root = Path(data_root)
    output = root / "marker_masks"
    output.mkdir(parents=True, exist_ok=True)
    ids = collect_target_ids(str(root), process_scope)
    # Background is needed by both image completion and difference-input masking.
    if start_sample_id is None and resolve_sample_stem(str(root / "images"), 0) is not None:
        ids = sorted(set(ids) | {0})
    ids = apply_start_sample_id(ids, start_sample_id)
    index = 0
    window = "Marker mask: original | base mask | overlay"
    while index < len(ids):
        sid = ids[index]
        stem = resolve_sample_stem(str(root / "images"), sid)
        path = root / "images" / f"{stem}_rgb.png"
        if not path.exists():
            path = root / "images" / f"{stem}_nir.png"
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        metadata_path = output / f"{sid}.json"
        deleted_markers, deleted_markers_history = set(), []
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            BLOCK_SIZE, C_VALUE = meta["detector"]["block_size"], meta["detector"]["c"]
            if meta.get("image_sha256") == file_hash(path):
                deleted_markers = {tuple(x) for x in meta.get("deleted_centers", [])}
                deleted_markers_history = list(deleted_markers)
        while True:
            params = default_config()["detector"]
            params.update(block_size=BLOCK_SIZE, c=C_VALUE)
            base = detect_markers(frame, params)
            mask = remove_markers_from_mask_by_center(base, deleted_markers)
            action = "save"
            if visualize:
                overlay = frame.copy()
                overlay[mask != 0] = (0.4 * overlay[mask != 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
                panels = [frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay]
                sizes = (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale)))
                canvas = np.concatenate([cv2.resize(x, sizes) for x in panels], axis=1)
                cv2.putText(canvas, f"ID {sid} ({index+1}/{len(ids)}) block={BLOCK_SIZE} C={C_VALUE}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(canvas, "z/x: block  c/v: C  click mask: remove  u: undo  Enter: save  Space: skip  b: back  q: quit",
                            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imshow(window, canvas)
                marker_centers = find_marker_centers(mask)
                mask_for_interaction = mask
                raw_mask_scale_x, raw_mask_scale_y = frame.shape[1]/sizes[0], frame.shape[0]/sizes[1]
                def click(event, x, y, flags, unused):
                    if sizes[0] <= x < 2*sizes[0]:
                        mouse_callback_on_mask(event, x-sizes[0], y, flags, None)
                cv2.setMouseCallback(window, click)
                key = cv2.waitKey(50) & 0xff
                if key in (ord('q'), 27):
                    cv2.destroyAllWindows()
                    return
                if key == ord('z'):
                    BLOCK_SIZE = max(3, BLOCK_SIZE-2)
                    continue
                elif key == ord('x'):
                    BLOCK_SIZE += 2
                    continue
                elif key == ord('c'):
                    C_VALUE -= 1
                    continue
                elif key == ord('v'):
                    C_VALUE += 1
                    continue
                elif key == ord('u'):
                    if deleted_markers_history:
                        deleted_markers.discard(deleted_markers_history.pop())
                    continue
                elif key == ord('b'): action = "back"
                elif key == 32: action = "skip"
                elif key not in (10, 13): continue
            if action == "save":
                # Recompute after mouse edits before writing; native image resolution.
                mask = remove_markers_from_mask_by_center(base, deleted_markers)
                if not cv2.imwrite(str(output / f"{sid}.png"), mask):
                    raise OSError("Failed to save marker mask")
                save_json(metadata_path, {"schema_version": 1, "sample_id": sid, "image": path.name,
                    "image_sha256": file_hash(path), "detector": params,
                    "deleted_centers": sorted(deleted_markers), "extra_dilation_applied": False})
                print(f"[Mask] saved {output / f'{sid}.png'}")
            index = max(0, index-1) if action == "back" else index+1
            break
    if visualize:
        cv2.destroyAllWindows()


def process_dataset_by_normal(*args, **kwargs):
    return process_dataset(*args, **kwargs)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Detect/edit marker masks; never writes repaired images")
    p.add_argument("--data-root", default="data/20260404_103156/train")
    p.add_argument("--batch", action="store_true", help="Save automatically without GUI")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--scope", choices=("all", "vertical"), default="all")
    p.add_argument("--start-id", type=int)
    p.add_argument("--block-size", type=int, default=151)
    p.add_argument("--c", type=float, default=26)
    a = p.parse_args()
    global BLOCK_SIZE, C_VALUE
    BLOCK_SIZE, C_VALUE = a.block_size, a.c
    process_dataset(a.data_root, visualize=not a.batch, scale=a.scale,
                    process_scope=a.scope, start_sample_id=a.start_id)


if __name__ == "__main__":
    main()
