# python
import cv2
import os
import time
import sys
import platform
from pathlib import Path

# --- 配置 ---
camera_index = "/dev/video0" if platform.system() == "Linux" else 0
camera_device = None
auto_detect_camera = True
# ----------------


def candidate_camera_sources(index_hint=0, device_hint=None):
    seen = set()
    candidates = []

    def add(src):
        key = str(src)
        if key in seen:
            return
        seen.add(key)
        candidates.append(src)

    if device_hint:
        add(device_hint)

    for base in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
        if base.exists():
            for path in sorted(base.iterdir()):
                add(str(path))

    add(index_hint)

    for path in sorted(Path("/dev").glob("video*")):
        add(str(path))

    return candidates


def open_camera_with_auto_detect(index_hint=0, device_hint=None):
    errors = []
    for src in candidate_camera_sources(index_hint=index_hint, device_hint=device_hint):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            errors.append(f"{src}: open failed")
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            errors.append(f"{src}: read failed")
            continue

        return cap, src, errors

    return None, None, errors

dir_timestamp = time.strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join("cam", dir_timestamp)
os.makedirs(output_dir, exist_ok=True)
print(f"图片将保存在 ` {output_dir} ` 文件夹中。", flush=True)
print("按 s 定格保存图片；按 x 撤销；按 q 退出。", flush=True)

if auto_detect_camera:
    cap, selected_camera, detect_errors = open_camera_with_auto_detect(
        index_hint=camera_index,
        device_hint=camera_device,
    )
    if cap is None:
        print("错误：自动检测后仍无法打开任何可用摄像头。", flush=True)
        for err in detect_errors:
            print(f"  - {err}", flush=True)
        sys.exit(1)
    print(f"自动检测成功，当前使用摄像头: {selected_camera}", flush=True)
else:
    source = camera_device if camera_device else camera_index
    cap = cv2.VideoCapture(source)
    selected_camera = source
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {source}。", flush=True)
        sys.exit(1)

# 显式设置分辨率为 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 打印实际生效的分辨率（有些相机/驱动可能不支持，会回退）
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"实际相机分辨率: {actual_w} x {actual_h}", flush=True)

img_counter = 0

window_name = 'Camera (press s/x/q)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, int(actual_w), int(actual_h))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法从摄像头读取画面。", flush=True)
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_name = f"{img_counter}_rgb.png"
            img_path = os.path.join(output_dir, img_name)
            # 保存当前帧到磁盘
            cv2.imwrite(img_path, frame)
            print(f"\n已定格并保存临时图片: ` {img_path} `", flush=True)
            img_counter += 1
            print(f"已保存图片 ` {img_name} `。", flush=True)

        elif key == ord('x'):
            # 撤销上一次保存
            if img_counter == 0:
                print("当前没有已保存的图片可撤销。", flush=True)
            else:
                last_idx = img_counter - 1
                last_name = f"{last_idx}_rgb.png"
                last_path = os.path.join(output_dir, last_name)
                # 删除文件（如果存在）
                if os.path.exists(last_path):
                    try:
                        os.remove(last_path)
                        print(f"已删除文件 ` {last_name} `", flush=True)
                    except Exception as e:
                        print(f"删除文件 ` {last_name} ` 失败: {e}", flush=True)
                else:
                    print(f"文件 ` {last_name} ` 不存在。", flush=True)

                img_counter = max(0, img_counter - 1)

        elif key == ord('q'):
            print("正在退出程序...", flush=True)
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
