# 不计算水平位移，仅记录力数据和图像
# 无相机模式时可用于标定多光谱传感器的角点位置，需同步DP显示

import serial
import time
import numpy as np
import cv2
import os
import math
import threading
import re
import csv
import socket
import struct
import sys
import select
from collections import deque
from datetime import datetime
import traceback

if os.name == 'nt':
    import msvcrt
else:
    import tty
    import termios

# ================== 配置 ==================

# SERIAL_PORT = 'COM14'
SERIAL_PORT = '/dev/ttyCNC'
BAUD_RATE = 115200

# 摄像头配置（支持多种指定方式）
# 方式 0: 自动检测当前连接并可读帧的相机（Linux 下优先使用稳定设备名）
AUTO_DETECT_CAMERA = True
ASYNC_CAMERA_READ = True

# 方式 1: 使用索引 (推荐)
CAMERA_INDEX = 0  # 0 = /dev/video0, 1 = /dev/video1, ...

# 方式 2: 使用设备路径 (仅Linux/Ubuntu，自动检测失败时回退到此配置)
CAMERA_DEVICE = None  # 例如: '/dev/video0' 或 '/dev/v4l/by-id/...'

WINDOW_NAME = "Live Feed"

# ======相机参数======
# 三指1号
# 白平衡参数=4600 曝光=670

# 三指2号
# 白平衡参数=4600 曝光=450

# ================= KWR36 传感器配置 =================
SENSOR_IP = "192.168.50.25"
SENSOR_PORT = 8886

# 力显示单位缩放（KWR36底层已换算为 N，此处设为 1.0 即可）
FORCE_SCALE = 1.0
FORCE_FILTER_ENABLED = True  # 是否启用低通滤波
FORCE_FILTER_CUTOFF_HZ = 10.0  # 低通滤波截止频率，单位 Hz
FORCE_SAMPLE_RATE_HZ = 1000.0  # 传感器采样率，单位 Hz
FORCE_HISTORY_SEC = 15.0  # 历史缓存长度，单位秒
FORCE_TARE_WINDOW_SEC = 1.0  # 去皮时均值窗口长度，单位秒
FORCE_STABILIZE_SEC = 0.3   # 达标后等待应力松弛再复检的时长，单位秒
FZ_REL_TOL = 0.20  # Fz 相对容差（双边窗口）
FZ_ABS_TOL = 0.05  # Fz 绝对容差（N）
FH_REL_TOL = 0.20  # Fh 相对容差（双边窗口）
FH_ABS_TOL = 0.05  # Fh 绝对容差（N）
MAX_VERTICAL_FZ_WINDOW_ITERS = 5  # 垂直阶段 Fz 双边窗口调节的最大循环次数，防止死循环
Z_MIN_STEP = 0.0025  # Z 最小步长，1 个脉冲 = 0.0025 mm
XY_MIN_STEP = 0.0025  # XY 最小步长，1 个脉冲 = 0.0025 mm

# 基线偏置（运行时标定后赋值）——传感器坐标系 A 下的偏置
FORCE_BIAS = np.array([0.0, 0.0, 0.0], dtype=float)
FORCE_AXES_LABEL = 'target_vts'

# 传感器与位置数值安全阈值（用于过滤通信抖动/坏帧）
FORCE_ABS_MAX_N = 500.0
CNC_POS_ABS_MAX_MM = 100000.0

# 标定区域（左下, 右上）验证集
# BOTTOM_LEFT  = {'x': 103.8, 'y': 146.2}
# TOP_RIGHT    = {'x': 111.8, 'y': 138.2}

# 标定区域（左下, 右上）测试集
# BOTTOM_LEFT  = {'x': 104.2, 'y': 146.3}
# TOP_RIGHT    = {'x': 111.9, 'y': 138.3}

# 标定区域（左下, 右上）训练集  三指1号
# BOTTOM_LEFT  = {'x': 104, 'y': 152}
# TOP_RIGHT    = {'x': 111, 'y': 142}

# 标定区域（左下, 右上）训练集  三指2号
BOTTOM_LEFT  = {'x': 104, 'y': 151}
TOP_RIGHT    = {'x': 112, 'y': 142}

# 三指1号 法向量标定 训练集采集参数
# GRID_SIZE = 10
# VERTICAL_FORCE_STEPS = 2
# MAX_VERTICAL_FORCE = 0.6
# HORIZONTAL_DIRECTIONS = 1
# HORIZONTAL_FORCE_STEPS = 1
# MAX_HORIZONTAL_FORCE = 0

# 三指1号 法向量标定 训练集采集参数
# GRID_SIZE = 5
# VERTICAL_FORCE_STEPS = 2
# MAX_VERTICAL_FORCE = 0.6
# HORIZONTAL_DIRECTIONS = 1
# HORIZONTAL_FORCE_STEPS = 1
# MAX_HORIZONTAL_FORCE = 0

# 三指 力标定 训练集采集参数
GRID_SIZE = 5
VERTICAL_FORCE_STEPS = 4
MAX_VERTICAL_FORCE = 1.5
HORIZONTAL_DIRECTIONS = 8
HORIZONTAL_FORCE_STEPS = 4
MAX_HORIZONTAL_FORCE = 0.9
USE_FIXED_HORIZONTAL_FZ = False  # False: 每级 Fz 后立即采水平力；True: 全部 Fz 完成后在固定 Fz 下采水平力
FIXED_HORIZONTAL_FZ_TARGET = 1.5  # USE_FIXED_HORIZONTAL_FZ=True 时生效

# 三指1号 力标定 测试集采集参数
# GRID_SIZE = 4                  # 4x4 网格，正好插在 5x5 的空隙中
# VERTICAL_FORCE_STEPS = 4       # 3个非零力阶梯
# MAX_VERTICAL_FORCE = 1.8       # 避开2.0N，力阶梯为: [0.6, 1.2, 1.8]
# HORIZONTAL_DIRECTIONS = 6      # 6个方向: 0, 60, 120, 180, 240, 300度 (避开了45度系列)
# HORIZONTAL_FORCE_STEPS = 5     # 3个非零力阶梯
# MAX_HORIZONTAL_FORCE = 1.05    # 力阶梯为: [0.35, 0.70, 1.05]

# 三指1号 力标定 验证集
# GRID_SIZE = 3                  # 3x3 稀疏网格
# VERTICAL_FORCE_STEPS = 3       # 2个非零力阶梯: [0.8, 1.6]
# MAX_VERTICAL_FORCE = 1.6       
# HORIZONTAL_DIRECTIONS = 4      # 4个方向: 0, 90, 180, 270
# HORIZONTAL_FORCE_STEPS = 5     # 2个非零水平力: [0.45, 0.9]
# MAX_HORIZONTAL_FORCE = 0.9

# 三指1号 采集测试
# GRID_SIZE = 2                  # 2x2 稀疏网格
# VERTICAL_FORCE_STEPS = 2       # 2个非零力阶梯: [0.8, 1.6]
# MAX_VERTICAL_FORCE = 1.6       
# HORIZONTAL_DIRECTIONS = 1      # 2个方向: 0, 180
# HORIZONTAL_FORCE_STEPS = 3     # 2个非零水平力: [0.45, 0.9]
# MAX_HORIZONTAL_FORCE = 0


# 每个水平方向最多尝试的步数，若达不到目标水平力则跳过该方向后续更大水平力
MAX_STEPS_PER_DIRECTION = 100
FH_RECHECK_MAX_RETRIES = 5  # Fh 超上界后的最大回退/重试次数，防止死循环
HORIZONTAL_FORCE_RATIO = 0.7  # USE_FIXED_HORIZONTAL_FZ=False 时，水平阶段最大 Fh = 当前 Fz * 该比例
MIN_HORIZONTAL_FORCE_TARGET = 0  # 水平力 target 下限，低于 0.3N 时提升到 0.3N

# 运动参数
Z_SAFE_HEIGHT = 135 # 小球压头
Z_PROBE_SPEED = 60.0
XY_PROBE_SPEED = 60.0
RAPID_SPEED = 1000.0
Z_STEP = 0.01
XY_STEP = 0.02
VERTICAL_Z_SETTLE_SEC = 0.05  # Z 微调后的等待时间，给机械与传感器响应留余量

# 存储
DATA_DIR = "force_data"

# 双 Z 同步检测配置
DUAL_Z_ENABLED = True  # 是否启用双 Z 检测
SECOND_Z_AXIS = 'A'  # 第二个 Z 电机所对应的轴字母(如未配置改为 None)
Z_SYNC_MIN_DELTA = 0.01  # 位移阈值(一侧小于该值而另一侧大于则判定异常)
Z_SYNC_CHECK_INTERVAL = 0.05  # 采样间隔秒
Z_SYNC_CHECK_DURATION = 0.4  # 在一次 Z 移动后检测总时长

# 采集中断标志（ESC 退出）
STOP_REQUESTED = False


class TerminalKeyReader:
    """跨平台终端非阻塞按键读取（支持后台线程监听，不受窗口焦点影响）。"""
    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_term = None
        self.current_key = None
        self.lock = threading.Lock()
        self.reader_thread = None
        self.running = False

    def enable(self):
        if os.name == 'nt':
            # Windows 下使用 msvcrt，无需修改终端模式
            self.enabled = True
            self.running = True
            self.reader_thread = threading.Thread(target=self._windows_reader_loop, daemon=True)
            self.reader_thread.start()
            return

        if not sys.stdin.isatty():
            print("[提示] 当前标准输入不是终端，键盘热键功能不可用。")
            return
        try:
            self.fd = sys.stdin.fileno()
            self.old_term = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
            self.running = True
            # 启动后台读取线程
            self.reader_thread = threading.Thread(target=self._unix_reader_loop, daemon=True)
            self.reader_thread.start()
        except Exception as e:
            self.enabled = False
            print(f"[警告] 无法启用终端按键监听: {e}")

    def disable(self):
        self.running = False
        if os.name == 'nt':
            self.enabled = False
            return

        if self.enabled and self.fd is not None and self.old_term is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_term)
            except Exception:
                pass
        self.enabled = False
        self.fd = None
        self.old_term = None

    def _windows_reader_loop(self):
        """Windows 后台键盘读取线程"""
        while self.running:
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    # 方向键等扩展键：先读到 0x00/0xE0，再读一个键值
                    if ch in (b'\x00', b'\xe0'):
                        if msvcrt.kbhit():
                            _ = msvcrt.getch()
                        continue
                    if ch == b'\x1b':
                        with self.lock:
                            self.current_key = 'esc'
                        print("[键盘] ESC 被检测到", flush=True)
                    else:
                        try:
                            key_str = ch.decode('utf-8', errors='ignore').lower()
                            with self.lock:
                                self.current_key = key_str
                        except Exception:
                            pass
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def _unix_reader_loop(self):
        """Linux/Unix 后台键盘读取线程"""
        while self.running:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        # 吞掉可能的转义序列剩余字符（如方向键）
                        while True:
                            r2, _, _ = select.select([sys.stdin], [], [], 0)
                            if not r2:
                                break
                            _ = sys.stdin.read(1)
                        with self.lock:
                            self.current_key = 'esc'
                        print("[键盘] ESC 被检测到", flush=True)
                    elif ch:
                        with self.lock:
                            self.current_key = ch.lower()
            except Exception:
                time.sleep(0.01)

    def poll_key(self, timeout=0.0):
        """
        返回单字符按键；ESC 返回 'esc'；无输入返回 None。
        使用后台线程，不依赖终端焦点。
        """
        if not self.enabled:
            if timeout > 0:
                time.sleep(timeout)
            return None

        # 如果后台线程正在运行，直接从缓冲区读取
        if self.running:
            with self.lock:
                key = self.current_key
                self.current_key = None  # 清除已读的按键
            if timeout > 0 and key is None:
                time.sleep(timeout)
            return key

        # Fallback: 如果线程未运行，使用轮询模式（向后兼容）
        if timeout > 0:
            time.sleep(timeout)
        return None


KEY_READER = TerminalKeyReader()

# ============== KWR36 传感器通信类 ==============

class LowPassFilter:
    def __init__(self, fs: float, cutoff_hz: float = FORCE_FILTER_CUTOFF_HZ, channels: int = 6):
        self.fs = float(fs)
        self.cutoff_hz = float(cutoff_hz)
        self.channels = int(channels)

        dt = 1.0 / self.fs
        rc = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        self.alpha = dt / (rc + dt)
        self.initialized = False
        self.y = np.zeros(self.channels, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.y[:] = x
            self.initialized = True
        else:
            self.y[:] = self.y + self.alpha * (x - self.y)
        return self.y.copy()

class KWR36Sensor:
    def __init__(
        self,
        ip=SENSOR_IP,
        port=SENSOR_PORT,
        filter_enabled: bool = FORCE_FILTER_ENABLED,
        lowpass_cutoff_hz: float = FORCE_FILTER_CUTOFF_HZ,
        sample_rate_hz: float = FORCE_SAMPLE_RATE_HZ,
        history_sec: float = FORCE_HISTORY_SEC,
        tare_window_sec: float = FORCE_TARE_WINDOW_SEC,
    ):
        self.ip = ip
        self.port = port
        self.filter_enabled = bool(filter_enabled)
        self.lowpass_cutoff_hz = float(lowpass_cutoff_hz)
        self.sample_rate_hz = float(sample_rate_hz)
        self.history_sec = float(history_sec)
        self.tare_window_sec = float(tare_window_sec)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        self.running = False
        self.thread = None
        self.latest_raw_data = np.zeros(6, dtype=np.float64)
        self.latest_filtered_data = np.zeros(6, dtype=np.float64)
        self.tare_offset_raw = np.zeros(6, dtype=np.float64)
        self.tare_offset_filtered = np.zeros(6, dtype=np.float64)
        self.lowpass_filter = LowPassFilter(fs=self.sample_rate_hz, cutoff_hz=self.lowpass_cutoff_hz, channels=6)
        self.times = deque()
        self.raw_history = deque()
        self.filtered_history = deque()
        self.lock = threading.Lock()

    def start(self):
        print(f"正在连接 KWR36 传感器 {self.ip}:{self.port} ...")
        self.sock.connect((self.ip, self.port))
        self.running = True
        
        # 发送 1KHz 连续读取指令 (48 AA 0D 0A)
        self.sock.sendall(bytes.fromhex('48 AA 0D 0A'))
        
        # 开启后台线程持续读取与解析
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        time.sleep(0.5) # 等待第一批数据到达
        print("KWR36 传感器连接并启动成功！")

    def _read_loop(self):
        buffer = b''
        while self.running:
            try:
                data = self.sock.recv(1024)
                if not data:
                    continue
                buffer += data
                
                # 寻找并解析 28 字节的数据帧
                while len(buffer) >= 28:
                    # 查找帧头 0x48 0xAA
                    idx = buffer.find(b'\x48\xAA')
                    if idx == -1:
                        buffer = b''
                        break
                    
                    if idx > 0:
                        buffer = buffer[idx:]
                        if len(buffer) < 28:
                            break
                    
                    # 校验帧尾 0x0D 0x0A
                    if buffer[26:28] == b'\x0D\x0A':
                        frame = buffer[0:28]
                        try:
                            # 解析 6 个小端单精度浮点数
                            parsed = np.array(struct.unpack('<ffffff', frame[2:26]), dtype=np.float64)
                            
                            # 数据有效性检查：确保所有值在合理范围内
                            # 传感器范围通常在 ±10N 左右，允许5倍范围作为缓冲
                            if np.all(np.abs(parsed) < 1000) and np.all(np.isfinite(parsed)):
                                raw_si = parsed * 9.80665
                                filtered_si = self.lowpass_filter.process(raw_si)
                                timestamp = time.time()
                                with self.lock:
                                    self.latest_raw_data = raw_si
                                    self.latest_filtered_data = filtered_si
                                    self.times.append(timestamp)
                                    self.raw_history.append(self.latest_raw_data.copy())
                                    self.filtered_history.append(self.latest_filtered_data.copy())

                                    while self.times and (self.times[-1] - self.times[0] > self.history_sec):
                                        self.times.popleft()
                                        self.raw_history.popleft()
                                        self.filtered_history.popleft()
                                
                                buffer = buffer[28:] # 推进缓冲区
                            else:
                                # 数据超出范围，跳过此帧
                                print(f"[KWR36] 数据超出范围或含NaN/Inf: {parsed}", flush=True)
                                buffer = buffer[2:]
                        except Exception as e:
                            # 解析失败，跳过此帧头
                            print(f"[KWR36] 帧解析失败: {e}", flush=True)
                            buffer = buffer[2:]
                    else:
                        # 帧尾不对，可能是错位了，跳过帧头继续找
                        buffer = buffer[2:]
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"\n[KWR36 读取错误]: {e}")
                self.running = False  # 标记为停止
                break

    def _get_recent_mean_locked(self, history_deque: deque, window_sec: float) -> tuple[np.ndarray, int]:
        if not self.times or not history_deque:
            return np.zeros(6, dtype=np.float64), 0

        latest_t = self.times[-1]
        selected = []
        for t, value in zip(reversed(self.times), reversed(history_deque)):
            if latest_t - t > window_sec:
                break
            selected.append(value)

        if not selected:
            return np.zeros(6, dtype=np.float64), 0

        selected.reverse()
        selected_arr = np.asarray(selected, dtype=np.float64)
        return np.mean(selected_arr, axis=0), selected_arr.shape[0]

    def getMeasurement(self):
        """返回当前的受力信息 [Fx, Fy, Fz, Mx, My, Mz]"""
        if not self.running:
            raise RuntimeError("Sensor disconnected")
        with self.lock:
            if self.filter_enabled:
                measurement = self.latest_filtered_data - self.tare_offset_filtered
            else:
                measurement = self.latest_raw_data - self.tare_offset_raw
            return measurement.tolist()

    def tare(self, samples=100):
        window_sec = self.tare_window_sec
        if samples is not None:
            try:
                samples = int(samples)
                if samples > 0:
                    window_sec = max(samples / self.sample_rate_hz, 1.0 / self.sample_rate_hz)
            except Exception:
                pass

        with self.lock:
            tare_raw, raw_count = self._get_recent_mean_locked(self.raw_history, window_sec)
            tare_filtered, filtered_count = self._get_recent_mean_locked(self.filtered_history, window_sec)

            if raw_count == 0:
                tare_raw = self.latest_raw_data.copy()
                raw_count = 1
            if filtered_count == 0:
                tare_filtered = self.latest_filtered_data.copy()
                filtered_count = 1

            self.tare_offset_raw = tare_raw
            self.tare_offset_filtered = tare_filtered

        tare_show = self.tare_offset_filtered if self.filter_enabled else self.tare_offset_raw
        print(
            f"[KWR36 TARE] 已去皮（均值窗口={window_sec:.3f}s，Raw样本={raw_count}，Filtered样本={filtered_count}）: "
            f"Fx={tare_show[0]:.3f}N Fy={tare_show[1]:.3f}N Fz={tare_show[2]:.3f}N"
        )

    def stopStreaming(self):
        self.running = False
        try:
            # 发送停止指令
            self.sock.sendall(bytes.fromhex('43 AA 0D 0A'))
            self.sock.close()
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=1.0)


# ============== 工具函数 ==============

def list_available_cameras():
    """
    列出系统中所有可用的摄像头（仅Linux/Ubuntu）。
    返回: [(index, device_path, description), ...]
    """
    if os.name == 'nt':
        print("[信息] Windows 下无法直接列举摄像头，请手动指定 CAMERA_INDEX")
        return []
    
    available = []
    for i in range(10):  # 检查 /dev/video0 到 /dev/video9
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            # 尝试打开并检测
            try:
                cap = cv2.VideoCapture(device_path)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append((i, device_path, f"Camera {i} ({device_path})"))
                    cap.release()
            except Exception:
                pass
    
    return available


def _unique_preserve_order(items):
    seen = set()
    unique = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def get_linux_camera_candidates(preferred_device=None, preferred_index=None):
    """
    获取 Linux 下按优先级排序的候选相机路径：
    1. 用户手动指定的稳定路径/设备路径
    2. /dev/v4l/by-id 下的稳定符号链接
    3. /dev/v4l/by-path 下的拓扑符号链接
    4. 用户指定的 /dev/video{index}
    5. 系统中的所有 /dev/video*
    """
    candidates = []

    if preferred_device:
        candidates.append(preferred_device)

    for link_dir in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        if os.path.isdir(link_dir):
            try:
                for name in sorted(os.listdir(link_dir)):
                    candidates.append(os.path.join(link_dir, name))
            except Exception:
                pass

    if preferred_index is not None:
        candidates.append(f"/dev/video{preferred_index}")

    try:
        video_nodes = [
            os.path.join("/dev", name)
            for name in sorted(os.listdir("/dev"))
            if re.fullmatch(r"video\d+", name)
        ]
        candidates.extend(video_nodes)
    except Exception:
        pass

    return _unique_preserve_order(candidates)


def detect_working_camera(camera_device=None, camera_index=None):
    """
    自动检测当前已连接且能成功读取帧的相机。
    Linux 下优先尝试稳定符号链接路径；Windows 下回退到索引。
    返回: (cap, success, selected_source)
    """
    if os.name == 'nt':
        try:
            index = 0 if camera_index is None else int(camera_index)
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, True, str(index)
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return None, False, None

    candidates = get_linux_camera_candidates(
        preferred_device=camera_device,
        preferred_index=camera_index,
    )

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        cap = None
        try:
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
            if not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue

            ret, _ = cap.read()
            if ret:
                return cap, True, candidate
        except Exception:
            pass

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    return None, False, None


class AsyncCameraReader:
    """
    后台线程持续抓取相机帧，仅保留最新一帧，避免主线程每次显示/保存都阻塞在 cap.read()。
    对外尽量模拟 cv2.VideoCapture 的最小接口。
    """
    def __init__(self, cap, source_label="camera"):
        self.cap = cap
        self.source_label = source_label
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ok = False
        self.running = True
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.latest_ok = True
                    self.started.set()
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.02)

    def wait_until_ready(self, timeout_sec=1.0):
        if self.started.wait(timeout=timeout_sec):
            return True
        with self.lock:
            return self.latest_frame is not None

    def isOpened(self):
        try:
            return self.cap.isOpened()
        except Exception:
            return False

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


def wrap_camera_capture(cap, source_label="camera"):
    """
    为已打开的 VideoCapture 创建异步读取包装。
    """
    if cap is None:
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    reader = AsyncCameraReader(cap, source_label=source_label)
    if reader.wait_until_ready(timeout_sec=1.0):
        return reader
    reader.release()
    return None


def get_force_tolerance(target, rel_tol, abs_tol):
    """计算力偏差：相对和绝对的最大值"""
    return max(abs(float(target)) * float(rel_tol), float(abs_tol))


def get_force_window(target, rel_tol, abs_tol):
    """计算双边力窗口 [lower, upper, tolerance]"""
    tol = get_force_tolerance(target, rel_tol, abs_tol)
    lower = max(0.0, float(target) - tol)
    upper = float(target) + tol
    return lower, upper, tol


def get_adaptive_xy_step(base_step, retry_count, min_step=XY_MIN_STEP):
    """根据失败次数计算自适应 XY 步长（指数衰减）"""
    retry_count = max(0, int(retry_count))
    return max(float(min_step), float(base_step) / (2 ** retry_count))


def get_adaptive_z_step(base_step, retry_count, min_step=Z_MIN_STEP):
    """根据失败次数计算自适应 Z 步长（指数衰减）"""
    retry_count = max(0, int(retry_count))
    return max(float(min_step), float(base_step) / (2 ** retry_count))


def init_camera(camera_index=None, camera_device=None, auto_detect=False):
    """
    跨平台摄像头初始化
    
    参数:
        camera_index (int): 摄像头索引（0, 1, ...）
        camera_device (str): 设备路径，仅Linux/Ubuntu (如 '/dev/video0')
                            若设置则优先使用此路径，忽略 camera_index
    
    返回: (cap, success)
        cap: cv2.VideoCapture 对象或 None
        success: 是否初始化成功
    """
    cap = None

    if auto_detect:
        cap, success, selected_source = detect_working_camera(
            camera_device=camera_device,
            camera_index=camera_index,
        )
        if success:
            if ASYNC_CAMERA_READ:
                async_cap = wrap_camera_capture(cap, source_label=selected_source)
                if async_cap is not None:
                    cap = async_cap
                else:
                    print(f"[警告] 相机异步线程启动失败，继续使用同步模式: {selected_source}")
            print(f"[相机] 自动检测成功，使用: {selected_source}")
            return cap, True
        print("[警告] 自动检测相机失败，回退到手动配置。")
    
    # 优先使用设备路径（仅Linux）
    if camera_device is not None and os.name != 'nt':
        if os.path.exists(camera_device):
            try:
                cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)  # Linux 下使用 V4L2 后端
                if cap.isOpened():
                    if ASYNC_CAMERA_READ:
                        async_cap = wrap_camera_capture(cap, source_label=camera_device)
                        if async_cap is not None:
                            cap = async_cap
                        else:
                            print(f"[警告] 相机异步线程启动失败，继续使用同步模式: {camera_device}")
                    print(f"[相机] 使用设备路径: {camera_device}")
                    return cap, True
                else:
                    print(f"[警告] 无法打开设备 {camera_device}")
                    cap = None
            except Exception as e:
                print(f"[错误] 打开设备 {camera_device} 失败: {e}")
                cap = None
        else:
            print(f"[警告] 设备 {camera_device} 不存在")
    
    # 回退到索引方式
    if camera_index is None:
        camera_index = 0
    
    try:
        if os.name == 'nt':
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            print(f"[相机] Windows DirectShow 后端, 索引: {camera_index}")
        else:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)  # Linux 下使用 V4L2 后端
            print(f"[相机] Linux/Ubuntu 后端，索引: {camera_index}")
        
        if cap.isOpened():
            if ASYNC_CAMERA_READ:
                async_cap = wrap_camera_capture(cap, source_label=str(camera_index))
                if async_cap is not None:
                    cap = async_cap
                else:
                    print(f"[警告] 相机异步线程启动失败，继续使用同步模式: {camera_index}")
            return cap, True
        else:
            print(f"[警告] 无法打开摄像头索引 {camera_index}")
            return None, False
    except Exception as e:
        print(f"[错误] 初始化摄像头失败: {e}")
        return None, False


def get_horizontal_force(forces):
    fx = safe_float(forces[0], default=0.0, abs_limit=FORCE_ABS_MAX_N)
    fy = safe_float(forces[1], default=0.0, abs_limit=FORCE_ABS_MAX_N)
    return math.sqrt(fx * fx + fy * fy)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_float(value, default=0.0, abs_limit=None):
    """健壮浮点转换：过滤 TypeError/ValueError/OverflowError/NaN/Inf/越界值。"""
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        print(f"[安全转换] float({repr(value)}) 失败: {type(e).__name__}: {e}", flush=True)
        return default
    except Exception as e:
        # 捕捉可能的其他异常，如 "int too large to convert to float"
        print(f"[安全转换] float({repr(value)}) 异常: {type(e).__name__}: {e}", flush=True)
        print(f"  值类型: {type(value)}, 值大小: {value}", flush=True)
        return default
    if not math.isfinite(f):
        print(f"[安全转换] float({value}) 结果非有限数 (NaN/Inf)", flush=True)
        return default
    if abs_limit is not None and abs(f) > abs_limit:
        print(f"[安全转换] float({f}) 超出限制 (abs_limit={abs_limit})", flush=True)
        return default
    return f


def sanitize_force_triplet(values):
    """将输入清洗为长度 3 的力向量；异常值返回 None。"""
    if values is None:
        return None
    try:
        if len(values) < 3:
            print(f"[清洗] 力向量长度不足: {len(values)} < 3, 值={values}", flush=True)
            return None
    except TypeError:
        print(f"[清洗] values 不支持 len(): {type(values)}, 值={values}", flush=True)
        return None
    
    fx = safe_float(values[0], default=None, abs_limit=FORCE_ABS_MAX_N)
    fy = safe_float(values[1], default=None, abs_limit=FORCE_ABS_MAX_N)
    fz = safe_float(values[2], default=None, abs_limit=FORCE_ABS_MAX_N)
    
    if fx is None or fy is None or fz is None:
        print(f"[清洗] 力值清洗失败: fx={fx}, fy={fy}, fz={fz}\n  原始values={values}", flush=True)
        return None
    
    try:
        result = np.array([fx, fy, fz], dtype=np.float64)
        return result
    except Exception as e:
        print(f"[清洗] numpy数组创建失败: {e}\n  fx={fx}, fy={fy}, fz={fz}", flush=True)
        return None


def append_log(row, csv_path):
    """
    修改：表头增加了 'id' 字段；添加详细错误跟踪
    """
    row_to_write = list(row)
    if len(row_to_write) == 13:
        row_to_write.append(FORCE_AXES_LABEL)

    try:
        # 检查行数据完整性
        if not isinstance(row, (list, tuple)):
            print(f"[日志错误] row 不是 list/tuple: {type(row)}", flush=True)
            return
        
        if len(row) != 13:
            print(f"[日志错误] row 长度不对: {len(row)} (期望13)\nrow={row}", flush=True)
        
        # 逐个检查每个元素，确保都是有效数字
        for i, val in enumerate(row):
            try:
                if isinstance(val, (int, float)):
                    # 检查是否是太大的整数
                    if isinstance(val, int) and abs(val) > 1e15:
                        print(f"[日志错误] row[{i}] 是超大整数: {val}", flush=True)
                    _ = float(val)  # 尝试转换以捕捉错误
                elif isinstance(val, str):
                    try:
                        float(val)
                    except:
                        pass  # 字符串在CSV中合法
                # 其他类型直接转换
                else:
                    _ = float(val) if val is not None else 0.0
            except OverflowError as e:
                print(f"[日志错误] row[{i}] 转换溢出: {val} - {e}", flush=True)
                return
            except ValueError as e:
                print(f"[日志错误] row[{i}] 值错误: {val} - {e}", flush=True)
                return
            except Exception as e:
                print(f"[日志错误] row[{i}] 未知错误: {val} ({type(val)}): {e}", flush=True)
                return
        
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    'id', 'timestamp', 'x', 'y', 'z',
                    'fx', 'fy', 'fz', 'fh',
                    'phase', 'target', 'angle_deg', 'point_index', 'force_axes'
                ])
            w.writerow(row_to_write)
    except Exception as e:
        print(f"[日志错误] append_log失败: {type(e).__name__}: {e}", flush=True)
        print(f"  row={row}", flush=True)


# ============== 坐标变换：传感器系 A -> 机床系 B ==============

def transform_force_A_to_B(F_A: np.ndarray) -> np.ndarray:
    """
    将力从传感器坐标系 A 变换到机床坐标系 B。
    变换步骤：
      1. 绕 Z 轴正方向逆时针旋转 30 度；
      2. 再绕 Y 轴正方向旋转 180 度。
    输入 F_A/输出 F_B 均为形如 [Fx, Fy, Fz] 的 1D numpy 数组（单位 N）。
    """
    F_A = np.asarray(F_A, dtype=float).reshape(3, )

    theta_z = math.radians(30.0)  # 绕 Z 逆时针 30°
    theta_y = math.radians(180.0)  # 绕 Y 正向 180°

    cz = math.cos(theta_z)
    sz = math.sin(theta_z)
    cy = math.cos(theta_y)
    sy = math.sin(theta_y)

    Rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    Ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy]
    ], dtype=float)

    R = Ry @ Rz
    F_B = R @ F_A
    return F_B


# ============== 传感器初始化与读取 ==============

def init_force_sensor(ft_sensor):
    try:
        if hasattr(ft_sensor, 'setUnits'):
            ft_sensor.setUnits('N')
    except Exception:
        pass
    try:
        if hasattr(ft_sensor, 'tare'):
            ft_sensor.tare(200)  # 平均200样本
    except Exception:
        pass
    calibrate_force_bias(ft_sensor, samples=300)


def calibrate_force_bias(ft_sensor, samples=300):
    global FORCE_BIAS
    buf = []
    for _ in range(samples):
        try:
            v = ft_sensor.getMeasurement()
            raw = sanitize_force_triplet(v)
            if raw is None:
                continue
            buf.append(raw.tolist())
        except Exception:
            pass
        time.sleep(0.005)
    if buf:
        FORCE_BIAS = np.mean(np.array(buf, dtype=float), axis=0)


def read_forces(ft_sensor):
    """
    从传感器读取原始力值，做去偏置与缩放，再转成视触觉传感器所受的力。
    返回 [Fx, Fy, Fz]（单位 N，已缩放）。
    """
    try:
        v = ft_sensor.getMeasurement()
        # 调试：记录原始值
        if v is not None:
            try:
                v_types = [type(x).__name__ for x in v]
                v_str = f"[{v_types[0]}={v[0]}, {v_types[1]}={v[1]}, {v_types[2]}={v[2]}]" if len(v) >= 3 else str(v)
            except:
                v_str = str(v)
            # print(f"[读力] 原始读取: {v_str}", flush=True)  # 注释以避免过多输出
        
        raw_A = sanitize_force_triplet(v)
        if raw_A is None:
            return [0.0, 0.0, 0.0]
        
        # 调试：记录清洁后的值
        # print(f"[读力] 清洁后: {raw_A.tolist()}", flush=True)
        
        corrected_A = (raw_A - FORCE_BIAS) * FORCE_SCALE
        result = (-corrected_A).tolist()
        # 调试：记录最终结果
        # print(f"[读力] 最终输出: {result}", flush=True)
        return result
    except RuntimeError as e:
        # 传感器断开连接，抛出异常供上层处理
        if "Sensor disconnected" in str(e):
            raise e
        return [0.0, 0.0, 0.0]
    except (TypeError, OverflowError) as e:
        print(f"[读力] 处理失败: {type(e).__name__}: {e}\\n  原始值={v}", flush=True)
        return [0.0, 0.0, 0.0]
    except Exception as e:
        print(f"[读力] 未同预期的异常: {type(e).__name__}: {e}\\n  原始值={v}", flush=True)
        return [0.0, 0.0, 0.0]


# ============== CNC 控制类（含双 Z 同步检测） ==============

class CNCController:
    AXIS_ORDER = ['X', 'Y', 'Z', 'A', 'B', 'C']

    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate, timeout=0.2)
        self._io_lock = threading.Lock()
        print(f"连接 {port}")
        time.sleep(1)
        # 软复位
        self.ser.write(b'\x18')
        time.sleep(2)
        self.ser.reset_input_buffer()
        # 基础设置：统一用 MPos，清掉 G92/G54 偏置
        self._send_with_ok('$X')  # 解锁
        self._send_with_ok('G21')  # mm
        self._send_with_ok('G90')  # 绝对
        self._send_with_ok('G54')  # 选 G54
        self._send_with_ok('G92.1')  # 清 G92
        self._send_with_ok('G10 L2 P1 X0 Y0 Z0')  # 清 G54 偏置

    def wait_for_idle(self):
        """阻塞直到 CNC 状态变为 Idle，确保物理运动已完成"""
        while True:
            _, _, _, state = self.get_status()
            if state == 'Idle':
                return
            # 如果是 Run (运动中) 或其他状态，则等待
            time.sleep(0.1)

    def _send_with_ok(self, cmd, timeout=60):
        with self._io_lock:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + '\n').encode())
            start = time.time()
            while time.time() - start < timeout:
                line = self.ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue
                if 'ok' in line:
                    return
                if 'error' in line or 'ALARM' in line:
                    raise RuntimeError(f"{cmd} -> {line}")
        raise TimeoutError(f"{cmd} 等待 ok 超时")

    def _read_status_line(self):
        with self._io_lock:
            self.ser.write(b'?\n')
            start = time.time()
            while time.time() - start < 0.5:
                line = self.ser.readline().decode(errors='ignore').strip()
                if line.startswith('<') and 'MPos:' in line:
                    return line
        raise TimeoutError("状态查询超时")

    def get_status(self):
        """
        统一只返回 MPos。
        返回: (pos_list[X,Y,Z,...], pin_status, 'MPos', machine_state)
        """
        try:
            line = self._read_status_line()
            mpos_match = re.search(r'\|MPos:([-\d.,]+)', line)
            if mpos_match:
                raw_vals = mpos_match.group(1).split(',')
                pos_list = [safe_float(v, default=0.0, abs_limit=CNC_POS_ABS_MAX_MM) for v in raw_vals]
            else:
                pos_list = [0.0] * 3
            pins = re.search(r'\|Pn:([A-Z]+)', line)

            # 3. 【新增】提取机器状态 (State)
            # 状态位于行首 '<' 之后，第一个 '|' 之前
            state_match = re.search(r'^<([a-zA-Z]+)', line)
            machine_state = state_match.group(1) if state_match else 'Unknown'

            return pos_list, (pins.group(1) if pins else '无'), 'MPos', machine_state
        except Exception:
            return [0.0, 0.0, 0.0], '未知', 'MPos', 'Unknown'

    def get_axis_map(self):
        pos_list, _, _, _ = self.get_status()
        axis_map = {}
        for i, val in enumerate(pos_list):
            if i < len(self.AXIS_ORDER):
                axis_map[self.AXIS_ORDER[i]] = val
        return axis_map

    def get_position(self):
        pos_list, _, kind, _ = self.get_status()
        return pos_list, kind

    def homing_cycle(self):
        """
        保留回零能力，用于偶尔校准。采集流程中不强制调用。
        """
        self._send_with_ok('$H', timeout=180)
        if DUAL_Z_ENABLED and SECOND_Z_AXIS and SECOND_Z_AXIS in self.AXIS_ORDER:
            try:
                self._dual_z_test()
            except Exception:
                self.emergency_stop()
                raise

    def set_work_origin(self):
        """
        若你确认只用 MPos，可以不再调用此接口。
        仍保留，按 O 键时若需要可用，但会引入 G92/G54 偏置。
        """
        self._send_with_ok('G92 X0 Y0 Z0')
        self._send_with_ok('G10 L20 P1 X0 Y0 Z0')

    def jog(self, axis, distance, speed):
        self._send_with_ok('G91')
        self._send_with_ok(f"G0 {axis}{distance:.4f} F{speed}")
        self._send_with_ok('G90')

    def move_to(self, x=None, y=None, z=None, speed=None):
        before = self.get_axis_map()
        cmd = "G1"
        if x is not None:
            cmd += f" X{x:.4f}"
        if y is not None:
            cmd += f" Y{y:.4f}"
        if z is not None:
            cmd += f" Z{z:.4f}"
        if speed is not None:
            cmd += f" F{speed}"
        self._send_with_ok(cmd)
        if z is not None:
            self._monitor_dual_z_sync(before)

    def rapid_move_to(self, x=None, y=None, z=None):
        before = self.get_axis_map()
        cmd = "G0"
        if x is not None:
            cmd += f" X{x:.4f}"
        if y is not None:
            cmd += f" Y{y:.4f}"
        if z is not None:
            cmd += f" Z{z:.4f}"
        self._send_with_ok(cmd)
        if z is not None:
            self._monitor_dual_z_sync(before)

    def feed_hold(self):
        with self._io_lock:
            self.ser.write(b'!')
        print("Feed Hold")

    def emergency_stop(self):
        with self._io_lock:
            self.ser.write(b'\x18')
        print("EMERGENCY STOP")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭")

    def _dual_z_test(self):
        if not (DUAL_Z_ENABLED and SECOND_Z_AXIS):
            return
        axis_map_before = self.get_axis_map()
        if 'Z' not in axis_map_before or SECOND_Z_AXIS not in axis_map_before:
            return
        self._send_with_ok('G91')
        self._send_with_ok('G1 Z-2 F300')
        self._send_with_ok('G1 Z2 F300')
        self._send_with_ok('G90')
        axis_map_after = self.get_axis_map()
        dz_main = abs(axis_map_after.get('Z', 0.0) - axis_map_before.get('Z', 0.0))
        dz_aux = abs(axis_map_after.get(SECOND_Z_AXIS, 0.0) - axis_map_before.get(SECOND_Z_AXIS, 0.0))
        if (dz_main > Z_SYNC_MIN_DELTA and dz_aux < Z_SYNC_MIN_DELTA) or \
                (dz_aux > Z_SYNC_MIN_DELTA and dz_main < Z_SYNC_MIN_DELTA):
            raise RuntimeError("双 Z 回零后测试不同步，停止。")

    def _monitor_dual_z_sync(self, axis_map_before):
        if not (DUAL_Z_ENABLED and SECOND_Z_AXIS):
            return
        if 'Z' not in axis_map_before or SECOND_Z_AXIS not in axis_map_before:
            return
        samples_main = []
        samples_aux = []
        start = time.time()
        while time.time() - start < Z_SYNC_CHECK_DURATION:
            axis_map_now = self.get_axis_map()
            samples_main.append(axis_map_now.get('Z', axis_map_before.get('Z', 0.0)))
            samples_aux.append(axis_map_now.get(SECOND_Z_AXIS, axis_map_before.get(SECOND_Z_AXIS, 0.0)))
            time.sleep(Z_SYNC_CHECK_INTERVAL)
        dz_main = abs(samples_main[-1] - samples_main[0])
        dz_aux = abs(samples_aux[-1] - samples_aux[0])
        if (dz_main > Z_SYNC_MIN_DELTA and dz_aux < Z_SYNC_MIN_DELTA) or \
                (dz_aux > Z_SYNC_MIN_DELTA and dz_main < Z_SYNC_MIN_DELTA):
            self.emergency_stop()
            raise RuntimeError(
                f"检测到 Z 与 {SECOND_Z_AXIS} 位移不同步(主:{dz_main:.3f} 副:{dz_aux:.3f})，程序终止。"
            )


def safe_get_position(cnc: CNCController):
    """
    安全包装：永远返回 (pos_list, kind)，即使出现异常也用默认值。
    防止 get_position 异常导致 None 被解包。
    """
    try:
        pos_list, kind = cnc.get_position()
        if not isinstance(pos_list, (list, tuple)) or len(pos_list) < 3:
            raise ValueError("invalid pos_list")
        return pos_list, kind
    except Exception:
        return [0.0, 0.0, 0.0], "未知"


def print_current_machine_state(cnc: CNCController, ft_sensor=None, force_reader=None):
    """打印当前机床坐标；若可读力传感器则一并打印力信息。"""
    pos_list, kind = safe_get_position(cnc)
    message = f"{kind}: X={pos_list[0]:.3f} Y={pos_list[1]:.3f} Z={pos_list[2]:.3f}"

    if ft_sensor is None:
        print(f"\n{message}")
        return

    reader = force_reader or read_forces
    try:
        forces_corr_B = reader(ft_sensor)
        fx = float(forces_corr_B[0])
        fy = float(forces_corr_B[1])
        fz = float(forces_corr_B[2])
        fh = get_horizontal_force(forces_corr_B)
        print(f"\n{message} | Fx={fx:.3f} Fy={fy:.3f} Fz={fz:.3f} Fh={fh:.3f}")
    except Exception as e:
        print(f"\n{message} | 力读取失败: {e}")


# ============== 图像叠加显示 ==============

def show_live(cap, cnc: CNCController, forces, window_name=WINDOW_NAME):
    """
    forces 参数为机床坐标系 B 下的 [Fx, Fy, Fz]。
    """
    if cap is None:
        return

    try:
        ret, frame = cap.read()
        if not ret:
            return
    except Exception as e:
        # 摄像头读取错误时不中断采集，仅打印警告
        print(f"[相机警告] 读取失败: {e}", flush=True)
        return

    try:
        pos_list, kind = safe_get_position(cnc)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        cv2.putText(
            frame,
            f"{kind}: X={pos_list[0]:.3f} Y={pos_list[1]:.3f} Z={pos_list[2]:.3f}",
            (10, 28), font, 0.55, color, 2, cv2.LINE_AA
        )

        fx, fy, fz = 0.0, 0.0, 0.0
        if forces is not None:
            try:
                raw = np.array([float(forces[0]), float(forces[1]), float(forces[2])], dtype=float)
                fx, fy, fz = raw.tolist()
            except (TypeError, OverflowError) as e:
                print(f"[显示] 力值转换失败: {e}\n  forces={forces}", flush=True)
            except Exception as e:
                print(f"[显示] 力值处理异常: {type(e).__name__}: {e}\n  forces={forces}\n  类型: {[type(f) for f in forces] if forces else None}", flush=True)

        cv2.putText(frame, f"Fx={fx:.2f} Fy={fy:.2f} Fz={fz:.2f}", (10, 56), font, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Fh={math.sqrt(fx * fx + fy * fy):.2f}", (10, 84), font, 0.55, color, 2, cv2.LINE_AA)

        if DUAL_Z_ENABLED and SECOND_Z_AXIS:
            axis_map = cnc.get_axis_map()
            if SECOND_Z_AXIS in axis_map:
                cv2.putText(
                    frame,
                    f"{SECOND_Z_AXIS}={axis_map[SECOND_Z_AXIS]:.3f}",
                    (10, 112), font, 0.55, (0, 200, 255), 1, cv2.LINE_AA
                )

        cv2.imshow(window_name, frame)
    except Exception as e:
        print(f"[相机警告] 显示失败: {e}", flush=True)


# ============== 手动键盘线程 ==============

def print_instructions(mode, inc):
    print("\n--- 手动控制 ---")
    print("W/S: Y+ / Y-  A/D: X+ / X-  Q/E: Z+ / Z-")
    print(f"J: 切换步长 (当前: {mode}, {inc} mm)")
    print("H: 回零(含双Z检测)  O: 传感器去皮  P: 打印当前坐标  C: 开始采集  ESC: 退出")
    print("----------------")


def keyboard_loop(cnc: CNCController, stop_evt: threading.Event, collect_evt: threading.Event, ft_sensor=None):
    MOVE_SPEED = 500
    LARGE_MOVE_INCREMENT = 1.0
    SMALL_MOVE_INCREMENT = 0.1
    large = True

    def current_inc():
        return LARGE_MOVE_INCREMENT if large else SMALL_MOVE_INCREMENT

    print_instructions("大步长", current_inc())

    key_actions = {
        'w': lambda inc: cnc.jog('Y', +inc, MOVE_SPEED),
        's': lambda inc: cnc.jog('Y', -inc, MOVE_SPEED),
        'd': lambda inc: cnc.jog('X', +inc, MOVE_SPEED),
        'a': lambda inc: cnc.jog('X', -inc, MOVE_SPEED),
        'q': lambda inc: cnc.jog('Z', +inc, MOVE_SPEED),
        'e': lambda inc: cnc.jog('Z', -inc, MOVE_SPEED),
        # 'p' 改为在主线程中按需打印，这里先占位
        # 'o' 原为设置机械原点，现改为传感器去皮
        'h': lambda: cnc.homing_cycle(),
        'c': lambda: collect_evt.set(),
    }

    while not stop_evt.is_set() and not collect_evt.is_set():
        try:
            k = KEY_READER.poll_key(timeout=0.05)
            if k is None:
                continue
            inc = current_inc()
            if k in key_actions:
                if k in ['w', 's', 'd', 'a', 'q', 'e']:
                    key_actions[k](inc)
                else:
                    key_actions[k]()
            elif k == 'o':
                 # 按 'o' 对传感器去皮
                 if ft_sensor:
                     print("\n[操作] 正在对力传感器去皮 (Tare)...")
                     try:
                         # 停止旧的 bias 计算，重新计算
                         calibrate_force_bias(ft_sensor, samples=200)
                         print("       去皮完成。当前读数归零。")
                     except Exception as e:
                         print(f"       去皮失败: {e}")
                 else:
                     print("\n[警告] 传感器未连接，无法去皮。")

            elif k == 'j':
                large = not large
                mode = "大步长" if large else "小步长"
                print(f"\n切换到 -> {mode} ({current_inc()}mm)")
            elif k == 'p':
                print_current_machine_state(cnc, ft_sensor)
            elif k == 'esc':
                stop_evt.set()
                break
        except Exception:
            time.sleep(0.02)


# ============== 主流程 ==============

def main():
    global STOP_REQUESTED
    STOP_REQUESTED = False
    KEY_READER.enable()
    
    socket.setdefaulttimeout(2.0)

    def request_stop_and_lift(cnc_obj):
        """检测到 ESC 后设置停止并尝试安全抬刀。"""
        global STOP_REQUESTED
        if STOP_REQUESTED:
            return
        print("\n检测到 ESC，准备停止采集并抬高 Z 轴...")
        STOP_REQUESTED = True
        if cnc_obj is None:
            return
        try:
            # 首先发送紧急停止（CTRL-X），立即停止所有运动
            cnc_obj.emergency_stop()
            time.sleep(0.5)  # 等待机器停止
            
            # 然后抬起 Z 轴到安全高度
            pos, _ = cnc_obj.get_position()
            current_z = float(pos[2])
            target_z = min(current_z + 20.0, float(Z_SAFE_HEIGHT))
            print(f"  [ESC] 抬起 Z: {current_z:.3f} -> {target_z:.3f}")
            cnc_obj.rapid_move_to(z=target_z)
        except Exception as e:
            print(f"  [ESC] 无法抬起 Z 轴: {e}")

    def poll_escape(cnc_obj):
        k = KEY_READER.poll_key(timeout=0.0)
        if k == 'esc':
            request_stop_and_lift(cnc_obj)
            return True
        if k == 'p':
            print_current_machine_state(cnc_obj, ft_sensor)
        return False

    cnc = None
    ft_sensor = None
    cap = None
    camera_available = False
    
    try:
        # 设备初始化
        cnc = CNCController(SERIAL_PORT, BAUD_RATE)
        
        # 使用 KWR36 传感器
        ft_sensor = KWR36Sensor(SENSOR_IP, SENSOR_PORT)
        ft_sensor.start()
        init_force_sensor(ft_sensor)

        # 跨平台摄像头初始化（支持设备路径和索引双重指定）
        cap, camera_open_success = init_camera(
            camera_index=CAMERA_INDEX,
            camera_device=CAMERA_DEVICE,
            auto_detect=AUTO_DETECT_CAMERA,
        )
        
        if not camera_open_success or cap is None:
            print("警告: 无法打开相机，进入无相机模式。")
            cap = None
            camera_available = False
        else:
            camera_available = True
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # 尝试读取一帧来确认相机是否真正工作
            ret_test, _ = cap.read()
            if not ret_test:
                 print("警告: 无法读取相机帧，进入无相机模式。")
                 cap.release()
                 cap = None
                 camera_available = False
                 cv2.destroyAllWindows()

        # 为本次采集创建带时间戳的会话目录
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(DATA_DIR, session_name)
        ensure_dir(session_dir)

        # 新增：创建 images 子文件夹用于存放序列图片
        img_dir = os.path.join(session_dir, "images")
        ensure_dir(img_dir)

        csv_path = os.path.join(session_dir, "force_log.csv")
        skip_log_path = os.path.join(session_dir, "skip_log.txt")

        # 跳过/失败日志记录函数
        def log_skip_reason(reason_text):
            """记录跳过或失败的原因到skip_log.txt"""
            try:
                with open(skip_log_path, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().isoformat()
                    f.write(f"[{timestamp}] {reason_text}\n")
            except Exception as e:
                print(f"[警告] 无法写入跳过日志: {e}")

        def save_rgb_snapshot(image_id, context_text):
            if not camera_available or cap is None:
                return False
            try:
                ret_img, frame = cap.read()
                if not ret_img:
                    return False
                img_name = f"{image_id}_rgb.png"
                cv2.imwrite(os.path.join(img_dir, img_name), frame)
                return True
            except Exception as e:
                print(f"[相机] {context_text}图像保存失败: {e}", flush=True)
                return False

        data_id = 1

        def record_sample(phase, target, angle_deg, point_index, forces_corr_B, fh, image_context):
            nonlocal data_id
            try:
                pos, _ = cnc.get_position()
            except Exception:
                pos = [0, 0, 0]

            save_rgb_snapshot(data_id, image_context)

            append_log([
                data_id,
                datetime.now().isoformat(),
                pos[0], pos[1], pos[2],
                forces_corr_B[0], forces_corr_B[1], forces_corr_B[2],
                fh,
                phase, target, angle_deg, point_index
            ], csv_path)

            data_id += 1

        def run_horizontal_collection(
            point_index,
            x,
            y,
            current_z,
            horizontal_fz_target,
            horizontal_forces,
            initial_message,
            reprobe_message,
            skip_initial_reached,
        ):
            if len(horizontal_forces) == 0:
                print("  [水平采集] 当前Fh阶梯为空，跳过。")
                return current_z

            print(initial_message)

            for angle_deg in angles:
                if STOP_REQUESTED:
                    break
                if poll_escape(cnc):
                    break

                print(f"    - 方向: {angle_deg}°")
                angle_rad = math.radians(angle_deg)
                curr_x, curr_y = x, y
                fail_this_direction = False

                for fh_target in horizontal_forces:
                    if STOP_REQUESTED or fail_this_direction:
                        break

                    fz_lower, fz_upper, _ = get_force_window(horizontal_fz_target, FZ_REL_TOL, FZ_ABS_TOL)
                    fh_lower, fh_upper, _ = get_force_window(fh_target, FH_REL_TOL, FH_ABS_TOL)
                    fh_recheck_retries = 0
                    step_count = 0
                    reached = False

                    while step_count < MAX_STEPS_PER_DIRECTION:
                        if STOP_REQUESTED:
                            break
                        if poll_escape(cnc):
                            break

                        forces_corr_B = read_forces(ft_sensor)
                        curr_fz = float(forces_corr_B[2])
                        fz_comp_retries = 0

                        if curr_fz < fz_lower:
                            z_limit = current_z - 1.0
                            while curr_fz < fz_lower:
                                if STOP_REQUESTED:
                                    break
                                if poll_escape(cnc):
                                    break
                                if current_z <= z_limit:
                                    print("    [Warn] Z compensation limit reached!")
                                    break

                                fz_comp_retries += 1
                                adaptive_z_step = get_adaptive_z_step(Z_STEP, fz_comp_retries)
                                current_z -= adaptive_z_step
                                cnc.move_to(z=current_z, speed=Z_PROBE_SPEED)
                                time.sleep(VERTICAL_Z_SETTLE_SEC)

                                forces_corr_B = read_forces(ft_sensor)
                                curr_fz = float(forces_corr_B[2])
                                if curr_fz > fz_upper:
                                    break
                        elif curr_fz > fz_upper:
                            z_limit_up = min(Z_SAFE_HEIGHT, current_z + 1.0)
                            while curr_fz > fz_upper:
                                if STOP_REQUESTED:
                                    break
                                if poll_escape(cnc):
                                    break
                                if current_z >= z_limit_up:
                                    print("    [Warn] Z upward compensation limit reached!")
                                    break

                                fz_comp_retries += 1
                                adaptive_z_step = get_adaptive_z_step(Z_STEP, fz_comp_retries)
                                current_z += adaptive_z_step
                                cnc.move_to(z=current_z, speed=Z_PROBE_SPEED)
                                time.sleep(VERTICAL_Z_SETTLE_SEC)

                                forces_corr_B = read_forces(ft_sensor)
                                curr_fz = float(forces_corr_B[2])

                        fh = get_horizontal_force(forces_corr_B)
                        if curr_fz > max(horizontal_fz_target * 1.5, 1.0):
                            print(f"    [Safety] Fz {curr_fz:.2f}N too high (Target {horizontal_fz_target:.2f}N)! Emergency lift.")
                            current_z += 1.0
                            if current_z > Z_SAFE_HEIGHT:
                                current_z = Z_SAFE_HEIGHT
                            cnc.move_to(z=current_z, speed=RAPID_SPEED)
                            fail_this_direction = True
                            continue

                        if fh >= fh_target:
                            if step_count == 0:
                                if skip_initial_reached:
                                    print(
                                        f"    [Skip] Initial Fh {fh:.2f} already reaches target {fh_target:.2f} "
                                        f"before any horizontal move; skip this target because it is effectively a normal-force sample."
                                    )
                                    reached = True
                                    break

                                print(
                                    f"    [Save] Initial Fh {fh:.2f} already reaches target {fh_target:.2f} "
                                    f"before any horizontal move; save this sample as a horizontal-force sample."
                                )
                                record_sample('horizontal', fh_target, angle_deg, point_index, forces_corr_B, fh, "水平采集(初始达标)")
                                reached = True
                                break

                            print(f"    Fh {fh:.2f} >= {fh_target:.2f}, Stabilizing...")
                            time.sleep(FORCE_STABILIZE_SEC)

                            forces_corr_B = read_forces(ft_sensor)
                            fh = get_horizontal_force(forces_corr_B)

                            if fh < fh_lower:
                                print(
                                    f"    [Recheck] Fh relaxed to {fh:.2f}, below "
                                    f"window lower bound {fh_lower:.2f}; continue stepping."
                                )
                            elif fh > fh_upper:
                                if fh_recheck_retries >= FH_RECHECK_MAX_RETRIES:
                                    print(
                                        f"    [Recheck] Fh stayed above upper bound after {FH_RECHECK_MAX_RETRIES} retries; "
                                        f"saving current sample as best effort."
                                    )
                                    record_sample('horizontal', fh_target, angle_deg, point_index, forces_corr_B, fh, "水平采集(超界)")
                                    reached = True
                                    break

                                fh_recheck_retries += 1
                                adaptive_xy_step = get_adaptive_xy_step(XY_STEP, fh_recheck_retries)
                                print(
                                    f"    [Recheck] Fh {fh:.2f} above upper bound {fh_upper:.2f}; "
                                    f"backing off by {adaptive_xy_step:.4f} mm and rechecking "
                                    f"(retry {fh_recheck_retries}/{FH_RECHECK_MAX_RETRIES})."
                                )

                                back_dx = adaptive_xy_step * math.cos(angle_rad)
                                back_dy = adaptive_xy_step * math.sin(angle_rad)
                                curr_x -= back_dx
                                curr_y -= back_dy
                                cnc.move_to(x=curr_x, y=curr_y, z=current_z, speed=XY_PROBE_SPEED)
                                time.sleep(FORCE_STABILIZE_SEC)

                                forces_corr_B = read_forces(ft_sensor)
                                fh = get_horizontal_force(forces_corr_B)
                                curr_fz = float(forces_corr_B[2])
                                if fh < fh_lower:
                                    print(
                                        f"    [Backoff Recheck] Fh dropped to {fh:.2f}, below lower bound {fh_lower:.2f}; "
                                        f"continue stepping with reduced step {adaptive_xy_step:.4f} mm."
                                    )
                                    continue
                                if curr_fz < fz_lower or curr_fz > fz_upper:
                                    print(
                                        f"    [Backoff Recheck] Fz moved to {curr_fz:.2f}, out of "
                                        f"[{fz_lower:.2f}, {fz_upper:.2f}]; continue closed-loop adjustment."
                                    )
                                    continue

                                if fh > fh_upper:
                                    print(
                                        f"    [Backoff Recheck] Fh still above upper bound at {fh:.2f} > {fh_upper:.2f}, "
                                        f"but accepted after adaptive backoff; saving sample."
                                    )
                                else:
                                    print("    [Backoff Recheck] Fh returned into window; saving current sample.")

                                record_sample('horizontal', fh_target, angle_deg, point_index, forces_corr_B, fh, "水平采集(回归)")
                                reached = True
                                break
                            else:
                                fh_recheck_retries = 0
                                record_sample('horizontal', fh_target, angle_deg, point_index, forces_corr_B, fh, "水平采集")
                                reached = True
                                break

                        current_xy_step = get_adaptive_xy_step(XY_STEP, fh_recheck_retries)
                        dx = current_xy_step * math.cos(angle_rad)
                        dy = current_xy_step * math.sin(angle_rad)
                        curr_x += dx
                        curr_y += dy
                        cnc.move_to(x=curr_x, y=curr_y, z=current_z, speed=XY_PROBE_SPEED)
                        step_count += 1

                    if not reached:
                        print(
                            f"    Cannot reach fh_target={fh_target:.2f} within {MAX_STEPS_PER_DIRECTION} steps, "
                            f"skip larger fh in this direction."
                        )
                        fail_this_direction = True

                print(f"    [Reset] Direction {angle_deg}° done. Resetting...")
                lift_z = Z_SAFE_HEIGHT
                cnc.move_to(z=lift_z, speed=Z_PROBE_SPEED)
                cnc.rapid_move_to(x=x, y=y)
                cnc.wait_for_idle()

                time.sleep(0.2)
                try:
                    ft_sensor.tare(50)
                except Exception:
                    pass

                is_last_angle = (angle_deg == angles[-1])
                if not is_last_angle:
                    print(reprobe_message)
                    approach_z = current_z + 0.5
                    if approach_z < lift_z:
                        cnc.move_to(z=approach_z, speed=Z_PROBE_SPEED)
                        current_z = approach_z
                    else:
                        current_z = lift_z

                    reset_fz_retries = 0
                    while True:
                        if STOP_REQUESTED:
                            break
                        if poll_escape(cnc):
                            break
                        forces_corr_B = read_forces(ft_sensor)
                        curr_fz_reset = float(forces_corr_B[2])
                        if curr_fz_reset >= horizontal_fz_target:
                            break

                        reset_fz_retries += 1
                        current_z -= get_adaptive_z_step(Z_STEP, reset_fz_retries)
                        cnc.move_to(z=current_z, speed=Z_PROBE_SPEED)
                        time.sleep(VERTICAL_Z_SETTLE_SEC)

                    time.sleep(0.5)

            return current_z

        # 手动控制阶段（可在此阶段偶尔按 H 做回零校准）
        stop_evt = threading.Event()
        collect_evt = threading.Event()
        # 传入 ft_sensor 以便在手动模式下打印力数据
        kb_thread = threading.Thread(
            target=keyboard_loop,
            args=(cnc, stop_evt, collect_evt, ft_sensor),
            daemon=True
        )
        kb_thread.start()

        print("\n进入手动模式（按 C 开始采集，ESC 退出）")
        if not camera_available:
            print("[信息] 无相机模式下，请关注终端输出。")

        while not stop_evt.is_set() and not collect_evt.is_set():
            try:
                forces_corr_B = read_forces(ft_sensor)
                if camera_available and cap is not None:
                    show_live(cap, cnc, forces_corr_B, WINDOW_NAME)
                    if cv2.waitKey(1) & 0xFF == 27:
                        stop_evt.set()
                        break
                else:
                    # 无相机时，降低刷新率，避免死循环占用过高
                    time.sleep(0.1)
            except Exception:
                time.sleep(0.01)

        kb_thread.join(timeout=2)
        if stop_evt.is_set():
            return

        # 在开始正式采集前，强制执行一次回零，确保 MPos 一致
        print("\n按下 C，开始回零...")
        cnc.homing_cycle()
        print("回零完成，开始采集准备。")

        # ================= 背景图采集 =================
        if camera_available and cap is not None:
            try:
                ret_bg, frame_bg = cap.read()
                if ret_bg:
                    bg_path = os.path.join(session_dir, "0_rgb.png")
                    cv2.imwrite(bg_path, frame_bg)
                    print(f"已保存背景图像: {bg_path}")
                else:
                    print("警告: 无法采集背景图像 (bg.png)")
            except Exception as e:
                print(f"警告: 背景图采集失败: {e}")
        else:
            print("警告: 无相机模式，跳过背景图像采集。")
        # =================================================

        # 采集阶段：不再强制回零，直接按预设 MPos 网格采集
        print("\n开始采集...")

        x_coords = np.linspace(BOTTOM_LEFT['x'], TOP_RIGHT['x'], GRID_SIZE)
        y_coords = np.linspace(BOTTOM_LEFT['y'], TOP_RIGHT['y'], GRID_SIZE)
             
        xy_grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

        vertical_forces = np.linspace(0, MAX_VERTICAL_FORCE, VERTICAL_FORCE_STEPS)[1:]
        angles = np.linspace(0, 360 - 360 / HORIZONTAL_DIRECTIONS, HORIZONTAL_DIRECTIONS)

        total_points = len(xy_grid)
        for i, (x, y) in enumerate(xy_grid):
            # ===== ESC 检查：每个网格点开始前 =====
            if poll_escape(cnc):
                break
            if STOP_REQUESTED:
                print("检测到 ESC，中止后续网格点采集。")
                break

            print(f"\n--- 网格 {i + 1}/{total_points}: X={x:.2f} Y={y:.2f} ---")
            cnc.rapid_move_to(z=Z_SAFE_HEIGHT)
            cnc.rapid_move_to(x=x, y=y)

            # 必须等待物理移动完全结束，悬空稳定后，才能去皮！
            print("  Waiting for machine to stop...", end='', flush=True)
            cnc.wait_for_idle()
            print(" Done.")

            # 等待一小段时间消除机械余震
            time.sleep(0.5)

            try:
                ft_sensor.tare(100)
            except Exception:
                pass
            time.sleep(0.5)

            current_z = Z_SAFE_HEIGHT

            # 垂直阶段：对每一级竖直力都进行水平力采集
            for idx, fz_target in enumerate(vertical_forces):
                if STOP_REQUESTED:
                    break

                fz_lower, fz_upper, fz_tol = get_force_window(fz_target, FZ_REL_TOL, FZ_ABS_TOL)

                # 1. 检查当前力值
                forces_corr_B = read_forces(ft_sensor)
                current_fz = float(forces_corr_B[2])

                # 2. 如果当前力值已经超过目标（或接近目标），先抬起
                if current_fz > fz_upper:
                    print(f"  [Adjust] Current Fz {current_fz:.2f} too high for target {fz_target:.2f}. Lifting...")
                    relief_z = current_z + 1.0  # 向上抬 1.0mm
                    if relief_z > Z_SAFE_HEIGHT: relief_z = Z_SAFE_HEIGHT

                    cnc.move_to(z=relief_z, speed=Z_PROBE_SPEED)
                    time.sleep(0.2)  # 等待稳定
                    current_z = relief_z  # 更新当前Z坐标记录

                vertical_locked = False
                vertical_best_effort_saved = False
                vertical_window_iters = 0
                fz_recheck_retries = 0
                vertical_recheck_started = False
                while True:
                    if poll_escape(cnc):
                        break
                    if STOP_REQUESTED:
                        break
                    if vertical_recheck_started and vertical_window_iters >= MAX_VERTICAL_FZ_WINDOW_ITERS:
                        forces_corr_B = read_forces(ft_sensor)
                        fz_best = float(forces_corr_B[2])
                        skip_reason = (
                            f"Grid ({x:.2f}, {y:.2f}): Fz {fz_target:.2f}N - could not settle into "
                            f"[{fz_lower:.2f}, {fz_upper:.2f}] within {MAX_VERTICAL_FZ_WINDOW_ITERS} "
                            f"recheck adjustments; saving best-effort sample at Fz={fz_best:.2f}N"
                        )
                        print(
                            f"  [Warn] {skip_reason}"
                        )
                        log_skip_reason(skip_reason)

                        fh = get_horizontal_force(forces_corr_B)
                        record_sample('vertical', fz_target, 0.0, i, forces_corr_B, fh, "垂直采集(best-effort)")
                        vertical_locked = True
                        vertical_best_effort_saved = True
                        break
                    
                    # ===== ESC 检查：即使相机出现问题也能响应 =====
                    if poll_escape(cnc):
                        break
                    
                    forces_corr_B = read_forces(ft_sensor)

                    fz = float(forces_corr_B[2])

                    if fz_lower <= fz <= fz_upper:
                        print(f"  Fz {fz:.2f} within [{fz_lower:.2f}, {fz_upper:.2f}], Stabilizing...")
                        time.sleep(FORCE_STABILIZE_SEC)

                        forces_corr_B = read_forces(ft_sensor)
                        fz_relaxed = float(forces_corr_B[2])
                        if fz_relaxed < fz_lower:
                            vertical_recheck_started = True
                            vertical_window_iters += 1
                            fz_recheck_retries += 1
                            print(
                                f"  [Recheck] Fz relaxed to {fz_relaxed:.2f}, below "
                                f"window lower bound {fz_lower:.2f}; continue probing down "
                                f"with adaptive step {get_adaptive_z_step(Z_STEP, fz_recheck_retries):.4f} mm."
                            )
                        elif fz_relaxed > fz_upper:
                            vertical_recheck_started = True
                            vertical_window_iters += 1
                            fz_recheck_retries += 1
                            print(
                                f"  [Recheck] Fz relaxed to {fz_relaxed:.2f}, above "
                                f"window upper bound {fz_upper:.2f}; easing up "
                                f"with adaptive step {get_adaptive_z_step(Z_STEP, fz_recheck_retries):.4f} mm."
                            )
                        else:
                            vertical_recheck_started = False
                            fz_recheck_retries = 0
                            print(f"  [Stable] Fz {fz_relaxed:.2f} stabilized near target {fz_target:.2f}.")
                            fh = get_horizontal_force(forces_corr_B)
                            record_sample('vertical', fz_target, 0.0, i, forces_corr_B, fh, "垂直采集")
                            vertical_locked = True
                            break
                    
                    # 未在窗口内，按需调整 Z
                    # 关键改进：第一阶段用固定步长逼近，只有进窗后掉出才启用自适应衰减
                    if fz < fz_lower:
                        # 下压：使用固定步长或自适应步长（取决于是否已进入过窗口）
                        if not vertical_recheck_started:
                            # 第一阶段：还未进入窗口过，用固定步长快速逼近
                            adaptive_z_step = Z_STEP
                        else:
                            # 第二阶段：已进入窗口并掉出，用自适应步长精细调整
                            fz_recheck_retries += 1
                            adaptive_z_step = get_adaptive_z_step(Z_STEP, fz_recheck_retries)
                            vertical_window_iters += 1
                        
                        current_z -= adaptive_z_step
                        cnc.move_to(z=current_z, speed=Z_PROBE_SPEED)
                    else:
                        # fz > fz_upper：抬起：使用固定步长或自适应步长（取决于是否已进入过窗口）
                        if not vertical_recheck_started:
                            # 第一阶段：还未进入窗口过，用固定步长快速逼近
                            adaptive_z_step = Z_STEP
                        else:
                            # 第二阶段：已进入窗口并掉出，用自适应步长精细调整
                            fz_recheck_retries += 1
                            adaptive_z_step = get_adaptive_z_step(Z_STEP, fz_recheck_retries)
                            vertical_window_iters += 1
                        
                        current_z += adaptive_z_step
                        if current_z > Z_SAFE_HEIGHT:
                            current_z = Z_SAFE_HEIGHT
                        cnc.move_to(z=current_z, speed=Z_PROBE_SPEED)

                if STOP_REQUESTED:
                    break
                if not vertical_locked:
                    skip_reason = f"Grid ({x:.2f}, {y:.2f}): Fz {fz_target:.2f}N - could not lock in window"
                    print(f"  [Skip] {skip_reason}")
                    log_skip_reason(skip_reason)
                    break
                if vertical_best_effort_saved:
                    next_stage = "下一个垂直力目标" if USE_FIXED_HORIZONTAL_FZ else "水平力采集"
                    print(f"  [BestEffort] Saved off-window vertical sample for target Fz={fz_target:.2f}N; continue to {next_stage}.")

                if STOP_REQUESTED:
                    break

                if not USE_FIXED_HORIZONTAL_FZ:
                    current_fz_for_horizontal = max(0.0, float(forces_corr_B[2]))
                    max_horizontal_force = HORIZONTAL_FORCE_RATIO * current_fz_for_horizontal
                    raw_horizontal_forces = np.linspace(0, max_horizontal_force, HORIZONTAL_FORCE_STEPS)[1:]
                    if raw_horizontal_forces.size > 0:
                        horizontal_forces = np.maximum(raw_horizontal_forces, MIN_HORIZONTAL_FORCE_TARGET)
                        horizontal_forces = np.unique(np.round(horizontal_forces, decimals=6))
                    else:
                        horizontal_forces = raw_horizontal_forces

                    current_z = run_horizontal_collection(
                        point_index=i,
                        x=x,
                        y=y,
                        current_z=current_z,
                        horizontal_fz_target=fz_target,
                        horizontal_forces=horizontal_forces,
                        initial_message=(
                            f"  [水平采集] 当前Fz={current_fz_for_horizontal:.2f}N, "
                            f"Max_Fh={HORIZONTAL_FORCE_RATIO:.2f}*Fz={max_horizontal_force:.2f}N, "
                            f"Fh阶梯: {[f'{fh:.2f}' for fh in horizontal_forces]}"
                        ),
                        reprobe_message=f"    [Reset] Re-probing Z to {fz_target:.2f}N for next angle...",
                        skip_initial_reached=True,
                    )

            if STOP_REQUESTED:
                break

            if USE_FIXED_HORIZONTAL_FZ:
                print("  [Stage] 法向力采集完成，现在开始固定 Fz 水平力采集。")

                if HORIZONTAL_DIRECTIONS <= 0 or HORIZONTAL_FORCE_STEPS <= 1:
                    print("  [水平采集] 当前配置未启用水平采集，跳过。")
                    continue

                horizontal_fz_target = FIXED_HORIZONTAL_FZ_TARGET
                horizontal_fz_lower, horizontal_fz_upper, _ = get_force_window(horizontal_fz_target, FZ_REL_TOL, FZ_ABS_TOL)
                print(f"  [水平采集] 固定预载 Fz={horizontal_fz_target:.2f}N")

                forces_corr_B = read_forces(ft_sensor)
                current_fz = float(forces_corr_B[2])
                if current_fz > horizontal_fz_upper:
                    print(
                        f"  [Adjust] Current Fz {current_fz:.2f} too high for fixed horizontal target "
                        f"{horizontal_fz_target:.2f}. Lifting..."
                    )
                    relief_z = current_z + 1.0
                    if relief_z > Z_SAFE_HEIGHT:
                        relief_z = Z_SAFE_HEIGHT
                    cnc.move_to(z=relief_z, speed=Z_PROBE_SPEED)
                    time.sleep(0.2)
                    current_z = relief_z

                horizontal_preload_locked = False
                horizontal_window_iters = 0
                horizontal_recheck_started = False
                horizontal_fz_recheck_retries = 0
                while True:
                    if poll_escape(cnc):
                        break
                    if STOP_REQUESTED:
                        break
                    if horizontal_recheck_started and horizontal_window_iters >= MAX_VERTICAL_FZ_WINDOW_ITERS:
                        fz_best = float(read_forces(ft_sensor)[2])
                        skip_reason = (
                            f"Grid ({x:.2f}, {y:.2f}): fixed horizontal Fz {horizontal_fz_target:.2f}N - "
                            f"could not settle into [{horizontal_fz_lower:.2f}, {horizontal_fz_upper:.2f}] "
                            f"within {MAX_VERTICAL_FZ_WINDOW_ITERS} recheck adjustments"
                        )
                        print(f"  [Warn] {skip_reason}; continue with best-effort Fz={fz_best:.2f}N")
                        log_skip_reason(skip_reason)
                        horizontal_preload_locked = True
                        break

                    forces_corr_B = read_forces(ft_sensor)
                    fz = float(forces_corr_B[2])
                    if horizontal_fz_lower <= fz <= horizontal_fz_upper:
                        print(f"  Fz {fz:.2f} within [{horizontal_fz_lower:.2f}, {horizontal_fz_upper:.2f}], Stabilizing...")
                        time.sleep(FORCE_STABILIZE_SEC)

                        forces_corr_B = read_forces(ft_sensor)
                        fz_relaxed = float(forces_corr_B[2])
                        if fz_relaxed < horizontal_fz_lower:
                            horizontal_recheck_started = True
                            horizontal_window_iters += 1
                            horizontal_fz_recheck_retries += 1
                            print(
                                f"  [Recheck] Fz relaxed to {fz_relaxed:.2f}, below "
                                f"window lower bound {horizontal_fz_lower:.2f}; continue probing down."
                            )
                        elif fz_relaxed > horizontal_fz_upper:
                            horizontal_recheck_started = True
                            horizontal_window_iters += 1
                            horizontal_fz_recheck_retries += 1
                            print(
                                f"  [Recheck] Fz relaxed to {fz_relaxed:.2f}, above "
                                f"window upper bound {horizontal_fz_upper:.2f}; easing up."
                            )
                        else:
                            horizontal_preload_locked = True
                            break

                    if fz < horizontal_fz_lower:
                        if horizontal_recheck_started:
                            horizontal_fz_recheck_retries += 1
                            horizontal_window_iters += 1
                        adaptive_z_step = get_adaptive_z_step(Z_STEP, horizontal_fz_recheck_retries)
                        current_z -= adaptive_z_step
                        cnc.move_to(z=current_z, speed=Z_PROBE_SPEED)
                        time.sleep(VERTICAL_Z_SETTLE_SEC)
                    else:
                        if horizontal_recheck_started:
                            horizontal_fz_recheck_retries += 1
                            horizontal_window_iters += 1
                        adaptive_z_step = get_adaptive_z_step(Z_STEP, horizontal_fz_recheck_retries)
                        current_z += adaptive_z_step
                        if current_z > Z_SAFE_HEIGHT:
                            current_z = Z_SAFE_HEIGHT
                        cnc.move_to(z=current_z, speed=Z_PROBE_SPEED)
                        time.sleep(VERTICAL_Z_SETTLE_SEC)

                if STOP_REQUESTED:
                    break
                if not horizontal_preload_locked:
                    skip_reason = (
                        f"Grid ({x:.2f}, {y:.2f}): fixed horizontal Fz {horizontal_fz_target:.2f}N - could not lock in window"
                    )
                    print(f"  [Skip] {skip_reason}")
                    log_skip_reason(skip_reason)
                    continue

                raw_horizontal_forces = np.linspace(0, MAX_HORIZONTAL_FORCE, HORIZONTAL_FORCE_STEPS)[1:]
                if raw_horizontal_forces.size > 0:
                    horizontal_forces = np.maximum(raw_horizontal_forces, MIN_HORIZONTAL_FORCE_TARGET)
                    horizontal_forces = np.unique(np.round(horizontal_forces, decimals=6))
                else:
                    horizontal_forces = raw_horizontal_forces

                current_z = run_horizontal_collection(
                    point_index=i,
                    x=x,
                    y=y,
                    current_z=current_z,
                    horizontal_fz_target=horizontal_fz_target,
                    horizontal_forces=horizontal_forces,
                    initial_message=(
                        f"  [水平采集] 当前Fz={float(read_forces(ft_sensor)[2]):.2f}N, "
                        f"固定Max_Fh={MAX_HORIZONTAL_FORCE:.2f}N, "
                        f"Fh阶梯: {[f'{fh:.2f}' for fh in horizontal_forces]}"
                    ),
                    reprobe_message=f"    [Reset] Re-probing Z to fixed Fz {horizontal_fz_target:.2f}N for next angle...",
                    skip_initial_reached=False,
                )

        print("\n采集完成。")

    except KeyboardInterrupt as e:
        print(f"\n中断: {e}")
        if cnc:
            cnc.emergency_stop()
    except RuntimeError as e:
        if "Sensor disconnected" in str(e):
            print(f"\n[严重错误] 传感器断开连接: {e}")
            if cnc:
                try:
                    print("尝试抬高 Z 轴至安全高度...")
                    cnc.rapid_move_to(z=Z_SAFE_HEIGHT)
                except Exception as lift_err:
                    print(f"抬高 Z 轴失败: {lift_err}")
        else:
            print(f"\n运行错误: {e}")
    except Exception as e:
        # print(f"\n错误: {e}")
        print(f"错误: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    finally:
        KEY_READER.disable()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if ft_sensor:
            try:
                ft_sensor.stopStreaming()
            except Exception:
                pass
        if cnc:
            try:
                print("采集结束，抬升 Z 轴到安全高度...")
                cnc.rapid_move_to(z=Z_SAFE_HEIGHT)
                print("Z 轴已抬升至安全高度")
            except Exception as e:
                print(f"抬升 Z 轴失败: {e}")
            cnc.close()


if __name__ == '__main__':
    main()
