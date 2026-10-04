import ast
import importlib.util
import inspect
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import cv2
import torchvision.transforms.functional as TF
# from scipy.fftpack import dst, idst
import time
from torch.utils.dlpack import to_dlpack
from torch.utils.data import DataLoader
# pip install cupy-cuda11x
# python -m cupyx.tools.install_library --cuda 11.x --library cutensor
import torch.nn.functional as F
import os
import datetime

from gel_marker_inpaint_viewer import generate_marker_mask, inpaint_markers  # 新增：复用标记点去除与插值逻辑
from dataset import Dataset_gel


@dataclass(frozen=True)
class VersionModelConfig:
    version: int
    version_dir: str
    training_script_path: str
    model_path: str
    basenet_class_name: str
    basenet_kwargs: dict
    use_pos: bool
    use_nir: bool


def _eval_config_ast(node: ast.AST, context: dict):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise KeyError(f"Unresolved name in config AST: {node.id}")
        return context[node.id]
    if isinstance(node, ast.List):
        return [_eval_config_ast(item, context) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_config_ast(item, context) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _eval_config_ast(key, context): _eval_config_ast(value, context)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_config_ast(node.operand, context)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_config_ast(node.operand, context)
    raise ValueError(f"Unsupported config AST node: {ast.dump(node, include_attributes=False)}")


class _BasenetAssignVisitor(ast.NodeVisitor):
    def __init__(self):
        self.call_node = None

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == 'self'
                and target.attr == 'basenet'
                and isinstance(node.value, ast.Call)
            ):
                self.call_node = node.value
        self.generic_visit(node)


def _extract_basenet_call(module_ast: ast.Module) -> ast.Call | None:
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Lit_Model':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                    visitor = _BasenetAssignVisitor()
                    visitor.visit(child)
                    return visitor.call_node
    return None


@lru_cache(maxsize=None)
def resolve_version_model_config(version: int) -> VersionModelConfig | None:
    version_dir = str(Path(__file__).resolve().parent / 'logs' / 'gel_experiment' / f'version_{version}')
    training_script_path = os.path.join(version_dir, '3_training_pix2normal.py')
    model_path = os.path.join(version_dir, 'model.py')

    config_path = Path(version_dir) / "geometry_config.json"
    if config_path.exists():
        from geometry_processing import load_config
        config = load_config(config_path)
        return VersionModelConfig(version, version_dir, training_script_path, model_path,
            config["network"]["class"], config["network"]["kwargs"],
            config["input"]["use_pos"], config["input"]["use_nir"])

    if not os.path.exists(training_script_path) or not os.path.exists(model_path):
        return None

    with open(training_script_path, 'r', encoding='utf-8') as f:
        module_ast = ast.parse(f.read(), filename=training_script_path)

    context = {'version': version}
    for node in module_ast.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            context[target.id] = _eval_config_ast(node.value, context)
        except Exception:
            continue

    basenet_call = _extract_basenet_call(module_ast)
    if basenet_call is None:
        return None

    if isinstance(basenet_call.func, ast.Name):
        basenet_class_name = basenet_call.func.id
    elif isinstance(basenet_call.func, ast.Attribute):
        basenet_class_name = basenet_call.func.attr
    else:
        raise ValueError(f"Unsupported basenet constructor: {ast.dump(basenet_call.func, include_attributes=False)}")

    basenet_kwargs = {}
    for keyword in basenet_call.keywords:
        if keyword.arg is None:
            raise ValueError("Unsupported **kwargs in basenet constructor")
        basenet_kwargs[keyword.arg] = _eval_config_ast(keyword.value, context)

    return VersionModelConfig(
        version=version,
        version_dir=version_dir,
        training_script_path=training_script_path,
        model_path=model_path,
        basenet_class_name=basenet_class_name,
        basenet_kwargs=basenet_kwargs,
        use_pos=bool(context.get('use_pos', False)),
        use_nir=bool(context.get('nir_training', True)),
    )


@lru_cache(maxsize=None)
def _load_version_model_module(model_path: str):
    module_name = f"version_model_{abs(hash(model_path))}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _load_repo_model_module():
    model_path = str(Path(__file__).resolve().with_name('model.py'))
    module_name = f"repo_model_{abs(hash(model_path))}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_fallback_basenet_class(model_module):
    for class_name in ('Modelv2_ori', 'Model_quan_l3_32', 'Model_quan_l4_32', 'Model_quan_l3_16', 'Model_quan_l4_16'):
        basenet_cls = getattr(model_module, class_name, None)
        if inspect.isclass(basenet_cls):
            return basenet_cls
    raise AttributeError("No supported fallback basenet class found in current model.py")


class Geo_Model():
    def __init__(self, version=0, use_pos=None, use_nir=None, H=480, W=640, enable_channels_last=False, device='cuda'):
        self.device = torch.device(device)

        # store default H/W from constructor so load_bg() can fall back if needed
        self._default_H = H
        self._default_W = W

        self.model = torch.nn.Module()
        self.version_config = resolve_version_model_config(version)
        self.geometry_config = None
        config_path = Path(__file__).resolve().parent / 'logs' / 'gel_experiment' / f'version_{version}' / 'geometry_config.json'
        if config_path.exists():
            from geometry_processing import load_config
            self.geometry_config = load_config(config_path)
            for supplied, key in ((use_pos, "use_pos"), (use_nir, "use_nir")):
                if supplied is not None and supplied != self.geometry_config["input"][key]:
                    raise ValueError(f"Explicit {key} conflicts with saved geometry configuration")

        if self.version_config is not None:
            resolved_use_pos = self.version_config.use_pos if use_pos is None else use_pos
            resolved_use_nir = self.version_config.use_nir if use_nir is None else use_nir
            version_model_module = _load_version_model_module(self.version_config.model_path)
            basenet_cls = getattr(version_model_module, self.version_config.basenet_class_name, None)
            if basenet_cls is None:
                raise AttributeError(
                    f"{self.version_config.model_path} does not define "
                    f"{self.version_config.basenet_class_name}"
                )
            basenet_kwargs = dict(self.version_config.basenet_kwargs)

            basenet_signature = inspect.signature(basenet_cls)
            if 'use_pos' in basenet_signature.parameters:
                basenet_kwargs['use_pos'] = resolved_use_pos
            if 'nir_training' in basenet_signature.parameters:
                basenet_kwargs['nir_training'] = resolved_use_nir

            self.model.basenet = basenet_cls(**basenet_kwargs)
            print(
                f"[Geo_Model] version_{version} config loaded from backup: "
                f"{self.version_config.basenet_class_name}(use_pos={resolved_use_pos}, use_nir={resolved_use_nir})"
            )
        else:
            raise FileNotFoundError(
                f"Geometry version {version} is missing its matching configuration/model snapshot. "
                "See README.md: Required resources; no root-model fallback is used."
            )
        # self.model.basenet = Model_quan(use_pos=resolved_use_pos, nir_training=resolved_use_nir)  # 量化模型
        # self.model.basenet = Modelv3(use_pos=resolved_use_pos, nir_training=resolved_use_nir, version=version,
        #                              save_input=False)  # FPGA推理 小模型

        # # 定义日志文件路径并确保目录存在
        # log_path = f'logs/gel_experiment/version_{version}/inference_layer_outputs_20250623.txt'
        # os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # # 使用 Model_monitor_inference 以便在推理时记录每一层输出
        # self.model.basenet = Model_monitor_inference(
        #     use_pos=use_pos,
        #     nir_training=use_nir,
        #     save_txt_path=log_path
        # )
        # 自动查找 checkpoints 目录下的 ckpt 文件：
        # 优先使用 best.ckpt；若不存在且仅有一个 .ckpt 文件则使用该文件；若存在多个则选择最新修改时间的一个并打印警告。
        ckpt_dir = str(Path(__file__).resolve().parent / 'logs' / 'gel_experiment' / f'version_{version}' / 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # 首先尝试 best.ckpt
        best_path = os.path.join(ckpt_dir, 'best.ckpt')
        ckpt_path = None
        if os.path.exists(best_path):
            ckpt_path = best_path
        else:
            # 列出所有以 .ckpt 结尾的文件
            ckpt_files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if len(ckpt_files) == 0:
                raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
            elif len(ckpt_files) == 1:
                ckpt_path = ckpt_files[0]
            else:
                # 多个 checkpoint，挑选最新修改时间的一个，并提示
                ckpt_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                ckpt_path = ckpt_files[0]
                print(f"[WARN] Multiple .ckpt files found in {ckpt_dir}; using newest: {os.path.basename(ckpt_path)}")

        self.checkpoint_path = ckpt_path
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if self.geometry_config is not None:
            embedded = checkpoint.get("hyper_parameters", {}).get("geometry_config")
            if embedded != self.geometry_config:
                raise ValueError("Checkpoint and geometry_config.json disagree")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)
        # if enable_channels_last:
        #     self.model = self.model.to(memory_format=torch.channels_last)
        self.use_pos = resolved_use_pos
        self.use_nir = resolved_use_nir
        self.sample = dict()
        # Position grid will be created from the background image size when load_bg() is called.
        self.position = None
        # # 第三个传感器
        # scale_factor = 1.3409
        # angle = -1.12  # 顺时针旋转
        # target_x, target_y = -72.23, -89.93
        # # 第二个传感器
        # scale_factor = 1.3455
        # angle = 2.48  # 顺时针旋转
        # target_x, target_y = -137.31, -129.12
        output_size = (640, 480)
        # self.transformer = ImageTransformer(scale_factor, angle, target_x, target_y, output_size)
        self.bg_rgb = None
        self.bg_nir = None
        self._warmed = False  # 标记是否已完成预热

    def load_bg(self, bg_rgb, bg_nir=None):
        if bg_rgb is None:
            raise ValueError("bg_rgb must not be None")

        bg_rgb_tensor = TF.to_tensor(bg_rgb).unsqueeze(0).to(self.device)
        self.bg_rgb = bg_rgb_tensor
        self.sample['bg_rgb'] = self.bg_rgb

        # 如果需要 position 编码，则根据传入的背景图像大小生成坐标网格（H, W）
        if self.use_pos:
            try:
                h, w = bg_rgb.shape[:2]
            except Exception:
                # 如果 bg_rgb 不是 numpy 数组（不太可能），则使用构造函数中的默认 H/W
                h, w = self._default_H, self._default_W
            ref_y = -torch.linspace(-h + 1, h - 1, h) / w
            ref_x = torch.linspace(-w + 1, w - 1, w) / w
            ref_y, ref_x = torch.meshgrid([ref_y, ref_x], indexing='ij')
            self.position = torch.cat([ref_x.unsqueeze(0), ref_y.unsqueeze(0)], dim=0).unsqueeze(0).to(self.device)
            self.sample['position'] = self.position

        if self.use_nir:
            if bg_nir is None:
                raise ValueError("use_nir=True requires bg_nir input")
            bg_nir_tensor = TF.to_tensor(bg_nir).unsqueeze(0).to(self.device)
            self.bg_nir = bg_nir_tensor
            self.sample['bg_nir'] = self.bg_nir
        else:
            # 确保旧的 NIR 状态被清理
            self.bg_nir = None
            self.sample.pop('bg_nir', None)

    # def warmup(self, example_rgb, example_nir=None, runs=10):
    #     # 供外部可单独调用；或首次测时自动触发
    #     _ = self.predict(example_rgb, example_nir, measure=False)  # 构建必需状态
    #     with torch.inference_mode():
    #         for _ in range(runs):
    #             _ = self.model.basenet(self.sample)
    #         if self.device.type == 'cuda':
    #             torch.cuda.synchronize()
    #     self._warmed = True

    def warmup(self, runs=10):
        """
        预热模型。
        首次调用 predict(measure=True) 时会自动调用此函数。
        """
        if self._warmed:
            return
        # 确保 self.sample 中有数据用于推理
        if 'rgb' not in self.sample:
            raise RuntimeError(
                "Please call predict() once with measure=False or load data into self.sample before warmup.")

        with torch.inference_mode():
            for _ in range(runs):
                _ = self.model.basenet(self.sample)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        self._warmed = True

    def _forward_once(self):
        # 只做一次前向
        return self.model.basenet(self.sample)

    # def predict(self, rgb, nir=None, measure=False, repeats=50):
    #     """
    #         measure=False 只前向一次返回 pred_FPGA
    #         measure=True  进行 repeats 次循环取平均时间(毫秒)并返回 (pred_FPGA, avg_ms)
    #     """
    #     if self.use_nir:
    #         assert (nir is not None) and (self.bg is not None)
    #         rgb, nir = self.transformer.apply_transformation(rgb, nir)
    #         cv2.imshow("rgb2", rgb)
    #         cv2.imshow("nir2", nir)
    #         rgb = TF.to_tensor(rgb).unsqueeze(0)  # 归一化并添加batch维度
    #         nir = TF.to_tensor(nir).unsqueeze(0)
    #         fg = torch.cat([rgb, nir], dim=1)  # 跟分开将rgb和nir制作成samples等价，只是为了方便
    #         # cv2.imshow("rgb_image", dst['rgb'])
    #         # cv2.imshow("nir_image", dst['ir'])
    #     else:
    #         # rgb, _ = self.transformer.apply_transformation(rgb, rgb)
    #         fg = TF.to_tensor(rgb).unsqueeze(0).to(device=self.device, dtype=torch.float32)
    #
    #     self.sample['rgb'] = fg
    #
    #     # 首次若需要测时且未 warmup，自动预热
    #     # if measure and (not self._warmed):
    #     #     self.warmup(rgb if not self.use_nir else rgb, nir if self.use_nir else None, runs=10)
    #
    #     # 首次若需要测时且未 warmup，自动预热
    #     if measure and not self._warmed:
    #         self.warmup(runs=10)
    #
    #     if not measure:
    #         with torch.inference_mode():
    #             pred = self._forward_once()
    #         pred = F.normalize(pred, p=2, dim=1).squeeze(0).permute(1, 2, 0)
    #         return pred  # 仅返回 pred_FPGA
    #
    #     # 测试平均耗时
    #     with torch.inference_mode():
    #         if self.device.type == 'cuda':
    #             torch.cuda.synchronize()
    #             starter = torch.cuda.Event(enable_timing=True)
    #             ender = torch.cuda.Event(enable_timing=True)
    #             starter.record()
    #             for _ in range(repeats):
    #                 pred = self._forward_once()
    #             ender.record()
    #             torch.cuda.synchronize()
    #             avg_ms = starter.elapsed_time(ender) / repeats
    #         else:  # CPU
    #             start_time = time.perf_counter()
    #             for _ in range(repeats):
    #                 pred = self._forward_once()
    #             end_time = time.perf_counter()
    #             avg_ms = (end_time - start_time) * 1000 / repeats
    #
    #     pred = F.normalize(pred, p=2, dim=1).squeeze(0).permute(1, 2, 0)
    #     return pred, avg_ms

    def predict(self, rgb_fg: torch.Tensor, nir_fg: torch.Tensor | None = None, measure=False, repeats=50):
        """
            接收预处理后的张量 fg。
            measure=False 只前向一次返回 pred_FPGA
            measure=True  进行 repeats 次循环取平均时间(毫秒)并返回 (pred_FPGA, avg_ms)
        """
        rgb_tensor = rgb_fg.to(self.device)
        self.sample['rgb'] = rgb_tensor

        if self.use_nir:
            if nir_fg is None:
                raise ValueError("use_nir=True requires nir_fg input")
            self.sample['nir'] = nir_fg.to(self.device)
        else:
            # 确保旧的 NIR 状态被清理
            self.sample.pop('nir', None)

        # 首次若需要测时且未 warmup，自动预热
        if measure and not self._warmed:
            self.warmup(runs=10)

        if not measure:
            with torch.inference_mode():
                pred = self._forward_once()
            pred = F.normalize(pred, p=2, dim=1).squeeze(0).permute(1, 2, 0)
            return pred, 0  # 仅返回 pred_FPGA

        # 测试平均耗时
        with torch.inference_mode():
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                for _ in range(repeats):
                    pred = self._forward_once()
                ender.record()
                torch.cuda.synchronize()
                avg_ms = starter.elapsed_time(ender) / repeats
            else:  # CPU
                start_time = time.perf_counter()
                for _ in range(repeats):
                    pred = self._forward_once()
                end_time = time.perf_counter()
                avg_ms = (end_time - start_time) * 1000 / repeats

        pred = F.normalize(pred, p=2, dim=1).squeeze(0).permute(1, 2, 0)
        return pred, avg_ms



def save_pred_as_png(pred: torch.Tensor, path: str = 'pred.png', time_format: str = '%Y%m%d_%H%M%S') -> str:
    """
    保存预测结果为 PNG，文件名自动加时间戳并返回最终路径。
    pred_FPGA: torch.Tensor, shape (H, W, 3)，float，范围大致在 -1..1（经 F.normalize）
    path: 基本保存路径，例如 `pred_FPGA.png` 或 `some/dir/pred_FPGA.png`
    time_format: 时间戳格式，默认 '%Y%m%d_%H%M%S'
    返回：实际保存的文件完整路径（字符串）
    """
    # 复制到 CPU 不改变原 tensor
    p = pred.clone().detach().cpu().float()

    # 对法向量的nx, ny通道进行缩放以增强可视化效果
    if p.ndim == 3 and p.shape[2] >= 2:
        p[..., :2] = p[..., :2] / 0.029769
    else:
        print("Warning: unexpected pred shape, expected (H, W, C) with C >= 2; skipping channel scaling")

    arr = p.numpy()
    arr = (arr * 0.5 + 0.5)  # -1..1 -> 0..1
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    # 处理文件名并插入时间戳
    base, ext = os.path.splitext(path)
    if ext == '':
        ext = '.png'
    timestamp = datetime.datetime.now().strftime(time_format)
    full_path = f"{base}_{timestamp}{ext}"

    # 确保目录存在
    dirpath = os.path.dirname(full_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # 转为 BGR 并保存
    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(full_path, img_bgr)
    cv2.imwrite(path, img_bgr)

    # 显示并返回保存路径
    print(f"Saved pred.png to: `{path}`")
    return img_bgr


def preprocess_rgb_with_inpaint(rgb_img, inpaint_radius=3, dilate_iter=5):
    """对单张 RGB 图像执行标记点区域检测和 inpaint 去除标记点，返回处理后的图像。"""
    if rgb_img is None:
        return None
    mask = generate_marker_mask(rgb_img)
    inpainted, _ = inpaint_markers(
        rgb_img,
        mask,
        method="telea",
        radius=inpaint_radius,
        dilate_iter=dilate_iter,
    )
    return inpainted


def preprocess_nir_with_inpaint(nir_img, mask_from_rgb, inpaint_radius=3, dilate_iter=5):
    """对 NIR 图像执行 inpaint，优先复用 RGB 的 mask。"""
    if nir_img is None:
        return None

    if mask_from_rgb is None:
        # 如果没有 RGB mask，尝试使用 NIR 生成 mask（转换为 BGR 以复用生成逻辑）
        if nir_img.ndim == 2:
            nir_bgr = cv2.cvtColor(nir_img, cv2.COLOR_GRAY2BGR)
        else:
            nir_bgr = nir_img
        mask = generate_marker_mask(nir_bgr)
    else:
        mask = mask_from_rgb

    inpainted, _ = inpaint_markers(
        nir_img,
        mask,
        method="telea",
        radius=inpaint_radius,
        dilate_iter=dilate_iter,
    )
    return inpainted


def list_rgb_images(img_dir: str) -> list[str]:
    if not os.path.isdir(img_dir):
        return []
    files = []
    for name in sorted(os.listdir(img_dir)):
        if not name.endswith('_rgb.png'):
            continue
        if name.startswith('0_'):
            continue
        files.append(os.path.join(img_dir, name))
    return files


def first_rgb_image(img_dir: str) -> str | None:
    files = list_rgb_images(img_dir)
    return files[0] if files else None


def get_nir_path_from_rgb(rgb_path: str) -> str:
    if rgb_path.endswith('_rgb.png'):
        return rgb_path.replace('_rgb.png', '_nir.png')
    return rgb_path


def read_nir_image(nir_path: str, nir_scale: float = 1.0) -> np.ndarray | None:
    """Read NIR with the same logic as Dataset_gel._read_nir: grayscale + optional scaling."""
    if not os.path.exists(nir_path):
        return None
    img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    scale = float(nir_scale)
    if scale != 1.0:
        img_f = img.astype(np.float32) * scale
        img_f = np.clip(img_f, 0, 255)
        img_u8 = img_f.astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)
    return img_u8


def compute_px_py_errors(pred_normal: torch.Tensor, gt_normal: np.ndarray) -> tuple[float, float, float]:
    """
    Compute per-pixel L1 errors for px and py and their sum (px+py).
    px = nx / nz, py = ny / nz
    Returns (px_err_mean, py_err_mean, pxpy_err_mean)
    pred_normal: torch.Tensor (H, W, 3)
    gt_normal: np.ndarray (H, W, 3)
    """
    if gt_normal is None:
        return float('nan'), float('nan'), float('nan')

    pred = pred_normal.detach().cpu()
    if pred.ndim != 3 or pred.shape[2] != 3:
        return float('nan'), float('nan'), float('nan')

    gt = torch.from_numpy(gt_normal).float()
    if gt.ndim != 3 or gt.shape[2] != 3:
        return float('nan'), float('nan'), float('nan')

    # mask out invalid gt (all zeros)
    mask = ~(gt == 0).all(dim=2)
    if mask.sum() == 0:
        return float('nan'), float('nan'), float('nan')

    eps = 1e-6
    px_pred = pred[..., 0] / (pred[..., 2] + eps)
    py_pred = pred[..., 1] / (pred[..., 2] + eps)

    px_gt = gt[..., 0] / (gt[..., 2] + eps)
    py_gt = gt[..., 1] / (gt[..., 2] + eps)

    # L1 error on valid pixels
    px_err = (px_pred - px_gt).abs()[mask].mean()
    py_err = (py_pred - py_gt).abs()[mask].mean()
    pxpy_err = (px_err + py_err)
    return float(px_err), float(py_err), float(pxpy_err)


def collect_normal_ids(normal_gt_dir: str) -> list[int]:
    if not os.path.isdir(normal_gt_dir):
        return []
    ids: list[int] = []
    for name in os.listdir(normal_gt_dir):
        if not name.startswith("normal") or not name.endswith(".npy"):
            continue
        core = name[len("normal"):-len(".npy")]
        if not core.isdigit():
            continue
        idx = int(core)
        if idx == 0:
            continue
        ids.append(idx)
    return sorted(ids)


def run_dataset_evaluation(
    model: Geo_Model,
    record: str,
    split: str,
    batch_size: int,
    num_workers: int,
    nir_training: bool,
    nir_scale: float,
):
    """Evaluate using Dataset_gel/DataLoader to match training preprocessing and loss."""
    data_root = "."
    split_dir = os.path.join(data_root, "data", record, split)
    normal_gt_dir = os.path.join(split_dir, "normal_gt")
    sample_ids = collect_normal_ids(normal_gt_dir)
    sample_num = len(sample_ids)
    if sample_num == 0:
        print(f"[WARN] No normal_gt files found for dataset eval: {normal_gt_dir}")
        return float('nan'), float('nan'), float('nan'), 0

    dataset = Dataset_gel(data_root, f"{record}/{split}", sample_num=sample_num, nir_training=nir_training, nir_scale=nir_scale, mode="eval")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    l1 = torch.nn.L1Loss()
    total_px = 0.0
    total_py = 0.0
    total_pxpy = 0.0
    count = 0

    with torch.inference_mode():
        for sample in dataloader:
            # move tensors to device
            for k, v in list(sample.items()):
                if torch.is_tensor(v):
                    sample[k] = v.to(model.device)

            pred = model.model.basenet(sample)
            normal_norm = F.normalize(pred, p=2, dim=1)

            eps = 1e-6
            nx = normal_norm[:, 0, ...]
            ny = normal_norm[:, 1, ...]
            nz = normal_norm[:, 2, ...]
            px_pred = nx / (nz + eps)
            py_pred = ny / (nz + eps)

            gt = sample['gt']
            gx = gt[:, 0, ...]
            gy = gt[:, 1, ...]
            gz = gt[:, 2, ...]
            px_gt = gx / (gz + eps)
            py_gt = gy / (gz + eps)

            loss_px = l1(px_pred, px_gt)
            loss_py = l1(py_pred, py_gt)
            loss_pxpy = loss_px + loss_py

            batch_size_actual = gt.shape[0]
            total_px += float(loss_px) * batch_size_actual
            total_py += float(loss_py) * batch_size_actual
            total_pxpy += float(loss_pxpy) * batch_size_actual
            count += batch_size_actual

    avg_px = total_px / count if count > 0 else float('nan')
    avg_py = total_py / count if count > 0 else float('nan')
    avg_pxpy = total_pxpy / count if count > 0 else float('nan')
    return avg_px, avg_py, avg_pxpy, count


def try_get_gpu_power_watts() -> float | None:
    """Try to read GPU power draw in watts. Returns None if unavailable."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power_mw / 1000.0
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            val = result.stdout.strip().splitlines()[0]
            return float(val)
    except Exception:
        pass

    return None


def sample_gpu_power_watts(model: Geo_Model, repeats: int = 60, sample_every: int = 5) -> tuple[float | None, int]:
    """Sample GPU power during repeated inference. Returns (avg_power_w, num_samples)."""
    if model.device.type != 'cuda':
        return None, 0

    readings: list[float] = []
    with torch.inference_mode():
        for i in range(repeats):
            _ = model._forward_once()
            if i % sample_every == 0:
                torch.cuda.synchronize()
                p = try_get_gpu_power_watts()
                if p is not None:
                    readings.append(p)

        torch.cuda.synchronize()

    if not readings:
        return None, 0
    return float(sum(readings) / len(readings)), len(readings)


def measure_inference_time(model: Geo_Model, rgb_tensor: torch.Tensor, nir_tensor: torch.Tensor | None, repeats: int = 30):
    # warmup once
    if not model._warmed:
        _ = model.predict(rgb_tensor, nir_tensor, measure=False)
        model.warmup(runs=10)

    if model.device.type == 'cuda':
        with torch.inference_mode():
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(repeats):
                _ = model._forward_once()
            ender.record()
            torch.cuda.synchronize()
            avg_ms = starter.elapsed_time(ender) / repeats
    else:
        start_time = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                _ = model._forward_once()
        end_time = time.perf_counter()
        avg_ms = (end_time - start_time) * 1000.0 / repeats
    return avg_ms


def run_folder_inference(
    model: Geo_Model,
    img_dir: str,
    normal_gt_dir: str,
    use_nir: bool,
    enable_inpaint: bool,
    inpaint_radius: int,
    inpaint_dilate_iter: int,
    nir_scale: float,
):
    # Iterate over ground-truth normal files; only process images that have a corresponding normal*.npy
    if not os.path.isdir(normal_gt_dir):
        print(f"[WARN] normal_gt directory not found: {normal_gt_dir}")
        return float('nan'), float('nan'), float('nan'), 0

    normal_files = [f for f in sorted(os.listdir(normal_gt_dir)) if f.startswith('normal') and f.endswith('.npy')]
    if not normal_files:
        print(f"[WARN] No normal_*.npy files found in {normal_gt_dir}")
        return float('nan'), float('nan'), float('nan'), 0

    total_px = 0.0
    total_py = 0.0
    total_pxpy = 0.0
    count = 0

    for nf in normal_files:
        # extract the id suffix from filename like 'normal5.npy' -> '5'
        base_id = nf[len('normal'):-4]
        rgb_path = os.path.join(img_dir, f"{base_id}_rgb.png")
        if not os.path.exists(rgb_path):
            # skip if RGB missing (this normal has no corresponding image)
            # print a debug message at verbose levels if needed
            # print(f"[DEBUG] RGB missing for normal {nf}: expected {rgb_path}")
            continue

        rgb_raw = cv2.imread(rgb_path)
        if rgb_raw is None:
            print(f"[WARN] Failed to read {rgb_path}")
            continue

        rgb_img = rgb_raw
        mask_from_rgb = None
        if enable_inpaint:
            try:
                rgb_img = preprocess_rgb_with_inpaint(rgb_raw, inpaint_radius=inpaint_radius,
                                                      dilate_iter=inpaint_dilate_iter)
                try:
                    mask_from_rgb = generate_marker_mask(rgb_raw)
                except Exception:
                    mask_from_rgb = None
            except Exception as e:
                print(f"[WARN] Inpaint failed for {rgb_path}: {e}")
                rgb_img = rgb_raw

        rgb_tensor = TF.to_tensor(rgb_img).unsqueeze(0)

        nir_tensor = None
        if use_nir:
            nir_path = os.path.join(img_dir, f"{base_id}_nir.png")
            nir_img = read_nir_image(nir_path, nir_scale=nir_scale)
            if nir_img is None:
                print(f"[WARN] NIR missing for {rgb_path}: {nir_path}")
            else:
                if enable_inpaint:
                    try:
                        nir_img = preprocess_nir_with_inpaint(
                            nir_img,
                            mask_from_rgb=mask_from_rgb,
                            inpaint_radius=inpaint_radius,
                            dilate_iter=inpaint_dilate_iter,
                        )
                    except Exception as e:
                        print(f"[WARN] NIR inpaint failed for {nir_path}: {e}")
                nir_tensor = TF.to_tensor(nir_img).unsqueeze(0)

        pred, _ = model.predict(rgb_tensor, nir_tensor, measure=False)

        # load gt normals (we already know filename from nf)
        gt_path = os.path.join(normal_gt_dir, nf)
        gt = np.load(gt_path)

        px_err, py_err, pxpy_err = compute_px_py_errors(pred, gt)
        if not np.isnan(pxpy_err):
            total_px += px_err
            total_py += py_err
            total_pxpy += pxpy_err
            count += 1

    avg_px = total_px / count if count > 0 else float('nan')
    avg_py = total_py / count if count > 0 else float('nan')
    avg_pxpy = total_pxpy / count if count > 0 else float('nan')
    return avg_px, avg_py, avg_pxpy, count


def export_inference_input(model: Geo_Model, save_path: str, sample_idx: str | int | None = None, img_dir: str | None = None):
    """
    Export the current input sample in the model to a text file
    compatible with convert_inputs.py (CHW flattened, one float per line).
    """
    print(f"Exporting input data from model sample to {save_path}...")
    sample = model.sample.copy() if isinstance(model.sample, dict) else dict(model.sample)

    # If a specific sample index is requested, try to load rgb/nir from img_dir
    if sample_idx is not None:
        if img_dir is None:
            raise ValueError("img_dir must be provided when sample_idx is specified")

        sid = str(sample_idx)
        # expected filename pattern: <id>_rgb.png and <id>_nir.png
        rgb_path = os.path.join(img_dir, f"{sid}_rgb.png")
        nir_path = os.path.join(img_dir, f"{sid}_nir.png")

        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"Requested rgb sample not found: {rgb_path}")

        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise RuntimeError(f"Failed to read rgb image: {rgb_path}")

        rgb_tensor = TF.to_tensor(rgb_img).unsqueeze(0).to(model.device)
        sample['rgb'] = rgb_tensor

        if model.use_nir:
            nir_img = read_nir_image(nir_path)
            if nir_img is None:
                raise FileNotFoundError(f"Requested nir sample not found: {nir_path}")
            nir_tensor = TF.to_tensor(nir_img).unsqueeze(0).to(model.device)
            sample['nir'] = nir_tensor
    
    # 构造输入张量 (RGB_diff, NIR_diff, Position)
    # 顺序需与 model.py / model_QAT.py 中的 forward 或 cat 顺序一致
    tensors = []

    # 1. RGB Diff
    if 'rgb' not in sample or 'bg_rgb' not in sample:
        print("Error: Missing rgb or bg_rgb in sample")
        return
    rgb_diff = sample['rgb'] - sample['bg_rgb']
    tensors.append(rgb_diff)
    
    # 2. NIR Diff
    if model.use_nir:
        if 'nir' not in sample or 'bg_nir' not in sample:
             print("Error: Missing nir or bg_nir in sample")
             return
        nir_diff = sample['nir'] - sample['bg_nir']
        tensors.append(nir_diff)
        
    # 3. Position
    if model.use_pos and 'position' in sample:
        tensors.append(sample['position'])
        
    combined = torch.cat(tensors, dim=1) # (N, C, H, W)
    
    # 取第一张
    if combined.shape[0] > 0:
        data = combined[0].detach().cpu().numpy() # (C, H, W)
        flat = data.flatten() # C-style flattening (iterate C, then H, then W)
        
        with open(save_path, 'w') as f:
            # f.write(f"# Shape: {data.shape} (C, H, W)\n")
            # f.write(f"# Channels order: RGB_diff(3), NIR_diff(1), Pos(2)\n")
            for val in flat:
                f.write(f"{val:.8f}\n")
                
        print(f"[Export] Saved inference input to {save_path} (Length: {len(flat)})")
    else:
        print("[Export] No batch data to export.")


if __name__ == '__main__':
    from geometry_evaluation import main
    main()
