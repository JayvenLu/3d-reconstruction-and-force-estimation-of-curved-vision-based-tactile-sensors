"""Numerical runtime: unchanged calibration, RBF, Poisson and force definitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import threading
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .bootstrap import ROOT, load_reference
from . import _matching
from .camera import DEFAULT_EXPOSURE, DEFAULT_AUTO_WHITE_BALANCE, DEFAULT_FOURCC


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    camera_fps: int = 30
    camera_fourcc: str = DEFAULT_FOURCC
    camera_exposure: int = DEFAULT_EXPOSURE
    camera_auto_white_balance: bool = DEFAULT_AUTO_WHITE_BALANCE
    display_hz: float = 5.0  # Diagnostic windows only; main preview follows every result.
    arrow_scale: float = 1000.0
    arrow_min_force: float = .005  # XY norm threshold, independent of Fz.
    normal_color_max: float = .08  # Fixed regional Fz color scale, N.
    geo_version: int = 118
    force_version: int = 76
    prior: str = 'depth_prior/dep_p_20260407_181422.npy'
    preset: str = 'cam/20260827-145539/grid_preset_rgb_perspective.npy'
    search_ms: float = 8.0
    opencv_threads: int = 1
    max_capture_age_ms: float = 20.0
    max_prepared_age_ms: float = 58.0
    # Verified byte-identical against full-frame Telea on the complete validation split.
    crop_inpaint: bool = True


@dataclass
class DetectionParams:
    block_size: int = 165
    c: int = 20
    circularity: float = .61


@dataclass
class SparseMatch:
    valid: bool
    source: np.ndarray
    destination: np.ndarray
    indices: np.ndarray
    flow: tuple | None
    mask: np.ndarray
    detected: int
    cost: float = -1
    timed_out: bool = False
    seed_accepted: bool = False
    elapsed_ms: float = 0
    reason: str = ''


@dataclass
class PreparedFrame:
    frame_id: int
    captured_at: float
    generation: int
    raw: np.ndarray
    clean: np.ndarray
    mask: np.ndarray
    match: SparseMatch
    prepare_ms: float


@dataclass
class ForceResult:
    frame_id: int
    captured_at: float
    output_at: float
    generation: int
    valid: bool
    force: np.ndarray | None
    prepared: PreparedFrame
    arrays: dict | None = None
    metrics: dict = field(default_factory=dict)
    reason: str = ''
    preview: np.ndarray | None = None  # [3,12,16], regional contributions in target_vts N.
    snapshot_tensors: dict | None = None  # Immutable tensors owned by this exact frame.

    def materialize_arrays(self):
        """Download on demand, retaining this frame even if the engine advances."""
        if self.arrays is not None:
            return self.arrays
        if not self.valid or self.snapshot_tensors is None:
            raise ValueError('No valid snapshot payload')
        return {name:value.cpu().numpy() for name,value in self.snapshot_tensors.items()}


class MarkerTracker:
    def __init__(self, config: Config):
        self.config = config
        self.ref = load_reference()
        import setting
        setting.init()
        self.rows, self.cols = setting.N_, setting.M_
        self.dx, self.dy = setting.dx_, setting.dy_
        pts_dst = np.float32([[0, 0], [config.width, 0], [config.width, config.height], [0, config.height]])
        self.matrix = cv2.getPerspectiveTransform(self.ref.PTS_SRC, pts_dst)
        self.inverse = np.linalg.inv(self.matrix)
        self.reference = np.load(ROOT / config.preset).astype(np.float64)
        self.previous = self.reference.copy()
        self.source_raw = self.to_raw(self.reference)
        self.ellipse5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.ellipse3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.ones11 = np.ones((11, 11), np.uint8)

    def to_raw(self, pts):
        return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), self.inverse).reshape(-1, 2)

    def detect(self, frame, params):
        warp = cv2.warpPerspective(frame, self.matrix, (self.config.width, self.config.height))
        v = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)[:, :, 2]
        blur = cv2.GaussianBlur(v, (35, 35), 0)
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, params.block_size, params.c)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.ones11)
        mask = cv2.dilate(mask, self.ellipse5)
        mask = cv2.erode(mask, self.ellipse3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
            circularity = 0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)
            if circularity < params.circularity:
                continue
            moments = cv2.moments(contour)
            if moments['m00']:
                points.append((int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])))
        points = np.asarray(sorted(points), dtype=np.float64).reshape(-1, 2)
        return points, mask

    def _candidates(self, points):
        candidates = []
        for reference in (self.previous, self.reference):
            distances = np.sum((points[:, None, :] - reference[None, :, :]) ** 2, axis=2)
            row, col = linear_sum_assignment(distances)
            candidate = np.full(len(points), -1, np.int32)
            candidate[row] = col
            if not any(np.array_equal(candidate, old) for old in candidates):
                candidates.append(candidate)
            # The reference retry permits one extra spurious point and one extra
            # missing grid slot. Propose those cases too, then replay every constraint
            # in C++; nearest-neighbour assignment alone is never accepted as a match.
            residual = distances[row, col]
            for ignored in row[np.argsort(residual)[-3:]]:
                keep = np.delete(np.arange(len(points)), ignored)
                rr, cc = linear_sum_assignment(distances[keep])
                proposed = np.full(len(points), -1, np.int32)
                proposed[keep[rr]] = cc
                if not any(np.array_equal(proposed, old) for old in candidates):
                    candidates.append(proposed)
        return np.asarray(candidates, dtype=np.int32)

    def match(self, frame, params=None, background=False, search=True):
        params = params or DetectionParams()
        t = time.perf_counter()
        points, mask = self.detect(frame, params)
        empty = np.empty((0, 2), np.float32)
        if len(points) < 4 or len(points) > 900:
            return SparseMatch(False, empty, empty, np.empty(0, np.int64), None, mask,
                               len(points), reason='fewer than 4 or too many detected markers')
        result = _matching.solve(self.reference, points, self._candidates(points), self.rows, self.cols,
                                 self.dx, self.dy, 100.0 if background else self.config.search_ms, search)
        if not result['valid']:
            return SparseMatch(False, empty, empty, np.empty(0, np.int64), None, mask, len(points),
                               timed_out=result['timed_out'], elapsed_ms=(time.perf_counter()-t)*1000,
                               reason='no feasible current-frame association')
        current, occupied = result['current'], result['occupied']
        if background:
            # Like the baseline: lock measured slots, leave missing reference slots intact.
            selected = occupied >= 0
            self.reference[selected] = current[selected]
            self.source_raw = self.to_raw(self.reference)
        self.previous = current.copy()
        selected = occupied >= 0
        indices = np.flatnonzero(selected)
        source = self.source_raw[indices]
        destination = self.to_raw(current[selected])
        ox, oy = self.reference.T.reshape(2, self.rows, self.cols)
        cx, cy = current.T.reshape(2, self.rows, self.cols)
        flow = (ox.copy(), oy.copy(), cx.copy(), cy.copy(), occupied.reshape(self.rows, self.cols).copy(),
                cx-ox, oy-cy, np.hypot(cx-ox, cy-oy))
        return SparseMatch(True, source, destination, indices, flow, mask, len(points),
                           float(result['cost']), bool(result['timed_out']), bool(result['seed_accepted']),
                           (time.perf_counter()-t)*1000)


class ImagePreparer:
    def __init__(self, config, tracker):
        self.config, self.tracker = config, tracker
        self.ref = load_reference()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='marker')
        self.params = DetectionParams()
        self.lock = threading.Lock()

    def inpaint(self, frame):
        mask = self.ref.generate_marker_mask(frame, block_size=self.ref.V_BLOCKSIZE_INPAINT,
                                              c=self.ref.V_C_INPAINT)
        if not self.config.crop_inpaint:
            clean, _ = self.ref.inpaint_markers(frame, mask, method='telea', radius=3, dilate_iter=5)
            return clean, mask
        expanded = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=5)
        ys, xs = np.nonzero(expanded)
        if not len(xs):
            return frame.copy(), mask
        x0, x1 = max(0, int(xs.min())-8), min(frame.shape[1], int(xs.max())+9)
        y0, y1 = max(0, int(ys.min())-8), min(frame.shape[0], int(ys.max())+9)
        clean = frame.copy()
        clean[y0:y1, x0:x1] = cv2.inpaint(frame[y0:y1, x0:x1].copy(), expanded[y0:y1, x0:x1].copy(), 3, cv2.INPAINT_TELEA)
        return clean, mask

    def prepare(self, frame, frame_id, captured_at, generation):
        t = time.perf_counter()
        with self.lock:
            params = DetectionParams(**vars(self.params))
        match_future = self.pool.submit(self.tracker.match, frame, params)
        clean, mask = self.inpaint(frame)
        match = match_future.result()
        return PreparedFrame(frame_id, captured_at, generation, frame, clean, mask, match,
                             (time.perf_counter()-t)*1000)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


class CachedRBF:
    def __init__(self, sources, height, width, device='cuda', epsilon=80.0, regularization=.02):
        self.height, self.width = height, width
        self.device = device
        self.regularization = regularization
        points = torch.as_tensor(sources, dtype=torch.float32, device=device)
        self.count = len(points)
        # Preserve the reference norm -> square -> kernel operation ordering.
        def kernel(a, b):
            distance = torch.linalg.vector_norm(a[:, None, :] - b[None, :, :], dim=2)
            return 1.0 / torch.sqrt(distance.square() + epsilon**2)
        self.kernel = kernel(points, points)
        yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                torch.arange(width, device=device, dtype=torch.float32), indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        self.basis = torch.empty((height*width, len(points)), dtype=torch.float32, device=device)
        for begin in range(0, len(grid), 10000):
            self.basis[begin:begin+10000] = kernel(grid[begin:begin+10000], points)
        self.cache = OrderedDict()

    def __call__(self, match):
        if not match.valid or len(match.indices) < 4:
            raise ValueError('A valid current-frame association with at least four points is required')
        key = tuple(int(i) for i in match.indices)
        if key not in self.cache:
            idx = torch.tensor(key, device=self.device, dtype=torch.long)
            matrix = self.kernel[idx][:, idx].clone()
            matrix.diagonal().add_(self.regularization)
            lu, pivots = torch.linalg.lu_factor(matrix)
            # Bound cached bases: 4 masks use at most about 230 MiB at 640x480.
            basis = self.basis if len(key) == self.count else self.basis[:, idx]
            self.cache[key] = (lu, pivots, basis)
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        lu, pivots, basis = self.cache[key]
        displacement = torch.as_tensor(match.destination-match.source, device=self.device)
        weights = torch.linalg.lu_solve(lu, pivots, displacement)
        return (basis @ weights).reshape(self.height, self.width, 2).permute(2, 0, 1).unsqueeze(0)


class CachedPoisson:
    def __init__(self, prior, device='cuda'):
        import depth_reconstruction_core as dc
        ys, xs = np.where(np.isnan(prior))
        if not len(xs):
            raise ValueError('Optimized solver requires the existing rectangular boundary prior')
        self.y0, self.y1, self.x0, self.x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        y0, y1, x0, x1 = self.y0, self.y1, self.x0, self.x1
        if min(y0, x0) < 1 or y1 >= prior.shape[0]-1 or x1 >= prior.shape[1]-1 or not np.isnan(prior[y0:y1+1,x0:x1+1]).all():
            raise ValueError('Invalid rectangular boundary prior')
        self.prior = torch.as_tensor(prior, device=device)
        self.prior_cpu = prior.copy()
        self.reciprocal = torch.as_tensor(1.0 / dc._poisson_denom(y1-y0+1, x1-x0+1), device=device)
        self.dc, self.device = dc, device

    def __call__(self, normal):
        y0, y1, x0, x1 = self.y0, self.y1, self.x0, self.x1
        n = normal[y0-1:y1+2, x0-1:x1+2]
        nz = n[:, :, 2].clamp_min(1e-6)
        p, q = -n[:, :, 0]/nz, n[:, :, 1]/nz
        # Only the interior divergence is needed; this is the same forward difference.
        inner = (p[1:-1,1:-1]-p[1:-1,:-2]) + (q[1:-1,1:-1]-q[:-2,1:-1])
        inner[0] -= self.prior[y0-1, x0:x1+1]
        inner[-1] -= self.prior[y1+1, x0:x1+1]
        inner[:, 0] -= self.prior[y0:y1+1, x0-1]
        inner[:, -1] -= self.prior[y0:y1+1, x1+1]
        spectrum = self.dc._dst2_ortho_torch(inner, device=self.device)
        # The tiny Laplacian eigenvalues amplify low-frequency FFT rounding. A pure
        # GPU FP32 transform exceeded the agreed 0.005 N parity bound on this sensor.
        # Preserve the original CPU's 8x8 low modes, NOT a learned/output correction.
        # This evaluates precisely the same RHS and DST-I definition as the reference.
        from scipy.fft import dst
        normal_cpu = n.cpu().numpy()
        pp, qq = self.dc._normal_to_pq(normal_cpu)
        rhs = self.dc._forward_divergence(pp, qq)[1:-1, 1:-1].copy()
        rhs[0] -= self.prior_cpu[y0-1, x0:x1+1]
        rhs[-1] -= self.prior_cpu[y1+1, x0:x1+1]
        rhs[:, 0] -= self.prior_cpu[y0:y1+1, x0-1]
        rhs[:, -1] -= self.prior_cpu[y0:y1+1, x1+1]
        count = min(8, *rhs.shape)
        first = dst(rhs, type=1, norm='ortho')
        low = dst(first[:, :count].T, type=1, norm='ortho').T[:count, :]
        spectrum[:count, :count] = torch.as_tensor(np.ascontiguousarray(low), device=self.device)
        depth = self.dc._idst2_ortho_torch(spectrum*self.reciprocal, device=self.device)
        out = self.prior.clone()
        out[y0:y1+1, x0:x1+1] = depth
        return out


class CachedForce:
    def __init__(self, wrapper):
        self.wrapper, self.net = wrapper, wrapper.model.basenet
        pos = self.net.downsample(wrapper.pos)
        self.weights = [gen(pos) for gen in (self.net.weight_gen_x, self.net.weight_gen_y, self.net.weight_gen_z)]
        self.bias = self.net.bias_gen(pos)

    def __call__(self, displacement):
        pooled = self.net.downsample(displacement)
        unfolded, b, c, h, w, k = self.net._unfold_displacement(pooled)
        maps = torch.cat([self.net._apply_dynamic_conv(unfolded, weight, b, c, h, w, k)
                          for weight in self.weights], dim=1)
        # Retain add/subtract rounding of the reference zero-displacement calibration.
        return ((maps+self.bias)-self.bias)/1000.0


class ForceEngine:
    def __init__(self, config=Config()):
        self.config, self.ref = config, load_reference()
        if not torch.cuda.is_available():
            raise RuntimeError('Optimized runtime needs CUDA; select baseline mode for CPU')
        torch.backends.cuda.matmul.allow_tf32 = False
        # Small per-frame host tensor conversions otherwise launch eight CPU workers
        # and contend with OpenCV and the reference low-mode FFT.
        torch.set_num_threads(1)
        self.device = torch.device('cuda')
        geo_cfg = self.ref.load_geo_version_config(config.geo_version)
        self.geo = self.ref.Geo_Model(config.geo_version, use_pos=bool(geo_cfg['use_pos']),
                                     use_nir=bool(geo_cfg['nir_training']), device=self.device)
        self.force = self.ref.HyperForce_Model(config.force_version, self.device)
        flags = self.force.config
        if any(flags[k] for k in ('lag_dz','normalize_disp','convert_dz_to_pixel','convert_disp_to_sensor')) or not flags['use_bg_sub']:
            raise ValueError('Optimized input path is validated for v76 preprocessing only')
        prior = self.ref.get_resized_prior(config.prior, (config.height, config.width))
        if prior is None:
            raise FileNotFoundError(config.prior)
        self.poisson = CachedPoisson(prior, self.device)
        self.background = None
        self.generation = 0

    @torch.inference_mode()
    def set_background(self, clean, sources, generation):
        self.geo.load_bg(clean)
        normal, _ = self.geo.predict(self.ref.TF.to_tensor(clean).unsqueeze(0).to(self.device))
        # Keep the original CPU reference background, computed only on recalibration.
        bg = self.ref.reconstruct_depth_via_poisson(normal.cpu().numpy(), self.config.prior)
        self.force.set_background(bg)
        self.background = torch.as_tensor(bg, device=self.device)
        self.cached_force = CachedForce(self.force)
        self.rbf = CachedRBF(sources, self.config.height, self.config.width, self.device)
        self.generation = generation
        torch.cuda.synchronize()

    @torch.inference_mode()
    def process(self, prepared, include_arrays=False):
        start = time.perf_counter()
        if prepared.generation != self.generation or not prepared.match.valid:
            return ForceResult(prepared.frame_id, prepared.captured_at, time.perf_counter(), prepared.generation,
                               False, None, prepared, reason=prepared.match.reason or 'stale background generation')
        image = self.ref.TF.to_tensor(prepared.clean).unsqueeze(0).to(self.device)
        normal, _ = self.geo.predict(image)
        depth = self.poisson(normal)
        dz = (self.background-depth).clamp_min(0)
        xy_pixel = self.rbf(prepared.match)
        # v76 transform_disp_to_model_axes returns +dz when both rotations are disabled.
        displacement = torch.cat([xy_pixel*self.force.pixmm, dz[None,None]], dim=1)
        force_map = self.cached_force(displacement)[0]
        total = force_map.sum(dim=(1,2)).cpu().numpy()
        force_ready = time.perf_counter()
        # Each block sums 10x10 original force cells; never resize then subsample force.
        preview = force_map.reshape(3,12,10,16,10).sum(dim=(2,4)).cpu().numpy()
        preview_ready = time.perf_counter()
        payload = {'normal':normal, 'depth':depth, 'dz':dz,
                   'dx':xy_pixel[0,0], 'dy':xy_pixel[0,1], 'force_map':force_map}
        arrays = None
        if include_arrays:
            arrays = {name:value.cpu().numpy() for name,value in payload.items()}
        done = time.perf_counter()
        return ForceResult(prepared.frame_id, prepared.captured_at, done, prepared.generation, True, total,
                           prepared, arrays, {'gpu_ms':(done-start)*1000, 'force_ready_ms':(force_ready-start)*1000,
                                              'preview_ms':(preview_ready-force_ready)*1000,
                                              'prepare_ms':prepared.prepare_ms,
                                              'latency_ms':(done-prepared.captured_at)*1000},
                           preview=preview, snapshot_tensors=payload)
