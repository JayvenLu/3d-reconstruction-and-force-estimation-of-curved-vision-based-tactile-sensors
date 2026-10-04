"""Bounded latest-frame pipeline; acquisition IDs, never loop FPS, identify outputs."""
from __future__ import annotations
import collections
import threading
import time
import traceback
import numpy as np
import cv2
from .core import ForceResult
from .camera import configure_camera


class LatestSlot:
    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.overwritten = 0

    def put(self, value):
        with self.condition:
            if self.value is not None:
                self.overwritten += 1
            self.value = value
            self.condition.notify_all()

    def take(self, timeout=.1):
        with self.condition:
            if self.value is None:
                self.condition.wait(timeout)
            value, self.value = self.value, None
            return value

    def clear(self):
        with self.condition:
            self.value = None
            self.condition.notify_all()


class CameraCapture:
    def __init__(self, config):
        self.cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Camera {config.camera_index} could not be opened')
        try:
            self.info = configure_camera(self.cap, config.width, config.height, config.camera_fps,
                                         config.camera_fourcc, config.camera_exposure,
                                         config.camera_auto_white_balance)
            self.info['source'] = config.camera_index
        except Exception:
            self.cap.release()
            raise
        self.slot = LatestSlot()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.acquired = 0
        self.failures = 0
        self.timestamps = collections.deque(maxlen=3000)
        self.thread = threading.Thread(target=self._reader, name='capture', daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            timestamp = time.perf_counter()
            if not ok:
                self.failures += 1
                self.stop_event.wait(.01)
                continue
            self.acquired += 1
            item = (self.acquired, timestamp, frame)
            with self.lock:
                self.latest = item
                self.timestamps.append(timestamp)
            self.slot.put(item)

    def peek(self):
        with self.lock:
            return self.latest

    def rate(self, since=None):
        with self.lock:
            times = [t for t in self.timestamps if since is None or t>=since]
        return (len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0.

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.cap.release()
        self.thread.join(timeout=2)


class Pipeline:
    def __init__(self, engine, preparer, raw_slot, on_result=None):
        self.engine, self.preparer, self.raw = engine, preparer, raw_slot
        self.prepared, self.display = LatestSlot(), LatestSlot()
        self.diagnostic_display = LatestSlot()
        self.diagnostics_enabled = threading.Event()
        self.stop_event = threading.Event()
        self.generation = engine.generation
        self.on_result = on_result
        self.records = collections.deque(maxlen=100000)
        self.record_lock = threading.Lock()
        self.error = None
        self.threads = []
        self.last_processed_id = -1
        self.stale_inputs_skipped = 0
        self.stale_prepared_skipped = 0

    def start(self):
        self.raw.clear()
        for name, target in [('cpu-prepare', self._prepare), ('gpu-decode', self._decode)]:
            thread = threading.Thread(name=name, target=target, daemon=True)
            self.threads.append(thread)
            thread.start()

    def _prepare(self):
        try:
            while not self.stop_event.is_set():
                item = self.raw.take()
                if item is None:
                    continue
                frame_id, timestamp, frame = item
                if (time.perf_counter()-timestamp)*1000 > self.engine.config.max_capture_age_ms:
                    self.stale_inputs_skipped += 1
                    continue
                prepared = self.preparer.prepare(frame, frame_id, timestamp, self.generation)
                if not self.stop_event.is_set():
                    self.prepared.put(prepared)
        except Exception:
            self.error = traceback.format_exc()
            self.stop_event.set()

    def _decode(self):
        try:
            next_diagnostic = 0.0
            while not self.stop_event.is_set():
                item = self.prepared.take()
                if item is None:
                    continue
                if item.generation != self.generation or item.frame_id <= self.last_processed_id:
                    continue
                if (time.perf_counter()-item.captured_at)*1000 > self.engine.config.max_prepared_age_ms:
                    self.stale_prepared_skipped += 1
                    continue
                self.last_processed_id = item.frame_id
                render = self.diagnostics_enabled.is_set() and time.perf_counter() >= next_diagnostic
                result = self.engine.process(item, include_arrays=render)
                if self.stop_event.is_set():
                    break
                with self.record_lock:
                    self.records.append({'frame_id':result.frame_id, 'captured_at':result.captured_at,
                                         'output_at':result.output_at, 'valid':result.valid,
                                         'force':None if result.force is None else result.force.tolist(),
                                         'detected':item.match.detected, 'match_timeout':item.match.timed_out,
                                         'reason':result.reason, **result.metrics})
                # Integration point for control consumers: one callback per processed new frame.
                if self.on_result is not None:
                    self.on_result(result)
                self.display.put(result)
                if render:
                    if self.diagnostics_enabled.is_set():self.diagnostic_display.put(result)
                    period = 1/self.engine.config.display_hz
                    now = time.perf_counter()
                    next_diagnostic = now + period - ((now-next_diagnostic) % period)
        except Exception:
            self.error = traceback.format_exc()
            self.stop_event.set()

    def snapshot_records(self):
        with self.record_lock:
            return list(self.records)

    def set_diagnostics(self,enabled):
        if enabled:self.diagnostics_enabled.set()
        else:
            self.diagnostics_enabled.clear()
            self.diagnostic_display.clear()

    def summary(self, since=None):
        records = self.snapshot_records()
        if since is not None:
            records = [r for r in records if r['output_at'] >= since]
        good = [r for r in records if r['valid']]
        elapsed = good[-1]['output_at']-good[0]['output_at'] if len(good)>1 else 0
        latency = [1000*(r['output_at']-r['captured_at']) for r in good]
        return {'processed':len(records), 'valid':len(good), 'invalid':len(records)-len(good),
                'valid_hz':(len(good)-1)/elapsed if elapsed>0 else 0,
                'latency_p50_ms':float(np.median(latency)) if latency else None,
                'latency_p95_ms':float(np.percentile(latency,95)) if latency else None,
                'latency_p99_ms':float(np.percentile(latency,99)) if latency else None,
                'raw_overwritten':self.raw.overwritten,'prepared_overwritten':self.prepared.overwritten,
                'stale_inputs_skipped':self.stale_inputs_skipped,
                'stale_prepared_skipped':self.stale_prepared_skipped,
                'error':self.error}

    def close(self):
        self.stop_event.set()
        self.prepared.clear()
        for thread in self.threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError('Worker did not stop; do not change background while it is active')
        self.prepared.clear()
        self.display.clear()
        self.diagnostic_display.clear()
