"""Sensor component display: +Fx left, +Fy down, positive Fz encoded by color."""
from functools import lru_cache
import importlib.util
import time
import collections
import cv2
import numpy as np
from .bootstrap import ROOT

WINDOW = 'Sensor XY Force on Camera'


@lru_cache(maxsize=2)
def sensor_to_pixel(version=76):
    # Read the convention stored with the trained model, not a guessed screen axis.
    path = ROOT/'force_logs'/'HyperNet_Run'/f'version_{version}'/'code'/'dataset.py'
    spec = importlib.util.spec_from_file_location(f'_overlay_axes_{version}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return np.asarray(module.PIXEL_TO_SENSOR_ROT, dtype=np.float64).T


def image_force(vectors, version=76):
    """Display components in stored CAD image axes; leave source force data intact."""
    return np.asarray(vectors, dtype=np.float64) @ sensor_to_pixel(version).T


def project_image(vectors):
    """Image-axis vectors to screen offsets: +X right, +Y down, +Z up-left."""
    pixel = np.asarray(vectors, dtype=np.float64)
    return np.stack((pixel[...,0]-.7*pixel[...,2], pixel[...,1]-.7*pixel[...,2]), axis=-1)


def project(vectors, version=76):
    pixel = image_force(vectors, version)
    return project_image(pixel), pixel


def arrow_geometry(preview, shape, config, scale):
    """No CAD rotation, depth projection or per-arrow length normalization."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Arrow scale must be positive and finite')
    _, rows, cols = preview.shape
    height, width = shape[:2]
    yy,xx = np.meshgrid((np.arange(rows)+.5)*height/rows-.5,
                        (np.arange(cols)+.5)*width/cols-.5,indexing='ij')
    starts = np.stack((xx.ravel(),yy.ravel()),axis=1)
    forces = preview.reshape(3,-1).T
    valid = np.isfinite(forces).all(axis=1)
    xy = np.linalg.norm(np.where(valid[:,None],forces[:,:2],0),axis=1)
    keep = valid & (xy >= config.arrow_min_force) & (xy * scale >= 2)
    starts,forces = starts[keep],forces[keep]
    delta = forces[:,:2].astype(np.float64) * np.array([-scale,scale])
    return starts, delta, delta.copy(), forces


def force_colors(fz, maximum):
    """Fixed light-to-dark orange scale; negative values affect color only as zero."""
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('Fz color maximum must be positive and finite')
    level = np.clip(np.asarray(fz,dtype=np.float64),0,maximum)/maximum
    light = np.array([235.,245.,255.])  # BGR; also the zero/negative color.
    dark = np.array([20.,60.,125.])
    return np.rint(light+(dark-light)*level[...,None]).astype(np.uint8)


def visible_endpoint(origin, delta, width, height):
    """Clip only at the viewport, retaining direction; no force magnitude cap."""
    factor = 1.0
    for coordinate,change,limit in zip(origin,delta,(width-1,height-1)):
        if change > 0: factor = min(factor,(limit-coordinate)/change)
        elif change < 0: factor = min(factor,-coordinate/change)
    endpoint = np.rint(np.asarray(origin)+max(0,factor)*delta).astype(int)
    return tuple(endpoint), factor < 1.0


def draw_overlay(result, summary, config, scale=None, stale=False):
    scale = config.arrow_scale if scale is None else scale
    raw = result.prepared.raw
    vis = raw.copy(); h,w = vis.shape[:2]
    active = result.valid and result.preview is not None and not stale
    count = 0; clipped_count = 0
    if active:
        starts,delta,planar,forces = arrow_geometry(result.preview,raw.shape,config,scale)
        count = len(starts)
        origins=np.rint(starts).astype(int).tolist()
        colors=force_colors(forces[:,2],config.normal_color_max).tolist()
        for origin,offset,color in zip(origins,delta,colors):
            a=tuple(origin);color=tuple(color)
            b,clipped = visible_endpoint(origin,offset,w,h)
            if clipped:
                clipped_count += 1
                cv2.line(vis,a,b,(15,15,15),3,cv2.LINE_AA)
                cv2.line(vis,a,b,color,1,cv2.LINE_AA)
                cv2.drawMarker(vis,b,color,cv2.MARKER_DIAMOND,9,2,cv2.LINE_AA)
            else:
                tip = min(.35,7/max(np.linalg.norm(offset),1))
                cv2.arrowedLine(vis,a,b,(15,15,15),3,cv2.LINE_AA,tipLength=tip)
                cv2.arrowedLine(vis,a,b,color,1,cv2.LINE_AA,tipLength=tip)
    top = np.zeros((60,w,3),np.uint8);bottom = np.zeros((112,w,3),np.uint8)
    def put(image,text,xy,color=(230,230,230),size=.43):
        cv2.putText(image,text,xy,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
    put(top,'MODEL CONTRIBUTIONS | SENSOR AXES | XY arrows / Fz color',(10,18),size=.45)
    if active:
        fx,fy,fz = result.force
        put(top,f'Fx {fx:+.4f} N   Fy {fy:+.4f} N   Fz {fz:+.4f} N',(10,39),size=.53)
    else:
        label = 'STALE: no new result for 500 ms' if stale else 'INVALID: '+result.reason[:60]
        put(top,label,(10,39),(70,100,255),.5)
    put(bottom,f"Frame {result.frame_id} | camera {summary.get('capture_hz',0):.1f} | force {summary.get('valid_hz',0):.1f} | view {summary.get('display_hz',0):.1f} Hz",(10,16))
    put(bottom,f"Force ready {result.metrics.get('latency_ms',0):.1f} ms | view submit p95 {summary.get('display_latency_p95_ms',0) or 0:.1f} ms",(10,33))
    put(bottom,f'{count}/192 arrows | {scale:.0f} px/N | {clipped_count} edge-clipped',(10,50))
    put(bottom,'b: background   d: diagnostics   ,/.: scale   Enter: save   q: quit',(10,68),size=.38)
    # Sensor component display convention, not a calibrated image-plane projection.
    basis = np.array([[-1,0],[0,1]])
    origin = np.array([w-47,37])
    for direction,label,color in zip(basis,('+Fx','+Fy'),((80,100,255),(80,240,100))):
        tip = np.rint(origin+direction*21).astype(int)
        cv2.arrowedLine(bottom,tuple(origin),tuple(tip),color,1,cv2.LINE_AA,tipLength=.25)
        put(bottom,label,tuple(tip+np.array([2,-2])),color,.35)
    # Positive regional Fz only; raw signed Fz remains visible in the total readout.
    bar = force_colors(np.linspace(0,config.normal_color_max,180),config.normal_color_max)
    bottom[80:92,10:190] = bar[None,:,:]
    put(bottom,'0',(10,107),size=.36)
    put(bottom,f'>={config.normal_color_max:.3f} N/region | darker = larger +Fz; Fz<0: color=0',
        (150,107),size=.36)
    # A literal length scale, independent of per-frame maxima.
    length = int(min(40,.02*scale));left = w-110
    cv2.line(top,(left,51),(left+length,51),(240,240,240),1)
    put(top,f'{length/scale:.3f} N',(left,43),size=.33)
    return np.vstack((top,vis,bottom))


class DisplayMetrics:
    """Count unique native-window submissions, never GUI pump iterations."""
    def __init__(self):
        self.records = collections.deque(maxlen=100000)
        self.last_id = None

    def submitted(self,result,render_ms,submitted_at=None):
        identity = (result.generation,result.frame_id)
        if self.last_id is not None and identity <= self.last_id:return
        self.last_id = identity
        at = time.perf_counter() if submitted_at is None else submitted_at
        self.records.append({'frame_id':result.frame_id,'generation':result.generation,
            'captured_at':result.captured_at,'submitted_at':at,'valid':result.valid,
            'display_latency_ms':(at-result.captured_at)*1000,'render_ms':render_ms,
            'preview_ms':result.metrics.get('preview_ms',0)})

    def summary(self,since=None):
        rows = [r for r in self.records if r['valid'] and (since is None or r['submitted_at']>=since)]
        duration = rows[-1]['submitted_at']-rows[0]['submitted_at'] if len(rows)>1 else 0
        return {'displayed':len(rows),'display_hz':(len(rows)-1)/duration if duration>0 else 0,
            'display_latency_p95_ms':float(np.percentile([r['display_latency_ms'] for r in rows],95)) if rows else None,
            'preview_and_render_p95_ms':float(np.percentile([r['render_ms']+r['preview_ms'] for r in rows],95)) if rows else None}
