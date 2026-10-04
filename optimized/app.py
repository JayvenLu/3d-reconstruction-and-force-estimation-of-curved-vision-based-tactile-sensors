"""OpenCV UI. All work for a displayed/saved result uses that result's frame ID."""
from dataclasses import replace
from pathlib import Path
import datetime
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import matplotlib
# Snapshot figures are saved by a worker. A GUI backend such as TkAgg is not
# thread-safe and is unnecessary: all interactive windows belong to OpenCV.
matplotlib.use('Agg')
from .bootstrap import ROOT, load_reference
from .core import Config, ForceEngine, ImagePreparer, MarkerTracker
from .pipeline import CameraCapture, Pipeline
from .overlay import WINDOW, draw_overlay, DisplayMetrics


def draw_flow(result):
    image = result.prepared.raw.copy()
    for a, b in zip(result.prepared.match.source, result.prepared.match.destination):
        cv2.arrowedLine(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                       (0,255,0), 1, tipLength=.25)
    return image


def render(result, summary, config=Config(), scale=None, stale=False):
    image=draw_overlay(result,summary,config,scale,stale)
    cv2.imshow(WINDOW,image)
    return image


DIAGNOSTIC_WINDOWS=('Camera & Geometry','Marker Mask','Inpainted Frame','Depth Diff (dz)',
    'Displacement dx (no-bar)','Displacement dy (no-bar)','Force Estimation','Marker Flow on Raw')


def close_diagnostics():
    for name in DIAGNOSTIC_WINDOWS:
        try:cv2.destroyWindow(name)
        except cv2.error:pass


def render_diagnostics(result, summary):
    ref = load_reference()
    raw = cv2.resize(result.prepared.raw, (384,288))
    if not result.valid:
        cv2.putText(raw, 'INVALID: '+result.reason[:45], (8,35), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        cv2.imshow('Force Estimation', raw)
        # Never leave old force values visibly presented as a current result.
        return
    arr = result.arrays
    if arr is None:
        return
    size=(384,288)
    norm = ref.visualize_normal(arr['normal'], size)
    depth = ref.visualize_map_with_bar(arr['depth'], size, cv2.COLORMAP_JET, unit='mm')
    cv2.imshow('Camera & Geometry', np.hstack([raw,norm,depth]))
    cv2.imshow('Marker Mask', cv2.resize(result.prepared.mask,size))
    cv2.imshow('Inpainted Frame', cv2.resize(result.prepared.clean,size))
    cv2.imshow('Depth Diff (dz)', ref.visualize_map_with_bar(arr['dz'],size,cv2.COLORMAP_JET,unit='mm'))
    dx,dy = ref.visualize_dx_dy_colormaps(arr['dx'],arr['dy'],size)
    cv2.imshow('Displacement dx (no-bar)',dx)
    cv2.imshow('Displacement dy (no-bar)',dy)
    force = ref.visualize_force_component_maps(*arr['force_map'],out_size=size)
    step=force.shape[1]//3
    for i, name in enumerate(('Fx','Fy','Fz')):
        cv2.putText(force,f'{name}: {result.force[i]:.4f} N',(i*step+10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    text=f"frame {result.frame_id} | {summary['valid_hz']:.1f} Hz | {result.metrics['latency_ms']:.1f} ms"
    cv2.putText(force,text,(10,force.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    cv2.imshow('Force Estimation',force)
    cv2.imshow('Marker Flow on Raw',cv2.resize(draw_flow(result),size))


def save_snapshot(result, config, overlay_image=None, arrow_scale=None):
    if not result.valid:
        raise ValueError('A current valid displayed result is required for saving')
    result=replace(result,arrays=result.materialize_arrays())
    ref=load_reference()
    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder=ROOT/'force_logs'/'HyperNet_Run'/f'version_{config.force_version}'/ts
    folder.mkdir(parents=True)
    a=result.arrays;size=(config.width,config.height)
    if overlay_image is None:overlay_image=draw_overlay(result,{},config,arrow_scale)
    images={'raw':result.prepared.raw,'img_clean':result.prepared.clean,
            'force_arrows_3d':overlay_image,
            'normal':ref.visualize_normal(a['normal'],size),
            'depth':ref.visualize_map_core_only(a['depth'],size,cv2.COLORMAP_JET),
            'dz_map':ref.visualize_map_core_only(a['dz'],size,cv2.COLORMAP_JET),
            'marker_flow_raw':draw_flow(result),
            'disp_field':ref.visualize_displacement_arrows(result.prepared.clean,a['dx'],a['dy'],a['dz'],step=25),
            'force_components':ref.visualize_force_component_maps(*a['force_map'],out_size=size)}
    images['dx_map'],images['dy_map']=ref.visualize_dx_dy_colormaps(a['dx'],a['dy'],size)
    for name, component in zip(('Fx','Fy','Fz'),a['force_map']):
        images[name+'_map']=ref.visualize_map_core_only(component,size,cv2.COLORMAP_INFERNO)
    for name, image in images.items():
        if not cv2.imwrite(str(folder/f'{name}_{ts}.png'),image):
            raise IOError(f'Could not save {name}')
    for name,array in a.items():
        np.save(folder/f'{name}_{ts}.npy',array)
    # Preserve the original snapshot filenames for existing downstream consumers.
    np.save(folder/f'force_{ts}.npy',a['force_map'])
    np.save(folder/f'depth_arr_{ts}.npy',a['depth'])
    (folder/'result.json').write_text(json.dumps({'frame_id':result.frame_id,'generation':result.generation,
        'captured_at':result.captured_at,'output_at':result.output_at,'force_N':result.force.tolist(),
        'force_axes':'target_vts',
        'display_force_N':result.force.tolist(),
        'display_force_axes':'target_vts',
        'metrics':result.metrics,'arrow_display':{'scale_px_per_N':config.arrow_scale if arrow_scale is None else arrow_scale,
        'min_xy_force_N':config.arrow_min_force,'max_length_px':None,
        'dots_enabled':False,'min_arrow_length_px':2,
        'grid':[12,16],'coordinates':'sensor components: screen dx=-scale*Fx, dy=scale*Fy',
        'readout_axes':'target_vts','legend_axes':'sensor +Fx left, +Fy down',
        'color_component':'regional sensor Fz, negative clipped to zero for color only',
        'color_max_N':config.normal_color_max,'color_bgr_zero':[235,245,255],
        'color_bgr_max':[20,60,125],'viewport_clipping':'diamond at boundary; no per-arrow length cap'}},indent=2),encoding='utf-8')
    # Full-resolution 3D plot is produced on demand, never on the decoding path.
    full_force=[cv2.resize(x,size) for x in a['force_map']]
    try:
        ref.visualize_3d_force_distribution(a['depth'],*full_force,step=5,scale_arrow=300.0,
                                          save_path=str(folder/f'force_3d_{ts}.png'))
    except Exception as exc:
        print(f'[Snapshot] 3D plot unavailable: {exc}')
    return folder


class SnapshotSaver:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='snapshot')
        self.future=None

    def submit(self,result,config,image,scale):
        self.poll()
        if self.future is not None:
            print('[Snapshot] Busy; current request skipped');return False
        if not result.valid:
            print('[Snapshot] No valid displayed frame');return False
        self.future=self.pool.submit(save_snapshot,result,config,image.copy(),scale)
        return True

    def poll(self):
        if self.future is not None and self.future.done():
            try:print('[Snapshot]',self.future.result())
            except Exception as exc:print('[Snapshot] Failed:',exc)
            self.future=None

    def close(self):
        self.pool.shutdown(wait=True);self.poll()


def run(config=Config(), engine=None, acceptance_frames=0):
    cv2.setNumThreads(config.opencv_threads)
    engine=engine or ForceEngine(config)
    tracker=MarkerTracker(config)
    preparer=ImagePreparer(config,tracker)
    capture=None;pipeline=None;last=None;last_image=None
    saver=SnapshotSaver();display=DisplayMetrics();diagnostics=False
    scale=config.arrow_scale;rendered_scale=scale;summary={};last_received=0.;stale_shown=False
    final_report=None
    try:
        capture=CameraCapture(config)
        print('[Camera]',capture.info)
        print("Controls: b background | d diagnostics | , . arrow scale | Enter save | [ ] block | - = threshold | n m circularity | q quit")
        preview_id=-1;next_report=time.perf_counter()+2;next_stats=0
        while True:
            now=time.perf_counter()
            saver.poll();submitted=None;draw_ms=0
            if pipeline is None:
                item=capture.peek()
                if item is not None and item[0]!=preview_id:
                    preview=item[2].copy()
                    cv2.putText(preview,"Press b with NO CONTACT to set background",(12,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
                    cv2.imshow(WINDOW,preview);preview_id=item[0]
            else:
                if pipeline.error:
                    raise RuntimeError(pipeline.error)
                if now>=next_stats:
                    summary={**pipeline.summary(since=now-5),**display.summary(since=now-5),
                             'capture_hz':capture.rate(since=now-5)}
                    next_stats=now+.5
                result=pipeline.display.take(timeout=0)
                if result is not None:
                    last=result
                    start_draw=time.perf_counter()
                    last_image=render(result,summary,config,scale)
                    rendered_scale=scale
                    draw_ms=(time.perf_counter()-start_draw)*1000
                    submitted=result;last_received=time.perf_counter();stale_shown=False
                elif last is not None and now-last_received>.5 and not stale_shown:
                    last_image=render(last,summary,config,scale,stale=True);stale_shown=True
                if diagnostics:
                    diagnostic=pipeline.diagnostic_display.take(timeout=0)
                    if diagnostic is not None:render_diagnostics(diagnostic,summary)
                if now>=next_report:
                    print('[Profiler]',json.dumps(summary),flush=True)
                    next_report=now+2
            key=cv2.pollKey()&255
            if submitted is not None:display.submitted(submitted,draw_ms)
            if acceptance_frames and pipeline and len(pipeline.snapshot_records())>=acceptance_frames+30:
                break
            if submitted is None:time.sleep(.001)
            if key==ord('q'):
                break
            if key==ord('b'):
                item=capture.peek()
                if item is None:
                    continue
                if pipeline:
                    pipeline.close();pipeline=None
                last=None;last_image=None;display=DisplayMetrics();summary={}
                cv2.destroyAllWindows()
                # Build a fresh tracker; partially failed recalibration cannot retain old state.
                new_tracker=MarkerTracker(config)
                with preparer.lock:
                    params=replace(preparer.params)
                match=new_tracker.match(item[2],params,background=True)
                if not match.valid:
                    print('[Background] Calibration failed:',match.reason);continue
                clean,_=preparer.inpaint(item[2])
                engine.set_background(clean,new_tracker.source_raw,engine.generation+1)
                preparer.tracker=new_tracker
                pipeline=Pipeline(engine,preparer,capture.slot)
                pipeline.set_diagnostics(diagnostics)
                pipeline.start()
                cv2.destroyAllWindows()
                print('[Background] Calibrated; generation',engine.generation)
            elif key==ord('d'):
                diagnostics=not diagnostics
                if pipeline:pipeline.set_diagnostics(diagnostics)
                if not diagnostics:close_diagnostics()
                print('[Diagnostics]',diagnostics)
            elif key in (ord(','),ord('.')):
                scale=float(np.clip(scale*(.8 if key==ord(',') else 1.25),50,10000))
                print('[Arrows] px/N:',scale)
            elif key in (10,13) and last is not None and last_image is not None and not stale_shown:
                saver.submit(last,config,last_image,rendered_scale)
            elif key in map(ord,'[]-=nm'):
                with preparer.lock:
                    p=preparer.params
                    if key==ord('['):p.block_size=max(3,p.block_size-2)
                    elif key==ord(']'):p.block_size+=2
                    elif key==ord('-'):p.c=max(0,p.c-1)
                    elif key==ord('='):p.c+=1
                    elif key==ord('n'):p.circularity=min(1.,p.circularity+.01)
                    elif key==ord('m'):p.circularity=max(0.,p.circularity-.01)
                    print('[Detection]',vars(p))
    finally:
        if pipeline:
            pipeline.close()
            records=pipeline.snapshot_records()
            if acceptance_frames:
                records=records[30:30+acceptance_frames]
                ids={r['frame_id'] for r in records}
                display.records=type(display.records)((r for r in display.records if r['frame_id'] in ids),maxlen=100000)
                with pipeline.record_lock:
                    pipeline.records.clear();pipeline.records.extend(records)
            capture_hz=capture.rate(since=records[0]['captured_at'] if records else None)
            final_report={'summary':{**pipeline.summary(),**display.summary(),'capture_hz':capture_hz},
                'records':records,'display_records':list(display.records),
                'display_timing':'after imshow and pollKey return; not optical screen latency',
                'includes_native_window_presentation':True,'camera':capture.info}
            if acceptance_frames:
                s=final_report['summary']
                final_report['pass']=bool(len(records)>=acceptance_frames and s['valid_hz']>=.95*capture_hz
                    and s['display_hz']>=.95*capture_hz and (s['display_latency_p95_ms'] or float('inf'))<=100
                    and (s['preview_and_render_p95_ms'] or float('inf'))<=5)
            name='camera_overlay_acceptance.json' if acceptance_frames else 'session_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'
            report=ROOT/'reports'/name
            report.write_text(json.dumps(final_report,indent=2),encoding='utf-8')
        saver.close()
        preparer.close()
        if capture:
            capture.close()
        cv2.destroyAllWindows()
    return final_report
