"""DirectShow camera controls, applied after format negotiation on each open."""
import math
import warnings
import cv2

DEFAULT_EXPOSURE = -4
DEFAULT_AUTO_WHITE_BALANCE = False
DEFAULT_FOURCC = 'MJPG'  # 'YUY2' or 'MJPG'; preserve the previously negotiated mode.


def configure_camera(cap, width=640, height=480, fps=30, fourcc=DEFAULT_FOURCC,
                     exposure=DEFAULT_EXPOSURE, auto_white_balance=DEFAULT_AUTO_WHITE_BALANCE):
    """Negotiate format first, then controls; return requested and actual values."""
    fourcc = fourcc.upper()
    if fourcc not in ('YUY2', 'MJPG'):
        raise ValueError(f'Camera format must be YUY2 or MJPG, got {fourcc!r}')
    # DShow may rebuild the graph when FPS/size changes. Set FOURCC last so
    # subsequent format negotiation cannot silently discard the chosen subtype.
    requests = [('width', cv2.CAP_PROP_FRAME_WIDTH, width),
                ('height', cv2.CAP_PROP_FRAME_HEIGHT, height),
                ('fps', cv2.CAP_PROP_FPS, fps),
                ('fourcc', cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))]
    settings = {name: bool(cap.set(prop, value)) for name, prop, value in requests}
    controls = apply_camera_controls(cap, exposure, auto_white_balance)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual = ''.join(chr((code >> (8*i)) & 255) for i in range(4))
    info = {'requested_fourcc': fourcc, 'requested_fps': fps, 'format_set_results': settings,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'reported_fps': cap.get(cv2.CAP_PROP_FPS), 'actual_fourcc': actual,
            'exposure': controls['exposure'], 'controls': controls}
    if (info['width'], info['height']) != (width, height):
        raise RuntimeError(f'Camera resolution does not match calibration: {info}')
    if actual != fourcc:
        warnings.warn(f'[Camera] Requested {fourcc}, driver negotiated {actual!r}', RuntimeWarning, stacklevel=2)
    return info


def apply_camera_controls(cap, exposure=DEFAULT_EXPOSURE,
                          auto_white_balance=DEFAULT_AUTO_WHITE_BALANCE):
    # DirectShow exposes white-balance temperature through BLUE_U. Disabling
    # AUTO_WB resets it to the driver default, so restore the preceding value.
    previous_wb = cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
    awb_ok = bool(cap.set(cv2.CAP_PROP_AUTO_WB, int(auto_white_balance)))
    restore_wb = None
    if not auto_white_balance and math.isfinite(previous_wb) and previous_wb > 0:
        restore_wb = bool(cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, previous_wb))
    # OpenCV's DShow AUTO_EXPOSURE getter is unsupported (-1). Do not treat it
    # as an auto/manual flag. Setting EXPOSURE is checked against driver readback;
    # tools/check_camera_settings.py additionally inspects IAMCameraControl flags.
    exposure_ok = bool(cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure)))
    actual_awb = cap.get(cv2.CAP_PROP_AUTO_WB)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    issues = []
    if actual_awb != int(auto_white_balance):
        issues.append(f'Auto white balance requested={auto_white_balance}, readback={actual_awb}')
    if not exposure_ok or actual_exposure != exposure:
        issues.append(f'Exposure requested={exposure}, set_ok={exposure_ok}, readback={actual_exposure}')
    if restore_wb is False:
        issues.append('Could not preserve the previous white-balance temperature')
    result = {
        'requested_exposure': exposure, 'requested_auto_white_balance': auto_white_balance,
        'exposure_set_ok': exposure_ok, 'auto_white_balance_set_ok': awb_ok,
        'white_balance_restore_ok': restore_wb, 'previous_white_balance': previous_wb,
        'exposure': actual_exposure, 'auto_white_balance': actual_awb,
        'white_balance': cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
        'warnings': issues,
    }
    for issue in issues:
        warnings.warn('[Camera] ' + issue, RuntimeWarning, stacklevel=2)
    return result


def run_baseline_with_camera_controls(reference, **kwargs):
    """Apply the same controls without editing the immutable baseline snapshot."""
    original_cv2 = reference.cv2

    class ConfiguredCV2:
        def __getattr__(self, name):
            return getattr(original_cv2, name)

        def VideoCapture(self, *args, **options):
            cap = original_cv2.VideoCapture(*args, **options)
            if cap.isOpened():
                try:
                    print('[Camera]', configure_camera(cap), flush=True)
                except Exception:
                    cap.release()
                    raise
            return cap

    # Replace only this module's reference; never monkey-patch the global cv2 API.
    reference.cv2 = ConfiguredCV2()
    try:
        return reference.main(**kwargs)
    finally:
        reference.cv2 = original_cv2
