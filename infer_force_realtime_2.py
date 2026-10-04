"""Force inference entry; --check-resources needs only the Python standard library."""
import argparse
from dataclasses import replace
import os
import sys

from runtime_resources import ROOT, Resources, require_resources

sys.dont_write_bytecode = True
RUNTIME_MODE = "optimized"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-resources", action="store_true", help="Check files/configuration without importing GPU packages or opening devices")
    parser.add_argument("--mode", choices=("optimized", "baseline"), default=os.environ.get("FORCE_RUNTIME_MODE", RUNTIME_MODE))
    parser.add_argument("--geo-version", type=int, default=Resources.geo_version)
    parser.add_argument("--force-version", type=int, default=Resources.force_version)
    parser.add_argument("--prior", default=Resources.prior, help="Project-relative boundary prior path")
    parser.add_argument("--preset", default=Resources.preset, help="Project-relative marker grid path")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.geo_version < 0 or args.force_version < 0:
        parser.error("Model versions must be nonnegative")
    if args.mode not in ("optimized", "baseline"):
        parser.error("FORCE_RUNTIME_MODE must be optimized or baseline")
    resources = Resources(args.geo_version, args.force_version, args.prior, args.preset)
    try:
        require_resources(resources)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.check_resources:
        print("Resource presence and configuration checks passed. Weights, numerical accuracy, GPU and hardware have not been validated.")
        return 0
    from optimized.bootstrap import configure, load_reference
    configure()
    if args.mode == "optimized":
        try:
            from optimized.core import Config, ForceEngine
            from optimized.app import run
            config = replace(Config(), geo_version=args.geo_version, force_version=args.force_version,
                             prior=args.prior, preset=args.preset, camera_index=args.camera_index)
            engine = ForceEngine(config)
        except Exception as exc:
            print(f"[Startup] Optimized runtime unavailable: {exc}")
            print("[Startup] Falling back to baseline. See README.md for dependencies and native extension build steps; 20 Hz is not guaranteed.")
        else:
            run(config, engine)
            return 0
    baseline = load_reference()
    baseline.PRESET_GRID_PATH = str(ROOT / args.preset)
    from optimized.camera import run_baseline_with_camera_controls
    run_baseline_with_camera_controls(baseline, camera_index=args.camera_index,
                                     model_version=args.geo_version, force_model_version=args.force_version,
                                     depth_prior_path=args.prior, save_dir="output_realtime",
                                     enable_force=True, pixmm=0.034)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
