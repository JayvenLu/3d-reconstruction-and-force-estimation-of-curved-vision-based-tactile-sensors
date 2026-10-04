# Geometry and Force Estimation for Curved Vision-Based Tactile Sensors

This source release contains the acquisition, calibration, geometry reconstruction, marker tracking, model training, and three-axis force inference pipeline for a curved vision-based tactile sensor. It includes the Python reference implementation and the optimized Python/C++ force runtime.

**No datasets, pretrained weights, calibration images, marker grids, depth priors, or CAD models are included.** The resource directories contain placeholders only. Training and live inference require resources produced with your own sensor.

## Contents

- Numbered root scripts: acquisition, camera/geometry calibration, and training.
- `dataset.py`, `model.py`: data preprocessing and neural network definitions.
- `geometry_*.py`, `depth_reconstruction_core.py`: versioned geometry processing and evaluation.
- `find_marker.py`, `marker_detection.py`, `RBF.py`: marker matching and displacement interpolation.
- `optimized/`: real-time force pipeline, visualization, camera controls, and native matching source.
- `baseline/infer_force_realtime_2.py`: the numerical reference used by the optimized pipeline and fallback runtime; it is a required dependency, not a disposable backup.
- `configs/geometry/`: example image-domain and gradient-domain experiment configurations.
- `tools/`: native builds, geometry experiment runners, and source-release checks.
- `runtime_resources.py`: dependency-free checks for missing runtime files and model metadata.

Run commands from this directory using an activated Python environment. Historical record IDs, model versions, sensor geometry, coordinate transforms, and calibration constants in the code are examples tied to the original setup. They must be reviewed for your sensor; renaming files does not recalibrate it.

## Installation

### Reference environment

The original optimized runtime used Windows x64, Python 3.11, PyTorch 2.5.1, torchvision 0.20.1, and CuPy 13.3.0 with CUDA 12.x. Dependency manifests use the recorded environment as a reference, with local wheel paths removed. They are not a fully tested cross-platform lockfile.

The optimized camera path uses Windows DirectShow. Acquisition contains Linux serial/camera support, and some offline geometry tools support CPU operation, but this release does not claim end-to-end Linux or CPU-only live-force support. The legacy reference imports CuPy as well; selecting baseline mode does not remove that dependency.

Create a separate environment, for example:

```powershell
conda create -n tactile-force python=3.11 -y
conda activate tactile-force
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch wheels first. The following is a CUDA 12.1 wheel example for the recorded PyTorch versions:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-cuda12.txt
python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt
python -m pip check
```

Use a compatible NVIDIA driver and CUDA toolkit/runtime for CuPy. Do not install multiple CuPy variants into one environment. Consult the [PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v251) and [CuPy 13.3 installation guide](https://docs.cupy.dev/en/v13.3.0/install.html) when choosing another CUDA configuration. PyTorch's CUDA wheel alone should not be assumed to supply every library needed by CuPy.

OpenCV windows, Pyrender, and Open3D need working display/OpenGL support. The optional `sparseqr` branch in `calculate_depth_single.py` requires a separate SuiteSparse/PySPQR installation; it is not required for the default boundary-prior reconstruction path and is not bundled.

### Native matching extension

On Windows, install a C++17-capable MSVC toolchain and Windows SDK, then use its developer terminal with the Python environment activated:

```powershell
python tools/build_native.py
```

This builds `optimized._matching` for the active Python version and platform. No prebuilt `.pyd` or vendored Python package is shipped. The build script also supplies non-Windows C++17 flags, but the live DirectShow pipeline remains Windows-specific. See [pybind11 build documentation](https://pybind11.readthedocs.io/en/stable/compiling.html).

The optional DirectShow driver-inspection extension can be built on Windows with:

```powershell
python tools/build_camera_driver.py
```

It is not required by the main force runtime. Rebuild native modules when changing Python version or platform.

## Required resources

Prepare resources locally under this project; none are retrieved from another checkout. Resource paths passed to the force entry must remain within this directory. Dataset and output folders are ignored by Git.

### Dataset layout

The training loaders use this layout, where `RECORD` is your recording ID and `ID` is the integer sample identifier:

```text
data/RECORD/
  train/
    force_log.csv
    images/ID_rgb.png
    images/ID_nir.png                 # only for NIR-enabled configurations
    images_removed/ID_rgb.png
    normal_gt/normalID.npy
    normal_complete_gt/normal_completeID.npy
    depth_gt/depthID.npy
    mask/maskID.npy
    depth_pred/depth_predID.npy       # or depth_pred_nir for NIR geometry
    dh/ID_dh.npy
  val/
    ... same layout ...
```

Sample `0` is the no-contact background. Keep its image, geometry, depth and displacement products consistent. Image arrays use OpenCV BGR order; normal maps have height-by-width-by-3 components. Depth maps are two-dimensional, with reconstruction depth in millimetres. Marker displacement files store horizontal image displacement in pixels, normally height-by-width-by-2; the force loader also accepts 2-by-height-by-width. The first displacement channel is image-horizontal and the second is image-vertical. The loaders apply the configured sign, scale and coordinate transformations.

The acquisition CSV header is:

```text
id,timestamp,x,y,z,fx,fy,fz,fh,phase,target,angle_deg,point_index,force_axes
```

`id` links a row to its image and arrays; `timestamp` records sample time; `x`, `y`, `z` are CNC positions in millimetres. `fx`, `fy`, `fz` are measured force components in newtons, and `fh` is the horizontal force magnitude. `phase` identifies the collection stage, `target` its force target, `angle_deg` the horizontal loading direction in degrees, and `point_index` the sampled grid point. `force_axes` specifies the force coordinate convention; the supplied collector writes `target_vts`. If it is absent, the loader assumes raw sensor axes and transforms them. Do not label raw forces as already transformed. Optional `tx`, `ty`, `tz` torque columns are supported by the loader but are not emitted by this collector.

Keep train/validation samples distinct. The geometry trainer audits duplicate images. The force loader can substitute zero horizontal displacement for missing `dh` files, so generate and inspect those files before force training rather than treating a successful launch as proof of complete inputs.

### Models and calibration

After training, retain each model's complete version bundle:

```text
logs/gel_experiment/version_N/
  checkpoints/*.ckpt
  geometry_config.json
  model.py
  3_training_pix2normal.py
  depth_prior.npy
  ... other source snapshots written by training ...
force_logs/HyperNet_Run/version_M/
  checkpoints/*.ckpt
  force_config.json
  code/
    3_training_dep2force_2.py
    dataset.py
    model.py
```

`N` and `M` are the versions generated by your runs. `force_config.json` is written by the force trainer in this release and records the effective preprocessing flags, including background subtraction. Geometry training already writes its JSON configuration, prior and source snapshots.

Legacy bundles may use saved training-script literals instead of JSON; force bundles must also contain the matching dataset snapshot so that omitted background-subtraction settings can be resolved from that exact version. Missing configuration causes an error. Offline geometry no longer silently guesses a root model when its version metadata is absent.

The force resource check also compares the runtime model/dataset definitions with their saved source snapshots. Use matching sources from the same training run; do not pair weights with another sensor's transforms. The check verifies presence and configuration, not checkpoint contents or numerical accuracy.

For a force model trained with displacement normalization, also retain its generated `data/RECORD/train/disp_norm_stats_*.npz` cache. This contains that training run's displacement mean and standard deviation; it is not included here. Missing statistics cause an error rather than silently disabling normalization.

Supply your own:

- CAD surface and calibration geometry under `mesh/`.
- Calibration images under `cam_calib/` and `cam/`.
- Perspective corners in `define_boundary_offline.py`, `calibrate_grid.py`, `offline_perspective_dfs_match.py`, and `baseline/infer_force_realtime_2.py`.
- Marker layout in `setting.py` and a generated `grid_preset_rgb_perspective.npy`.
- Boundary prior generated from the same CAD, camera parameters, resolution and coordinate system used for reconstruction.

Calibration scripts do not automatically propagate camera parameters or perspective corners into other scripts. Update all relevant consumers explicitly. Existing calibration paths refer to different historical sessions; align them to your own session before running the pipeline.

## Workflow

### 1 Collect synchronized images and force labels

Entry: `0_force_data_collection_KWR_unified.py`.

Configure `SERIAL_PORT`, `BAUD_RATE`, `SENSOR_IP`, `SENSOR_PORT`, camera settings, CNC axes, coordinate transforms, motion bounds and probing parameters at the top of the script. It controls the CNC and reads the force sensor; run it only on your configured apparatus. The supplied addresses and motion parameters describe an example setup.

```powershell
python 0_force_data_collection_KWR_unified.py
```

Output is a timestamped recording under `force_data/`, including images and `force_log.csv`. Organize your own recordings into the `data/RECORD/train` and `val` structure before the processing/training stages, preserving the matching IDs and backgrounds. Acquisition does not automatically create that train/validation split.

### 2 Calibrate the camera

Capture the calibration sleeve/target with `0_cap_img.py`. Its camera settings are at the top; `s` saves an image, `x` removes the last captured image, and `q` exits. Images are written under `cam/<timestamp>/`.

```powershell
python 0_cap_img.py
python 1_get_cam_param_auto.py
```

Before the second command, supply the target image (default `cam_calib/cam_calib.png`) and CAD geometry (default `mesh/sanzhi_curved.STL`). Review `ENABLE_AUTO_CALIBRATION`, target correspondences, camera position, orientation and intrinsics. The script supports point-based estimation and interactive rendering alignment. It prints the final parameters; copy the resulting values into the geometry-label and prior-generation scripts.

### 3 Generate geometric supervision

Entry: `2_normal_generate_auto.py`.

Set `DATA_ROOT`, `MESH_PATH`, camera parameters, probe geometry and CNC-to-render transforms. Inputs are the recorded images, CNC poses in `force_log.csv`, and your CAD surface. Interactive calibration aligns the rendered probe with the real image; per-sample manual refinement is available.

```powershell
python 2_normal_generate_auto.py
```

Outputs include `normal_gt`, `normal_complete_gt`, `depth_gt`, `mask`, and the manual calibration record. Repeat with the validation split. Use `offline_perspective_dfs_match.py` for the measured horizontal displacement fields used in force training.

### 4 Define perspective correction and marker grid

```powershell
python define_boundary_offline.py
python calibrate_grid.py
```

Set `IMAGE_PATH` in the first script. Drag the four boundary corners, use `p` to print their coordinates, and `s` to save the perspective-corrected image. Transfer those corners to the consumer scripts listed under Models and calibration.

In `calibrate_grid.py`, configure `IMG_PATH`, `PRESET_PATH`, the perspective corners and `setting.py` marker dimensions. Review/edit the marker correspondence and save the grid preset. Reuse this same preset in offline matching and live inference.

### 5 Remove markers from images

Entry: `gel_marker_inpaint_viewer.py`.

```powershell
python gel_marker_inpaint_viewer.py --data-root data/RECORD/train --batch
python gel_marker_inpaint_viewer.py --data-root data/RECORD/val --batch
```

This detects/inpaints markers and writes `images_removed`. Inspect detector settings and output quality. Available controls include `--scope`, `--start-id`, `--block-size` and `--c`. The current geometry trainer can also perform configured completion from raw images; use the route in its saved configuration consistently rather than applying unrecorded preprocessing twice.

### 6 Generate the boundary depth prior

Entry: `get_depth_prior.py`. This was step 8 in the original procedure; it must precede current geometry training and depth reconstruction.

Set camera pose/intrinsics and choose render and inner reconstruction dimensions to match your sensor. The script's current default is a historical 128-by-128 setup, whereas the supplied force runtime expects 640-by-480 input. Do not reuse those small-resolution camera intrinsics for the live pipeline.

After setting camera parameters for a 640-by-480 setup, an example invocation is:

```powershell
python get_depth_prior.py --mesh-path mesh/sanzhi_bg.STL --render-width 640 --render-height 480 --inner-width 630 --inner-height 466 --depth-output-path depth_prior/rendered_depth.npy --prior-output-path depth_prior/boundary.npy --no-show-point-cloud
```

The 630-by-466 inner region is a historical example, not a universal sensor setting. The command generates the rendered depth and a boundary prior; inspect their geometry, units and alignment before use.

### 7 Train the normal prediction model

Entry: `3_training_pix2normal.py`.

```powershell
python 3_training_pix2normal.py --config configs/geometry/image_telea.json --record RECORD --prior depth_prior/boundary.npy --device cuda
```

The trainer uses the record's train/validation splits, geometry supervision and boundary prior. Adjust the selected JSON's input dimensions, network, detector and training settings for your experiment. Outputs are saved under `logs/gel_experiment/version_N/`, including checkpoints, configuration, prior, source snapshots and training results. Record the actual version assigned by the logger.

Image-domain and gradient-domain completion variants are available in `configs/geometry/`. The supplied live runtime remains the image-domain `Modelv2_ori` implementation; other routes are supported by offline geometry evaluation, not automatically by live force inference.

### 8 Inspect geometry inference

For live geometry, set `model_idx` and the camera/inpainting settings in `infer_geo_realtime.py` to match the trained model, then run:

```powershell
python infer_geo_realtime.py
```

Use `b` for a no-contact background, `s` to save a result and `q` to quit. This script primarily previews predicted normals; use the following depth stage or `GeometryPipeline` for reconstructed depth. Its CUDA calls and display requirements remain unchanged.

For offline geometry evaluation, after preparing a trained version and validation data:

```powershell
python geometry_evaluation.py --versions 0 --record RECORD --split val --prior depth_prior/boundary.npy --device cuda --output reports/my_geometry
```

Replace `0` with your actual version. `tools/run_geometry_experiments.py` and `tools/run_geometry_evaluation.py` provide batch experiment and re-evaluation entry points; their `--help` lists the available options. They generate new reports locally; no historical reports are shipped.

### 9 Reconstruct sample depth maps

Entry: `calculate_depth_offline.py`.

Set `data_folder`, `model_version`, `depth_prior_path`, NIR usage and visualization options in its main block, then run:

```powershell
python calculate_depth_offline.py
```

Inputs are the sample images/background, matched geometry model and boundary prior. The output used by force training is `depth_pred/depth_predID.npy` (or `depth_pred_nir` when enabled). Process both splits and include the background sample. `calculate_depth_single.py` supplies additional integration utilities and is retained as a dependency.

### 10 Compute horizontal displacement fields

Entry: `offline_perspective_dfs_match.py`.

```powershell
python offline_perspective_dfs_match.py --image-dir data/RECORD/train/images --preset cam/SESSION/grid_preset_rgb_perspective.npy --auto
python offline_perspective_dfs_match.py --image-dir data/RECORD/val/images --preset cam/SESSION/grid_preset_rgb_perspective.npy --auto
```

Set matching geometry and detection parameters in the script and `setting.py`. The script detects markers, matches them to the calibrated grid and interpolates the horizontal field, producing `dh/ID_dh.npy`. Automatic processing can enter manual review for timeout cases. Omitting `--auto` uses interactive mode. `SESSION` is the directory holding your calibrated grid.

### 11 Train the force estimation model

Entry: `3_training_dep2force_2.py`.

Set the top-level `record`, `pixmm`, `use_nir`, `lag_dz`, `pos_dim`, `depth_source`, normalization, coordinate-conversion and training parameters. Inputs include both splits' CSV force labels, depth/background maps and displacement fields.

```powershell
python 3_training_dep2force_2.py
```

The trainer writes `force_logs/HyperNet_Run/version_M/`, with checkpoints, `force_config.json` and `code/` snapshots. Keep the bundle together. The optimized runtime accepts the original preprocessing combination: `lag_dz`, `normalize_disp`, `convert_dz_to_pixel`, and `convert_disp_to_sensor` are false, while `use_bg_sub` is true. Other combinations require the reference path and separate validation; do not change flags just to bypass the check.

The model is supervised by global forces. Its local predicted force distribution is not an independently validated local pressure or per-pixel force measurement.

### 12 Run three-axis force inference

First run the dependency-free check. Replace the example versions and paths with your own:

```powershell
python infer_force_realtime_2.py --check-resources --geo-version 0 --force-version 0 --prior depth_prior/boundary.npy --preset cam/SESSION/grid_preset_rgb_perspective.npy
```

A fresh source release intentionally exits with status 2 and a missing-resource list. The check does not import PyTorch/CuPy/OpenCV, open devices, read training samples, or validate binary weights. A successful check is a prerequisite, not a runtime accuracy certification.

Then start the same selection:

```powershell
python infer_force_realtime_2.py --geo-version 0 --force-version 0 --prior depth_prior/boundary.npy --preset cam/SESSION/grid_preset_rgb_perspective.npy --camera-index 0
```

`run_force.cmd` forwards the same arguments using `python` from the active environment. There is no automatic switch to a maintainer's Conda installation. `--mode baseline` explicitly chooses the reference path; `FORCE_RUNTIME_MODE` remains supported. Initialization failure in the optimized path prints its reason before attempting the reference implementation. Runtime worker/device failures are not silently replaced by stale results.

Both paths validate resources before opening the camera. The supplied model definitions must match their archived model/dataset sources. Coordinate calibration, detector parameters and image preprocessing still require validation for your sensor, even when the file check passes.

## Live controls and outputs

Current defaults in `optimized/camera.py` request **MJPG**, exposure **-4**, and disabled automatic white balance; the existing white-balance temperature is preserved when possible. The pipeline requests 640-by-480 at 30 FPS on camera 0 via DirectShow. Actual negotiated mode and measured output rate depend on the device. Recalibrate the no-contact background after changing imaging conditions.

- `b`: acquire/reset the no-contact background.
- `Enter`: asynchronously save the currently displayed result under `force_logs/HyperNet_Run/version_M/<timestamp>/`.
- `q`: exit.
- `d`: toggle diagnostic views, refreshed at a lower rate.
- `,` / `.`: decrease/increase force-arrow scale.
- `[` / `]`, `-` / `=`, `n` / `m`: tune marker threshold window, threshold constant and circularity.

The main view draws sensor force components: positive `Fx` points left, positive `Fy` down, and positive `Fz` controls color. `Fx`, `Fy`, `Fz` are the three force components in the original sensor convention, measured in newtons. Negative `Fz` is clipped only for coloring; original values and summed force remain signed. Saved images, arrays and `result.json` share a frame ID. Main updates follow completed force results, while diagnostics alone are rate-limited. Session timing reports are generated under `reports/`.

## Troubleshooting and limitations

- **Missing resources:** follow the listed paths and generation stages. Weights alone are insufficient; keep model configuration and source snapshots. Historical version numbers do not imply included pretrained models.
- **Source/configuration mismatch:** use definitions and coordinate conventions from the same training run. Do not remove the check or substitute a different experiment's metadata.
- **`optimized._matching` import failure:** activate the intended Python environment, install build requirements, and rebuild. Binaries from another Python version are not interchangeable.
- **CUDA or CuPy error:** check the NVIDIA driver, CUDA runtime/toolkit and installed CuPy variant. Baseline selection is not a general CPU-only deployment solution.
- **Camera failure or wrong dimensions:** verify the camera index, DirectShow support, format negotiation and resolution. Revisit all dependent calibrations when resolution changes.
- **Rendering failure:** check the OpenGL/display environment and CAD paths. Image-based rendering calibration requires actual target images and meshes.
- **Missing historical performance reports:** this distribution intentionally excludes measurements and outputs. Benchmark your own hardware and data. Host-side frame timings do not measure the full exposure-to-controller latency.
- **License:** no project license has been selected or added in this preparation. There is no bundled dataset or pretrained model license grant.

## Source release checks

The following checks need only Python and do not connect to hardware:

```powershell
python -B tools/test_release.py
python -B infer_force_realtime_2.py --help
python -B infer_force_realtime_2.py --check-resources
```

The last command is expected to fail until you supply resources. Tests create temporary synthetic file/configuration fixtures outside the release and remove them afterwards; these are not sample sensor data. Source syntax, configuration parsing and missing-resource behavior can be checked without performing training or inference.

This source packaging does not constitute new validation of model accuracy, native extension performance, CUDA execution, CNC operation or live sensing.
