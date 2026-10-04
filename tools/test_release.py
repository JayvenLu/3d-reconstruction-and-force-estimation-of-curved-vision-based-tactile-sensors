"""Source-only checks. Synthetic fixtures stay in TemporaryDirectory; no device imports."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from runtime_resources import Resources, resource_errors, geometry_settings, force_settings


class SourceReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tactile-release-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selection = Resources(1, 2, "depth_prior/example.npy", "cam/example.npy")

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def complete_fixture(self):
        # These nonempty text files test presence only. They are never loaded as arrays/weights.
        for path in (self.selection.prior, self.selection.preset,
                     "logs/gel_experiment/version_1/checkpoints/example.ckpt",
                     "force_logs/HyperNet_Run/version_2/checkpoints/example.ckpt"):
            self.write(path, "synthetic presence fixture, not a trained model")
        for path in ("model.py", "logs/gel_experiment/version_1/model.py",
                     "force_logs/HyperNet_Run/version_2/code/model.py"):
            self.write(path, "EXAMPLE = 1\n")
        dataset = "class Dataset_gel_force_dynamic_weight:\n    def __init__(self, use_bg_sub=True):\n        pass\n"
        self.write("dataset.py", dataset)
        self.write("force_logs/HyperNet_Run/version_2/code/dataset.py", dataset)
        cfg = json.loads((ROOT / "configs/geometry/image_telea.json").read_text())
        self.write("logs/gel_experiment/version_1/geometry_config.json", json.dumps(cfg))
        self.force_cfg = dict(record="example", pixmm=.034, use_nir=False, lag_dz=False,
                              pos_dim=3, depth_source="pred", normalize_disp=False,
                              convert_dz_to_pixel=False, convert_disp_to_sensor=False, use_bg_sub=True)
        self.write("force_logs/HyperNet_Run/version_2/force_config.json", json.dumps(self.force_cfg))

    def test_all_python_syntax_and_json(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for path in (ROOT / "configs").rglob("*.json"):
            self.assertIsInstance(json.loads(path.read_text()), dict)

    def test_empty_release_lists_models_and_calibration(self):
        errors = "\n".join(resource_errors(self.selection, self.root))
        for expected in ("boundary depth prior", "marker grid", "geometry configuration",
                         "force configuration", "checkpoints", "dataset.py"):
            self.assertIn(expected, errors)

    def test_matching_bundle_passes_presence_checks(self):
        self.complete_fixture()
        self.assertEqual(resource_errors(self.selection, self.root), [])

    def test_missing_configuration_is_not_guessed(self):
        self.complete_fixture()
        (self.root / "force_logs/HyperNet_Run/version_2/force_config.json").unlink()
        self.assertTrue(any("Missing force configuration" in e for e in resource_errors(self.selection, self.root)))
        self.write("force_logs/HyperNet_Run/version_2/force_config.json", '{"record":"example"}')
        self.assertTrue(any("Incomplete model configuration" in e for e in resource_errors(self.selection, self.root)))

    def test_legacy_defaults_come_from_saved_dataset(self):
        self.complete_fixture()
        (self.root / "force_logs/HyperNet_Run/version_2/force_config.json").unlink()
        values = {k: v for k, v in self.force_cfg.items() if k != "use_bg_sub"}
        self.write("force_logs/HyperNet_Run/version_2/code/3_training_dep2force_2.py",
                   "\n".join(f"{k} = {v!r}" for k, v in values.items()))
        self.assertTrue(force_settings(2, self.root)["use_bg_sub"])
        # Removing the version's dataset must not borrow a default from the root dataset.
        (self.root / "force_logs/HyperNet_Run/version_2/code/dataset.py").unlink()
        with self.assertRaisesRegex(ValueError, "use_bg_sub"):
            force_settings(2, self.root)

    def test_wrong_model_sources_and_routes_are_rejected(self):
        self.complete_fixture()
        self.write("model.py", "EXAMPLE = 2\n")
        self.assertTrue(any("Source mismatch" in e for e in resource_errors(self.selection, self.root)))
        path = self.root / "logs/gel_experiment/version_1/geometry_config.json"
        cfg = json.loads(path.read_text()); cfg["route"] = "gradient"
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "image-route"):
            geometry_settings(1, self.root)

    def test_normalized_force_model_requires_training_statistics(self):
        self.complete_fixture()
        self.force_cfg["normalize_disp"] = True
        self.write("force_logs/HyperNet_Run/version_2/force_config.json", json.dumps(self.force_cfg))
        self.assertTrue(any("normalization statistics" in e for e in resource_errors(self.selection, self.root)))

    def test_resource_paths_cannot_refer_to_original_checkout(self):
        selection = Resources(1, 2, "../another-checkout/prior.npy", "cam/example.npy")
        self.assertTrue(any("inside this project" in e for e in resource_errors(selection, self.root)))

    def test_force_entry_has_no_device_imports_during_preflight(self):
        spec = importlib.util.spec_from_file_location("release_entry", ROOT / "infer_force_realtime_2.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        import builtins
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {"cv2", "torch", "cupy", "serial", "optimized"}:
                raise AssertionError(f"Device/GPU import during resource check: {name}")
            return original_import(name, *args, **kwargs)
        self.complete_fixture()
        from runtime_resources import require_resources
        with patch.object(entry, "require_resources", side_effect=lambda selection: require_resources(selection, self.root)), patch("builtins.__import__", guarded_import):
            self.assertEqual(entry.main(["--check-resources", "--geo-version", "1", "--force-version", "2",
                                         "--prior", self.selection.prior, "--preset", self.selection.preset]), 0)
            (self.root / "depth_prior/example.npy").unlink()
            self.assertEqual(entry.main(["--geo-version", "1", "--force-version", "2",
                                         "--prior", self.selection.prior, "--preset", self.selection.preset]), 2)

    def test_cli_help_is_independent_of_working_directory(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "infer_force_realtime_2.py"), "--help"],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check-resources", result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
