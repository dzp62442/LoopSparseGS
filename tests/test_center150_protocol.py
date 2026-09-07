import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comp_svfgs.dataset_omniscene import (
    CENTER150_FILENAME,
    CENTER150_SAMPLE_COUNT,
    load_center150_tokens,
)
from scripts import run_omniscene


EVAL_ITERATIONS = (1000, 5000, 10000)


class DummyMetricEvaluator:
    class Device:
        type = "cpu"

    device = Device()

    def __init__(self, value=4.0):
        self.value = value

    def evaluate(self, render_dir, gt_dir, image_names):
        assert tuple(image_names) == run_omniscene.NOVEL_IMAGE_NAMES
        return {
            "l1": self.value + 0.01,
            "psnr": self.value + 20.0,
            "ssim": self.value / 10.0,
            "lpips": self.value / 20.0,
        }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_scene(scene_dir):
    _write_json(
        scene_dir / "cams" / "test.json",
        {"frames": [{"image_name": "{:02d}".format(index)} for index in range(18)]},
    )


def _make_evaluation(model_dir, iteration, value, training_time):
    metrics = {
        "l1": value + 0.01,
        "psnr": value + 20.0,
        "ssim": value / 10.0,
        "lpips": value / 20.0,
    }
    _write_json(
        model_dir / "evaluation" / "iteration_{}.json".format(iteration),
        {
            "format_version": 1,
            "iteration": iteration,
            "split": "test",
            "num_views": 18,
            "metrics": metrics,
            "training_time_seconds": training_time,
        },
    )
    (model_dir / "metrics_{}.txt".format(iteration)).write_text(
        "PSNR : {:.7f}\nSSIM : {:.7f}\nLPIPS : {:.7f}\n".format(
            metrics["psnr"], metrics["ssim"], metrics["lpips"]
        ),
        encoding="utf-8",
    )
    (model_dir / "training_time_{}.txt".format(iteration)).write_text(
        "TRAINING_TIME_SECONDS : {:.7f}\n".format(training_time),
        encoding="utf-8",
    )
    for split in ("renders", "gt"):
        output_dir = model_dir / "test" / "ours_{}".format(iteration) / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(18):
            (output_dir / "{:02d}.png".format(index)).write_bytes(b"png")


def _make_complete_sample(root, name, base_value=1.0):
    scene_dir = root / "prepared" / name
    model_dir = root / "runs" / name
    _make_scene(scene_dir)
    for offset, iteration in enumerate(EVAL_ITERATIONS, 1):
        _make_evaluation(model_dir, iteration, base_value + offset, offset * 10.0)
    final_ply = model_dir / "point_cloud" / "iteration_10000" / "point_cloud.ply"
    final_ply.parent.mkdir(parents=True, exist_ok=True)
    final_ply.write_bytes(b"ply")
    return scene_dir, model_dir


class Center150ManifestTest(unittest.TestCase):
    def _make_version_dir(self, root):
        version_dir = root / "interp_12Hz_trainval"
        info_dir = version_dir / "bin_infos_3.2m"
        info_dir.mkdir(parents=True)
        groups = []
        for scene_index in range(CENTER150_SAMPLE_COUNT):
            scene_token = "{:032x}".format(scene_index + 1)
            groups.append([
                "scene{}_bin{:03d}".format(scene_token, bin_index)
                for bin_index in range(3)
            ])
        all_bins = [token for group in groups for token in group]
        selected = [group[1] for group in groups]
        _write_json(
            version_dir / "bins_val_3.2m.json",
            {"bins": all_bins, "adjacent_bins": groups},
        )
        _write_json(version_dir / CENTER150_FILENAME, {"bins": selected})
        for token in selected:
            (info_dir / (token + ".pkl")).write_bytes(b"pkl")
        return version_dir, selected

    def test_loads_strict_lower_median_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            version_dir, selected = self._make_version_dir(Path(temporary_dir))
            self.assertEqual(load_center150_tokens(str(version_dir)), selected)

    def test_rejects_manifest_not_generated_by_expected_rule(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            version_dir, selected = self._make_version_dir(Path(temporary_dir))
            _write_json(version_dir / CENTER150_FILENAME, {"bins": list(reversed(selected))})
            with self.assertRaisesRegex(ValueError, "lower-median"):
                load_center150_tokens(str(version_dir))


class Center150RunnerTest(unittest.TestCase):
    def test_backfill_adds_novel_metrics_without_changing_training_time(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _, model_dir = _make_complete_sample(root, "scene")
            time_path = model_dir / "training_time_1000.txt"
            original_time_bytes = time_path.read_bytes()
            original_evaluation = run_omniscene.load_evaluation_record(model_dir, 1000)

            result = run_omniscene.backfill_novel_12_metrics(
                model_dir, 1000, DummyMetricEvaluator()
            )

            updated = run_omniscene.load_evaluation_record(model_dir, 1000)
            self.assertEqual(time_path.read_bytes(), original_time_bytes)
            self.assertEqual(updated["training_time_seconds"], 10.0)
            self.assertEqual(updated["metrics"], original_evaluation["metrics"])
            self.assertAlmostEqual(result["psnr"], 24.0)
            self.assertTrue(run_omniscene.has_novel_12_metrics(model_dir, 1000))

    def test_nonempty_unversioned_run_root_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_root = Path(temporary_dir)
            old_result = run_root / "old-result.txt"
            old_result.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "保护既有实验"):
                run_omniscene._freeze_experiment_config(
                    run_root, {"format_version": 1, "stage": "center150"}
                )
            self.assertEqual(old_result.read_text(encoding="utf-8"), "keep")

    def test_train_command_is_one_continuous_run_without_checkpoints(self):
        command = run_omniscene.build_train_command(
            "/env/python",
            Path("/prepared/scene"),
            Path("/runs"),
            "scene-name",
            10000,
            EVAL_ITERATIONS,
            6,
            True,
        )
        self.assertEqual(command.count("train.py"), 1)
        self.assertIn("--full_eval_metrics", command)
        self.assertIn("-sps", command)
        self.assertNotIn("--checkpoint_iterations", command)
        self.assertNotIn("--start_checkpoint", command)
        test_index = command.index("--test_iterations")
        self.assertEqual(command[test_index + 1:test_index + 4], ["1000", "5000", "10000"])

    def test_completion_requires_all_artifacts_and_monotonic_time(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            scene_dir, model_dir = _make_complete_sample(root, "scene")
            self.assertTrue(
                run_omniscene.scene_complete(model_dir, scene_dir, 10000, EVAL_ITERATIONS)
            )

            missing_image = model_dir / "test" / "ours_5000" / "renders" / "07.png"
            missing_image.unlink()
            self.assertFalse(
                run_omniscene.scene_complete(model_dir, scene_dir, 10000, EVAL_ITERATIONS)
            )

            missing_image.write_bytes(b"png")
            _make_evaluation(model_dir, 5000, 3.0, 5.0)
            self.assertFalse(
                run_omniscene.scene_complete(model_dir, scene_dir, 10000, EVAL_ITERATIONS)
            )

    def test_restart_invalidates_only_completion_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _, model_dir = _make_complete_sample(root, "scene")
            unrelated = model_dir / "keep.txt"
            unrelated.write_text("keep", encoding="utf-8")
            run_omniscene.invalidate_incomplete_scene(model_dir, 10000, EVAL_ITERATIONS)
            self.assertTrue(unrelated.exists())
            self.assertFalse(
                (model_dir / "point_cloud" / "iteration_10000" / "point_cloud.ply").exists()
            )
            for iteration in EVAL_ITERATIONS:
                self.assertFalse(
                    (model_dir / "evaluation" / "iteration_{}.json".format(iteration)).exists()
                )

    def test_aggregate_uses_equal_scene_mean(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            samples = []
            for index, value in enumerate((1.0, 3.0), 1):
                name = "scene{}".format(index)
                scene_dir, model_dir = _make_complete_sample(root, name, value)
                samples.append((name, "token{}".format(index), scene_dir, model_dir))
                for iteration in EVAL_ITERATIONS:
                    run_omniscene.backfill_novel_12_metrics(
                        model_dir, iteration, DummyMetricEvaluator(value + 10.0)
                    )

            with mock.patch.object(run_omniscene, "CENTER150_SAMPLE_COUNT", 2):
                summary = run_omniscene.aggregate_center150(
                    root / "runs", samples, (112, 200), 10000, EVAL_ITERATIONS
                )
            self.assertEqual(summary["scene_count"], 2)
            self.assertAlmostEqual(summary["averages"]["1000"]["psnr"], 23.0)
            self.assertAlmostEqual(
                summary["averages"]["10000"]["training_time_seconds"], 30.0
            )
            self.assertAlmostEqual(
                summary["view_subsets"]["novel_12"]["averages"]["1000"]["psnr"],
                32.0,
            )
            self.assertTrue((root / "runs" / "center150_metrics_summary.json").exists())
            self.assertTrue((root / "runs" / "center150_metrics_summary.txt").exists())


if __name__ == "__main__":
    unittest.main()
