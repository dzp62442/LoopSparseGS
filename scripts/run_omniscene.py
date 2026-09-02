import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comp_svfgs.dataset_omniscene import (  # noqa: E402
    CENTER150_SAMPLE_COUNT,
    get_bin_tokens,
    preprocess_scene,
)


DEFAULT_TOTAL_ITERATIONS = 10_000
DEFAULT_EVAL_ITERATIONS = (1_000, 5_000, 10_000)
EVALUATION_FORMAT_VERSION = 1
EXPERIMENT_FORMAT_VERSION = 1
SUMMARY_FORMAT_VERSION = 1
METRIC_NAMES = ("psnr", "ssim", "lpips")


def _run_cmd(cmd: Sequence[str]) -> None:
    print("[CMD] {}".format(" ".join(cmd)), flush=True)
    subprocess.run(list(cmd), cwd=str(REPO_ROOT), check=True)


def _atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _parse_resolution(value: str) -> Tuple[int, int]:
    try:
        height_str, width_str = value.lower().split("x")
        height, width = int(height_str), int(width_str)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("分辨率必须写成 HxW，例如 112x200") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("分辨率必须为正数")
    return height, width


def _validate_protocol(iterations: int, eval_iterations: Sequence[int]) -> Tuple[int, ...]:
    if iterations <= 0:
        raise ValueError("--iterations 必须为正数")
    normalized = tuple(eval_iterations)
    if not normalized:
        raise ValueError("--eval_iterations 不能为空")
    if any(iteration <= 0 for iteration in normalized):
        raise ValueError("评估迭代数必须为正数")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("--eval_iterations 必须严格递增且不能重复")
    if normalized[-1] != iterations:
        raise ValueError("最后一个评估里程碑必须等于 --iterations")
    return normalized


def _default_output_roots(stage: str, resolution: Tuple[int, int], iterations: int,
                          eval_iterations: Sequence[int]) -> Tuple[Path, Path]:
    resolution_tag = "{}x{}".format(*resolution)
    eval_tag = "-".join(str(iteration) for iteration in eval_iterations)
    preproc_root = REPO_ROOT / "output" / "omniscene_preproc" / resolution_tag
    run_root = REPO_ROOT / "output" / (
        "omniscene_{}_{}_iter{}_eval{}".format(
            stage, resolution_tag, iterations, eval_tag
        )
    )
    return preproc_root, run_root


def _experiment_config(data_root: Path, stage: str, resolution: Tuple[int, int],
                       iterations: int, eval_iterations: Sequence[int], train_sub: int,
                       sparse_sampling: bool) -> Dict:
    return {
        "format_version": EXPERIMENT_FORMAT_VERSION,
        "dataset": "omniscene",
        "data_root": str(data_root.resolve()),
        "stage": stage,
        "resolution": list(resolution),
        "iterations": iterations,
        "eval_iterations": list(eval_iterations),
        "train_sub": train_sub,
        "sparse_sampling": sparse_sampling,
        "sample_resume_policy": "completed-skip_incomplete-restart",
    }


def _freeze_experiment_config(run_root: Path, config: Dict) -> None:
    config_path = run_root / "experiment_config.json"
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("实验配置文件损坏: {}".format(config_path)) from exc
        if existing != config:
            raise RuntimeError(
                "输出目录已有不同实验配置，请更换 --run_root，避免混合结果: {}".format(run_root)
            )
        return
    existing_entries = list(run_root.iterdir())
    if existing_entries:
        raise RuntimeError(
            "输出目录非空但没有 experiment_config.json；为保护既有实验，"
            "请更换 --run_root: {}".format(run_root)
        )
    _atomic_write_json(config_path, config)


def expected_test_image_names(scene_dir: Path) -> Tuple[str, ...]:
    test_json = scene_dir / "cams" / "test.json"
    data = json.loads(test_json.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != 18:
        raise ValueError("测试相机必须恰好包含 18 帧: {}".format(test_json))
    names = tuple(frame["image_name"] + ".png" for frame in frames)
    canonical_names = tuple("{:02d}.png".format(index) for index in range(18))
    if names != canonical_names:
        raise ValueError("测试图像名必须保持 SVF-GS 的 00.png--17.png: {}".format(test_json))
    return names


def _parse_metric_text(path: Path) -> Dict[str, float]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        key = name.strip().lower()
        if separator and key in METRIC_NAMES:
            values[key] = float(value.strip())
    if set(values) != set(METRIC_NAMES):
        raise ValueError("指标文件不完整: {}".format(path))
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("指标文件含非有限值: {}".format(path))
    return values


def _parse_training_time_text(path: Path) -> float:
    name, separator, value = path.read_text(encoding="utf-8").strip().partition(":")
    if not separator or name.strip() != "TRAINING_TIME_SECONDS":
        raise ValueError("训练耗时文件格式错误: {}".format(path))
    training_time = float(value.strip())
    if not math.isfinite(training_time) or training_time < 0.0:
        raise ValueError("训练耗时无效: {}".format(path))
    return training_time


def load_evaluation_record(model_dir: Path, iteration: int) -> Dict:
    evaluation_path = model_dir / "evaluation" / "iteration_{}.json".format(iteration)
    record = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if record.get("format_version") != EVALUATION_FORMAT_VERSION:
        raise ValueError("评估记录版本不匹配: {}".format(evaluation_path))
    if record.get("iteration") != iteration or record.get("split") != "test":
        raise ValueError("评估记录身份不匹配: {}".format(evaluation_path))
    if record.get("num_views") != 18:
        raise ValueError("评估记录必须包含 18 个视角: {}".format(evaluation_path))

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("评估记录缺少 metrics: {}".format(evaluation_path))
    required_values = [metrics.get(name) for name in METRIC_NAMES]
    required_values.append(metrics.get("l1"))
    required_values.append(record.get("training_time_seconds"))
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in required_values):
        raise ValueError("评估记录包含无效数值: {}".format(evaluation_path))
    if record["training_time_seconds"] < 0.0:
        raise ValueError("训练耗时不能为负数: {}".format(evaluation_path))

    text_metrics = _parse_metric_text(model_dir / "metrics_{}.txt".format(iteration))
    text_time = _parse_training_time_text(
        model_dir / "training_time_{}.txt".format(iteration)
    )
    for metric_name in METRIC_NAMES:
        if not math.isclose(
            float(metrics[metric_name]), text_metrics[metric_name], rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("JSON 与文本指标不一致: {}".format(evaluation_path))
    if not math.isclose(
        float(record["training_time_seconds"]), text_time, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError("JSON 与文本训练耗时不一致: {}".format(evaluation_path))
    return record


def iteration_complete(model_dir: Path, scene_dir: Path, iteration: int) -> bool:
    try:
        load_evaluation_record(model_dir, iteration)
        expected_names = set(expected_test_image_names(scene_dir))
        iteration_dir = model_dir / "test" / "ours_{}".format(iteration)
        render_paths = list((iteration_dir / "renders").glob("*.png"))
        gt_paths = list((iteration_dir / "gt").glob("*.png"))
        return (
            {path.name for path in render_paths} == expected_names
            and {path.name for path in gt_paths} == expected_names
            and all(path.stat().st_size > 0 for path in render_paths + gt_paths)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def scene_complete(model_dir: Path, scene_dir: Path, iterations: int,
                   eval_iterations: Sequence[int]) -> bool:
    final_ply = model_dir / "point_cloud" / "iteration_{}".format(iterations) / "point_cloud.ply"
    if not final_ply.is_file() or final_ply.stat().st_size == 0:
        return False
    if not all(
        iteration_complete(model_dir, scene_dir, iteration)
        for iteration in eval_iterations
    ):
        return False
    try:
        training_times = [
            load_evaluation_record(model_dir, iteration)["training_time_seconds"]
            for iteration in eval_iterations
        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return training_times == sorted(training_times)


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def invalidate_incomplete_scene(model_dir: Path, iterations: int,
                                eval_iterations: Sequence[int]) -> None:
    """Remove completion evidence so stale milestones cannot mix across restarted runs."""
    _unlink_if_exists(model_dir / "scene_complete.json")
    _unlink_if_exists(
        model_dir / "point_cloud" / "iteration_{}".format(iterations) / "point_cloud.ply"
    )
    for iteration in eval_iterations:
        _unlink_if_exists(model_dir / "evaluation" / "iteration_{}.json".format(iteration))
        _unlink_if_exists(model_dir / "metrics_{}.txt".format(iteration))
        _unlink_if_exists(model_dir / "training_time_{}.txt".format(iteration))


def build_train_command(python_bin: str, scene_dir: Path, run_root: Path, exp_name: str,
                        iterations: int, eval_iterations: Sequence[int], train_sub: int,
                        sparse_sampling: bool) -> List[str]:
    command = [
        python_bin,
        "train.py",
        "-s",
        str(scene_dir),
        "--exp_name",
        exp_name,
        "--model_path",
        str(run_root),
        "--dataset_type",
        "omniscene",
        "--train_sub",
        str(train_sub),
        "--iterations",
        str(iterations),
        "--test_iterations",
    ]
    command.extend(str(iteration) for iteration in eval_iterations)
    command.extend([
        "--save_iterations",
        str(iterations),
        "--eval",
        "-r",
        "1",
        "--pseudo_loop_iters",
        "0",
        "--full_eval_metrics",
    ])
    if sparse_sampling:
        command.append("-sps")
    return command


def _scene_completion_payload(scene_name: str, bin_token: str, resolution: Tuple[int, int],
                              iterations: int, eval_iterations: Sequence[int],
                              model_dir: Path) -> Dict:
    return {
        "format_version": SUMMARY_FORMAT_VERSION,
        "split": "center150",
        "scene_name": scene_name,
        "bin_token": bin_token,
        "resolution": list(resolution),
        "iterations": iterations,
        "eval_iterations": list(eval_iterations),
        "milestones": {
            str(iteration): load_evaluation_record(model_dir, iteration)
            for iteration in eval_iterations
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def aggregate_center150(run_root: Path, samples: Sequence[Tuple[str, str, Path, Path]],
                        resolution: Tuple[int, int], iterations: int,
                        eval_iterations: Sequence[int]) -> Dict:
    if len(samples) != CENTER150_SAMPLE_COUNT:
        raise ValueError("Center150 汇总要求恰好 150 个样本，实际为 {}".format(len(samples)))

    sample_records = []
    accumulators = {
        iteration: {name: [] for name in METRIC_NAMES + ("training_time_seconds",)}
        for iteration in eval_iterations
    }
    for scene_name, bin_token, scene_dir, model_dir in samples:
        if not scene_complete(model_dir, scene_dir, iterations, eval_iterations):
            raise RuntimeError("样本尚未完整结束，不能汇总: {}".format(scene_name))
        milestones = {}
        for iteration in eval_iterations:
            evaluation = load_evaluation_record(model_dir, iteration)
            record = {
                name: float(evaluation["metrics"][name]) for name in METRIC_NAMES
            }
            record["training_time_seconds"] = float(evaluation["training_time_seconds"])
            milestones[str(iteration)] = record
            for name, value in record.items():
                accumulators[iteration][name].append(value)
        sample_records.append({
            "scene_name": scene_name,
            "bin_token": bin_token,
            "milestones": milestones,
        })

    averages = {
        str(iteration): {
            name: sum(values) / len(values)
            for name, values in accumulators[iteration].items()
        }
        for iteration in eval_iterations
    }
    summary = {
        "format_version": SUMMARY_FORMAT_VERSION,
        "split": "center150",
        "scene_count": len(samples),
        "aggregation": "equal_scene_mean",
        "resolution": list(resolution),
        "iterations": iterations,
        "eval_iterations": list(eval_iterations),
        "averages": averages,
        "samples": sample_records,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(run_root / "center150_metrics_summary.json", summary)

    lines = [
        "LoopSparseGS OmniScene Center150 summary",
        "scene_count: {}".format(len(samples)),
        "aggregation: equal_scene_mean",
        "",
        "iteration,PSNR,SSIM,LPIPS,TRAINING_TIME_SECONDS",
    ]
    for iteration in eval_iterations:
        values = averages[str(iteration)]
        lines.append(
            "{},{:.7f},{:.7f},{:.7f},{:.7f}".format(
                iteration,
                values["psnr"],
                values["ssim"],
                values["lpips"],
                values["training_time_seconds"],
            )
        )
    _atomic_write_text(run_root / "center150_metrics_summary.txt", "\n".join(lines) + "\n")
    return summary


def _print_summary(summary: Dict) -> None:
    print("\nCenter150 全部完成（150 个场景等权平均）")
    print("iteration      PSNR      SSIM     LPIPS   train_time_s")
    for iteration in summary["eval_iterations"]:
        values = summary["averages"][str(iteration)]
        print(
            "{:>9d}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>13.3f}".format(
                iteration,
                values["psnr"],
                values["ssim"],
                values["lpips"],
                values["training_time_seconds"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoopSparseGS OmniScene Center150：连续训练、阶段评估、断点跳过与汇总"
    )
    parser.add_argument("--data_root", default="datasets/omniscene",
                        help="OmniScene 数据集根目录")
    parser.add_argument("--stage", default="center150",
                        choices=["center150", "train", "val", "test", "demo"],
                        help="默认加载 SVF-GS 已生成的 Center150 清单")
    parser.add_argument("--reso", type=_parse_resolution, default=(112, 200),
                        help="预处理分辨率，默认 112x200")
    parser.add_argument("--iterations", type=int, default=DEFAULT_TOTAL_ITERATIONS,
                        help="连续训练总迭代数，默认 10000")
    parser.add_argument("--eval_iterations", nargs="+", type=int,
                        default=list(DEFAULT_EVAL_ITERATIONS),
                        help="评估里程碑，默认 1000 5000 10000")
    parser.add_argument("--train_sub", type=int, default=6, help="训练视角数，默认 6")
    parser.add_argument("--preproc_root", default=None,
                        help="覆盖预处理输出目录")
    parser.add_argument("--run_root", default=None,
                        help="覆盖实验输出目录")
    parser.add_argument("--disable_sps", action="store_true",
                        help="关闭默认启用的 sparse-friendly sampling")
    args = parser.parse_args()

    eval_iterations = _validate_protocol(args.iterations, args.eval_iterations)
    if args.train_sub <= 0:
        raise ValueError("--train_sub 必须为正数")
    data_root = Path(args.data_root).expanduser().resolve()
    default_preproc_root, default_run_root = _default_output_roots(
        args.stage, args.reso, args.iterations, eval_iterations
    )
    preproc_root = (
        Path(args.preproc_root).expanduser().resolve()
        if args.preproc_root else default_preproc_root
    )
    run_root = (
        Path(args.run_root).expanduser().resolve()
        if args.run_root else default_run_root
    )
    bin_tokens = get_bin_tokens(str(data_root), args.stage)
    if args.stage == "center150" and len(bin_tokens) != CENTER150_SAMPLE_COUNT:
        raise RuntimeError("Center150 清单必须恰好包含 150 个样本")

    preproc_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    sparse_sampling = not args.disable_sps
    _freeze_experiment_config(
        run_root,
        _experiment_config(
            data_root, args.stage, args.reso, args.iterations,
            eval_iterations, args.train_sub, sparse_sampling,
        ),
    )

    prefix_width = max(2, len(str(len(bin_tokens))))
    samples = []

    for index, token in enumerate(bin_tokens, 1):
        scene_prefix = str(index).zfill(prefix_width)
        scene_dir = Path(preprocess_scene(
            data_root=str(data_root),
            output_root=str(preproc_root),
            bin_token=token,
            scene_prefix=scene_prefix,
            reso=args.reso,
        ))
        exp_name = "omniscene_{}_{}".format(scene_prefix, token)
        model_dir = run_root / exp_name
        samples.append((exp_name, token, scene_dir, model_dir))

        if scene_complete(model_dir, scene_dir, args.iterations, eval_iterations):
            if args.stage == "center150":
                _atomic_write_json(
                    model_dir / "scene_complete.json",
                    _scene_completion_payload(
                        exp_name, token, args.reso, args.iterations,
                        eval_iterations, model_dir,
                    ),
                )
            print("[SKIP {}/{}] 已完整完成: {}".format(index, len(bin_tokens), exp_name))
            continue

        if args.stage == "center150":
            _unlink_if_exists(run_root / "center150_metrics_summary.json")
            _unlink_if_exists(run_root / "center150_metrics_summary.txt")
        if model_dir.exists():
            print("[RESTART {}/{}] 样本不完整，从头训练: {}".format(
                index, len(bin_tokens), exp_name
            ))
            invalidate_incomplete_scene(model_dir, args.iterations, eval_iterations)
        else:
            print("[START {}/{}] {}".format(index, len(bin_tokens), exp_name))

        _run_cmd(build_train_command(
            sys.executable,
            scene_dir,
            run_root,
            exp_name,
            args.iterations,
            eval_iterations,
            args.train_sub,
            sparse_sampling,
        ))
        if not scene_complete(model_dir, scene_dir, args.iterations, eval_iterations):
            raise RuntimeError("训练命令结束，但样本产物不完整: {}".format(model_dir))
        if args.stage == "center150":
            _atomic_write_json(
                model_dir / "scene_complete.json",
                _scene_completion_payload(
                    exp_name, token, args.reso, args.iterations,
                    eval_iterations, model_dir,
                ),
            )

    if args.stage == "center150":
        summary = aggregate_center150(
            run_root, samples, args.reso, args.iterations, eval_iterations
        )
        _print_summary(summary)
        print("汇总文件: {}".format(run_root / "center150_metrics_summary.json"))


if __name__ == "__main__":
    main()
