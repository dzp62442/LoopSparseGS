import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from comp_svfgs.dataset_omniscene import get_bin_tokens, preprocess_scene


def _run_cmd(cmd, cwd):
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="OmniScene 实验：预处理 + 训练 + 渲染 + 评估")
    parser.add_argument("--data_root", type=str, default="datasets/omniscene",
                        help="OmniScene 数据集根目录")
    parser.add_argument("--stage", type=str, default="val", choices=["train", "val", "test", "demo"],
                        help="数据分割模式（默认 val）")
    parser.add_argument("--reso", type=str, default="112x200", choices=["112x200", "224x400"],
                        help="图像分辨率")
    parser.add_argument("--preproc_root", type=str, default="./output/omniscene_preproc",
                        help="预处理输出目录")
    parser.add_argument("--run_root", type=str, default="./output/omniscene_runs",
                        help="训练输出目录")
    parser.add_argument("--rounds", type=int, default=4, help="训练轮数")
    parser.add_argument("--iters_per_round", type=int, default=2500, help="每轮迭代次数（总计 10k）")
    parser.add_argument("--train_sub", type=int, default=6, help="训练视角数")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.isabs(args.preproc_root):
        args.preproc_root = os.path.join(repo_root, args.preproc_root)
    if not os.path.isabs(args.run_root):
        args.run_root = os.path.join(repo_root, args.run_root)

    if args.reso == "112x200":
        reso = (112, 200)
    else:
        reso = (224, 400)

    os.makedirs(args.preproc_root, exist_ok=True)
    os.makedirs(args.run_root, exist_ok=True)

    bin_tokens = get_bin_tokens(args.data_root, args.stage)

    for idx, token in enumerate(bin_tokens, 1):
        scene_prefix = f"{idx:02d}"
        scene_dir = preprocess_scene(
            data_root=args.data_root,
            output_root=args.preproc_root,
            bin_token=token,
            scene_prefix=scene_prefix,
            reso=reso,
        )

        base_exp = f"omniscene_{scene_prefix}_{token}"
        for round_idx in range(args.rounds):
            exp_name = base_exp if round_idx == 0 else f"{base_exp}_{round_idx}"
            cmd = [
                "python", "train.py",
                "-s", scene_dir,
                "--exp_name", exp_name,
                "--model_path", args.run_root,
                "--dataset_type", "omniscene",
                "--train_sub", str(args.train_sub),
                "--iterations", str(args.iters_per_round),
                "--eval",
                "-r", "1",
                "--pseudo_loop_iters", str(round_idx),
            ]
            if round_idx > 0:
                cmd.append("-sps")
            _run_cmd(cmd, cwd=repo_root)

            metrics_cmd = [
                "python", "metrics.py",
                "-m", os.path.join(args.run_root, exp_name),
            ]
            _run_cmd(metrics_cmd, cwd=repo_root)


if __name__ == "__main__":
    main()
