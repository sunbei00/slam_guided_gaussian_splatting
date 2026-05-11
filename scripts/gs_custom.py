#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_RESULT_DIR = "./results/custom2"
DEFAULT_PORT = 8082
DEFAULT_GPU = "0"
METADATA_NAME = "gs_custom_run.json"


def repo_root():
    return Path(__file__).resolve().parent.parent


def gsplat_dir():
    return repo_root() / "gsplat"


def resolve_from_cwd(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def resolve_from_gsplat(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (gsplat_dir() / path).resolve()


def result_dir_arg(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(path)


def parse_step(path):
    match = re.search(r"ckpt_(\d+)_rank\d+\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def find_latest_ckpt(result_dir):
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted(
        ckpt_dir.glob("ckpt_*_rank0.pt"),
        key=lambda path: (parse_step(path), path.stat().st_mtime),
    )
    if not ckpts:
        raise FileNotFoundError(f"No rank0 checkpoint found in: {ckpt_dir}")
    return ckpts[-1]


def metadata_path(result_dir):
    return result_dir / METADATA_NAME


def write_metadata(result_dir, data_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_dir": str(data_dir),
        "result_dir": str(result_dir),
    }
    metadata_path(result_dir).write_text(json.dumps(payload, indent=2) + "\n")


def read_metadata_data_dir(result_dir):
    path = metadata_path(result_dir)
    if path.exists():
        payload = json.loads(path.read_text())
        data_dir = payload.get("data_dir")
        if data_dir:
            return resolve_from_gsplat(data_dir)

    cfg_path = result_dir / "cfg.yml"
    if cfg_path.exists():
        for line in cfg_path.read_text().splitlines():
            if line.startswith("data_dir:"):
                value = line.split(":", 1)[1].strip().strip("'\"")
                if value:
                    return resolve_from_gsplat(value)

    return None


def run(cmd, cuda_visible_devices):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    print("[INFO] Current working directory:", gsplat_dir())
    print("[INFO] CUDA_VISIBLE_DEVICES:", cuda_visible_devices)
    print("[INFO] Running command:")
    print(" ".join(cmd))

    try:
        completed = subprocess.run(cmd, cwd=gsplat_dir(), env=env, check=True)
        return completed.returncode
    except FileNotFoundError as exc:
        print(f"[ERROR] Command not found: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Process failed with return code {exc.returncode}", file=sys.stderr)
        return exc.returncode


def train(args):
    data_dir = resolve_from_cwd(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    result_dir = resolve_from_gsplat(args.result_dir)
    write_metadata(result_dir, data_dir)

    cmd = [
        "pixi",
        "run",
        "-e",
        args.pixi_env,
        "python3",
        "examples/custom_trainer.py",
        "default",
        "--data_dir",
        str(data_dir),
        "--ssim_lambda",
        "0.1",
        "--init_scale",
        "0.7",
        "--opacity_reg",
        "1e-4",
        "--strategy.prune_opa",
        "0.01",
        "--strategy.prune_scale3d",
        "0.07",
        "--strategy.prune_scale2d",
        "0.10",
        "--pose_opt",
        "--pose_opt_lr",
        "1e-4",
        "--pose_opt_reg",
        "1e-4",
        "--viewer_z_axis_move_scale",
        "0.08",
        "--viewer_orbit_pivot_distance",
        "0.4",
        "--port",
        str(args.port),
        "--result_dir",
        result_dir_arg(args.result_dir),
    ]
    return run(cmd, args.gpu)


def view(args):
    result_dir = resolve_from_gsplat(args.result_dir)
    try:
        ckpt = find_latest_ckpt(result_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.data_dir is None:
        try:
            data_dir = read_metadata_data_dir(result_dir)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[ERROR] Could not read saved training metadata: {exc}", file=sys.stderr)
            return 1
        if data_dir is None:
            print(
                "[ERROR] --data-dir was not provided and no saved training data_dir was found.",
                file=sys.stderr,
            )
            return 1
    else:
        data_dir = resolve_from_cwd(args.data_dir)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    cmd = [
        "pixi",
        "run",
        "-e",
        args.pixi_env,
        "python3",
        "examples/simple_viewer.py",
        "--ckpt",
        str(ckpt),
        "--output_dir",
        str(result_dir),
        "--port",
        str(args.port),
        "--data_dir",
        str(data_dir),
        "--data_factor",
        "1",
        "--flip_yz",
    ]
    return run(cmd, args.gpu)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="gs_custom.py",
        description="Small wrapper for gsplat custom training and viewing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="run custom_trainer.py")
    train_parser.add_argument(
        "--data-dir",
        required=True,
        help="COLMAP dataset directory. Required.",
    )
    train_parser.add_argument(
        "--result-dir",
        default=DEFAULT_RESULT_DIR,
        help=f"Training output directory relative to gsplat. Default: {DEFAULT_RESULT_DIR}",
    )
    train_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Viewer port. Default: {DEFAULT_PORT}",
    )
    train_parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help=f"CUDA_VISIBLE_DEVICES value. Default: {DEFAULT_GPU}",
    )
    train_parser.add_argument(
        "--pixi-env",
        default="gs",
        help="Pixi environment name. Default: gs",
    )
    train_parser.set_defaults(func=train)

    view_parser = subparsers.add_parser("view", help="run simple_viewer.py")
    view_parser.add_argument(
        "--data-dir",
        default=None,
        help="COLMAP dataset directory. Defaults to the data_dir saved by train.",
    )
    view_parser.add_argument(
        "--result-dir",
        default=DEFAULT_RESULT_DIR,
        help=f"Training output directory relative to gsplat. Default: {DEFAULT_RESULT_DIR}",
    )
    view_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Viewer port. Default: {DEFAULT_PORT}",
    )
    view_parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help=f"CUDA_VISIBLE_DEVICES value. Default: {DEFAULT_GPU}",
    )
    view_parser.add_argument(
        "--pixi-env",
        default="gs",
        help="Pixi environment name. Default: gs",
    )
    view_parser.set_defaults(func=view)

    return parser.parse_args()


def main():
    target_dir = gsplat_dir()
    if not target_dir.exists():
        print(f"[ERROR] Target directory not found: {target_dir}", file=sys.stderr)
        return 1

    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
