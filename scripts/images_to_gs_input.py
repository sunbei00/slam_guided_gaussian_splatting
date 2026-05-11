#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        prog="images_to_gs_input.py",
        description=(
            "Run the ORB-SLAM2 monocular executable for an image sequence and "
            "store results in a Gaussian-Splatting-style output directory.\n\n"
            "This script first changes the working directory to:\n"
            "  ../ORB_SLAM2\n"
            "relative to this Python file.\n\n"
            "Because of that, relative default paths such as:\n"
            "  ./Examples/Monocular/mono\n"
            "  ./Vocabulary/ORBvoc.txt\n"
            "are resolved inside ORB_SLAM2."
        ),
        epilog=(
            "Examples:\n"
            "  python3 images_to_gs_input.py -i /root/share/kimm_bag/test_dataset/image_0\n\n"
            "  python3 images_to_gs_input.py -i /root/share/kimm_bag/test_dataset/image_0 -o /root/share/kimm_bag/colmap\n\n"
            "  python3 images_to_gs_input.py -m ./Examples/Monocular/mono -v ./Vocabulary/ORBvoc.txt -c ./Examples/Monocular/example.yaml -i /root/share/kimm_bag/test_dataset/image_0 -o ./Results_GS_INPUT"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-m", "--mono-exe",
        type=str,
        default="./Examples/Monocular/mono",
        help=(
            "Path to the ORB-SLAM2 monocular executable.\n"
            "Default: ./Examples/Monocular/mono"
        ),
    )
    parser.add_argument(
        "-v", "--vocab-path",
        type=str,
        default="./Vocabulary/ORBvoc.txt",
        help=(
            "Path to the ORB vocabulary file.\n"
            "Default: ./Vocabulary/ORBvoc.txt"
        ),
    )
    parser.add_argument(
        "-c", "--config-path",
        type=str,
        default="./Examples/Monocular/example.yaml",
        help=(
            "Path to the camera / dataset YAML config file.\n"
            "Default: ./Examples/Monocular/example.yaml"
        ),
    )
    parser.add_argument(
        "-i", "--image-dir",
        type=str,
        required=True,
        help=(
            "Path to the input image directory.\n"
            "Required."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./Results_GS_INPUT",
        help=(
            "Path to the output directory.\n"
            "Default: ./Results_GS_INPUT"
        ),
    )
    parser.add_argument(
        "-e", "--pixi-env",
        type=str,
        default="orbslam2",
        help=(
            "Pixi environment name.\n"
            "Default: orbslam2"
        ),
    )

    return parser.parse_args()


def validate_paths(args):
    mono_exe = Path(args.mono_exe).resolve()
    vocab_path = Path(args.vocab_path).resolve()
    config_path = Path(args.config_path).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("[DEBUG] mono_exe   :", mono_exe)
    print("[DEBUG] vocab_path :", vocab_path)
    print("[DEBUG] config_path:", config_path)
    print("[DEBUG] image_dir  :", image_dir)
    print("[DEBUG] output_dir :", output_dir)

    if not mono_exe.exists():
        raise FileNotFoundError(f"Executable not found: {mono_exe}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    return mono_exe, vocab_path, config_path, image_dir, output_dir


def main():
    target_dir = (Path(__file__).resolve().parent / "../ORB_SLAM2").resolve()

    if not target_dir.exists():
        print(f"[ERROR] Target directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    os.chdir(target_dir)

    args = parse_args()

    try:
        mono_exe, vocab_path, config_path, image_dir, output_dir = validate_paths(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "pixi",
        "run",
        "-e",
        args.pixi_env,
        str(mono_exe),
        str(vocab_path),
        str(config_path),
        str(image_dir),
        str(output_dir),
    ]

    print("[INFO] Current working directory:", Path.cwd())
    print("[INFO] Running command:")
    print(" ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except FileNotFoundError as e:
        print(f"[ERROR] Command not found: {e}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Process failed with return code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
