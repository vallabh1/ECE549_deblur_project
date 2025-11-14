"""
Model interfaces and CLI to run inference with DeblurGANv2, DeblurDiff, or both.

Usage examples:
  - Run DeblurGANv2 on test images:
      python -m src.interfaces --model gan

  - Run DeblurDiff on test images (requires conda env):
      python -m src.interfaces --model diff --conda_env deblurdiff38 --device cuda

  - Run both and save to separate result folders:
      python -m src.interfaces --model both
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Literal


class ProjectPaths:
    def __init__(self) -> None:
        # src/ → project root is parent
        self.project_root: Path = Path(__file__).resolve().parents[1]
        self.models_dir: Path = self.project_root / "models"
        self.results_dir: Path = self.project_root / "results"
        self.test_dir: Path = self.project_root / "test"

        self.deblurganv2_dir: Path = self.models_dir / "DeblurGANv2"
        self.deblurdiff_dir: Path = self.models_dir / "DeblurDiff"


def assert_dir_exists(path: Path, context: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{context} not found: {path}")


def copy_images(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_path in src_dir.glob("*"):
        if src_path.is_file() and src_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            shutil.copy2(src_path, dst_dir / src_path.name)


def run_deblurganv2_inference(paths: ProjectPaths, input_dir: Path | None = None, output_dir: Path | None = None) -> Path:
    """
    Runs DeblurGANv2 predict script by copying test images into the repo's expected
    input folder and then moving the outputs back to the project results directory.
    """
    assert_dir_exists(paths.deblurganv2_dir, "DeblurGANv2 submodule")

    input_dir = input_dir or paths.test_dir
    output_dir = output_dir or (paths.results_dir / "DeblurGANv2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # DeblurGANv2 (common pattern in examples): expects dataset1/blur → writes to submit/
    gan_input_dir = paths.deblurganv2_dir / "dataset1" / "blur"
    gan_submit_dir = paths.deblurganv2_dir / "submit"

    # Prepare input/output dirs
    if gan_input_dir.exists():
        shutil.rmtree(gan_input_dir)
    gan_input_dir.mkdir(parents=True, exist_ok=True)
    if gan_submit_dir.exists():
        shutil.rmtree(gan_submit_dir)

    # Copy images into expected input
    assert_dir_exists(input_dir, "Test images directory")
    copy_images(input_dir, gan_input_dir)

    # Run the repo's predict.py
    cmd = ["python", "predict.py"]
    subprocess.run(cmd, cwd=str(paths.deblurganv2_dir), check=True)

    # Copy results back into project results directory
    assert_dir_exists(gan_submit_dir, "DeblurGANv2 submit output directory")
    for out_path in gan_submit_dir.glob("*"):
        if out_path.is_file():
            shutil.copy2(out_path, output_dir / out_path.name)

    return output_dir


def run_deblurdiff_inference(
    paths: ProjectPaths,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    conda_env: str = "deblurdiff38",
    device: Literal["cuda", "cpu"] = "cuda",
) -> Path:
    """
    Runs DeblurDiff's inference via 'conda run -n <env> python inference.py ...'
    Requires that model weights exist at models/DeblurDiff/model.pth.
    """
    assert_dir_exists(paths.deblurdiff_dir, "DeblurDiff submodule")

    input_dir = input_dir or paths.test_dir
    output_dir = output_dir or (paths.results_dir / "DeblurDiff")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = paths.deblurdiff_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"DeblurDiff weights not found. Expected at: {model_path}\n"
            f"Place checkpoint file as 'model.pth' in the DeblurDiff submodule root."
        )

    assert_dir_exists(input_dir, "Test images directory")

    # Use conda to execute within the target environment
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "inference.py",
        "--model",
        str(model_path),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--device",
        device,
    ]
    subprocess.run(cmd, cwd=str(paths.deblurdiff_dir), check=True)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with DeblurGANv2 and/or DeblurDiff on test images.")
    parser.add_argument(
        "--model",
        choices=["gan", "diff", "both"],
        required=True,
        help="Which model(s) to run.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Path to input test images. Defaults to <project>/test.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Base results directory. Defaults to <project>/results.",
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default="deblurdiff38",
        help="Conda env for DeblurDiff (used with conda run).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for DeblurDiff inference.",
    )

    args = parser.parse_args()

    paths = ProjectPaths()
    test_dir = Path(args.test_dir) if args.test_dir else paths.test_dir
    results_dir = Path(args.results_dir) if args.results_dir else paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ("gan", "both"):
        print("Running DeblurGANv2 inference...")
        gan_out = run_deblurganv2_inference(paths, input_dir=test_dir, output_dir=results_dir / "DeblurGANv2")
        print(f"DeblurGANv2 results saved to: {gan_out}")

    if args.model in ("diff", "both"):
        print("Running DeblurDiff inference...")
        diff_out = run_deblurdiff_inference(
            paths,
            input_dir=test_dir,
            output_dir=results_dir / "DeblurDiff",
            conda_env=args.conda_env,
            device=args.device,
        )
        print(f"DeblurDiff results saved to: {diff_out}")


if __name__ == "__main__":
    main()


