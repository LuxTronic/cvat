from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full Hasty -> SAM -> YOLO-seg package -> CVAT import pipeline."
    )
    parser.add_argument("--project-id", required=True, help="Hasty project UUID.")
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Hasty dataset UUID filter. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="Segmentation device. Default 'gpu' maps to cuda:0.",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="CVAT project name to create/use.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching exports from Hasty.",
    )
    parser.add_argument(
        "--skip-segment",
        action="store_true",
        help="Skip SAM segmentation generation.",
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help="Skip import dataset packaging zip.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip CVAT import step.",
    )
    return parser.parse_args()


def run_step(args: list[str]) -> None:
    print("\n$ " + " ".join(args))
    result = subprocess.run(args, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed with exit code {result.returncode}: {' '.join(args)}")


def main() -> None:
    args = parse_args()
    py = sys.executable
    device = "cuda:0" if args.device == "gpu" else args.device

    if not args.skip_fetch:
        cmd = [py, str(SCRIPT_DIR / "fetch_hasty_exports.py"), "--project-id", args.project_id, "--overwrite"]
        for ds in args.dataset_id:
            cmd.extend(["--dataset-id", ds])
        run_step(cmd)

    if not args.skip_segment:
        run_step(
            [
                py,
                str(SCRIPT_DIR / "segment.py"),
                "--device",
                device,
                "--overwrite",
            ]
        )

    if not args.skip_package:
        run_step([py, str(SCRIPT_DIR / "prepare_import_dataset.py"), "--overwrite"])

    if not args.skip_push:
        run_step(
            [
                py,
                str(SCRIPT_DIR / "push_to_cvat.py"),
                "--dataset-zip",
                str(SCRIPT_DIR / "hasty_exports" / "import_dataset.zip"),
                "--project-name",
                args.project_name,
            ]
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
