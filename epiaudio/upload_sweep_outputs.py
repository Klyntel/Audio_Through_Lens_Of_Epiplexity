
"""Upload existing local Pareto-front sweep outputs to Comet.

Example:
    uv run python -m epiaudio.upload_sweep_outputs --sweep birdset_hsn
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from epiaudio.sweep_summary import discover_sweep_outputs, upload_sweep_summary


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "sweep_outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload aggregate pf_grid CSVs and plots to Comet."
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="Sweep-name prefix, e.g. birdset_hsn.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing sweep outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--project", default="epiaudio")
    parser.add_argument("--workspace", default="epi-audio")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching output bundles without uploading them.",
    )
    args = parser.parse_args()

    bundles = discover_sweep_outputs(args.output_dir, args.sweep)
    if not bundles:
        raise SystemExit(
            f"No Pareto outputs matching {args.sweep!r} found in {args.output_dir}"
        )

    if not args.dry_run:
        load_dotenv()
        if not os.environ.get("COMET_ML_API"):
            raise SystemExit("COMET_ML_API is not set")

    failures = 0
    for bundle in bundles:
        if args.dry_run:
            print(f"WOULD UPLOAD: {bundle.sweep_name}")
            continue
        try:
            experiment_key = upload_sweep_summary(
                bundle,
                project_name=args.project,
                workspace=args.workspace,
                tags=[args.sweep],
            )
            suffix = f" ({experiment_key})" if experiment_key else ""
            print(f"UPLOADED: {bundle.sweep_name}{suffix}")
        except Exception as exc:
            failures += 1
            print(f"FAILED: {bundle.sweep_name}: {exc!r}")

    if failures:
        raise SystemExit(f"{failures} sweep summary upload(s) failed")


if __name__ == "__main__":
    main()