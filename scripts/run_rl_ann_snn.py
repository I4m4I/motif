from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the Fig. 4 figures from bundled result arrays.")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Run scripts/prepare_rl_data.py before plotting.",
    )
    return parser.parse_args()


def run_script(script_path: Path, cwd: Path) -> None:
    subprocess.run([sys.executable, str(script_path)], cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent
    figures_dir = repo_root / "results" / "reinforcement_learning" / "figures"

    if args.refresh_data:
        run_script(base_dir / "prepare_rl_data.py", repo_root)

    run_script(repo_root / "evaluation" / "plot_rl_suite.py", repo_root)

    summary_svg = figures_dir / "summary_ann_snn.svg"
    if not summary_svg.exists():
        raise FileNotFoundError(f"Expected output was not generated: {summary_svg}")

    print(f"Fig. 4 figures are available in {figures_dir}")


if __name__ == "__main__":
    main()

