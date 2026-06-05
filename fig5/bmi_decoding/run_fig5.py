#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PlotJob:
    name: str
    script: Path
    result_root: Path
    out_file: Path


PLOT_JOBS = (
    PlotJob(
        name="Jango",
        script=ROOT / "scripts" / "plot_jango_seed_multiseed.py",
        result_root=ROOT / "artifacts" / "results" / "jango",
        out_file=ROOT / "figures" / "jango.png",
    ),
    PlotJob(
        name="Calcium Action",
        script=ROOT / "scripts" / "plot_calcium_seed_multiseed.py",
        result_root=ROOT / "artifacts" / "results" / "calcium",
        out_file=ROOT / "figures" / "calcium_action.png",
    ),
    PlotJob(
        name="mice lick",
        script=ROOT / "scripts" / "plot_mice_lick_seed_multiseed.py",
        result_root=ROOT / "artifacts" / "results" / "mice_lick",
        out_file=ROOT / "figures" / "mice_lick.png",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click runner for Fig5 plotting scripts.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of skipping when a result folder is missing",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default="all",
        help="comma-separated subset: jango,calcium,mice_lick, or all",
    )
    return parser.parse_args()


def has_seed_dirs(path: Path) -> bool:
    return path.is_dir() and any(child.is_dir() and child.name.startswith("seed") for child in path.iterdir())


def selected_jobs(spec: str) -> list[PlotJob]:
    aliases = {
        "jango": "Jango",
        "calcium": "Calcium Action",
        "mice": "mice lick",
        "mice_lick": "mice lick",
    }
    if spec.strip().lower() == "all":
        return list(PLOT_JOBS)
    wanted = {aliases.get(token.strip().lower(), token.strip()) for token in spec.split(",") if token.strip()}
    return [job for job in PLOT_JOBS if job.name in wanted]


def run_job(job: PlotJob, strict: bool) -> bool:
    if not has_seed_dirs(job.result_root):
        message = f"[SKIP] {job.name}: no seed result folders under {job.result_root}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return False
    job.out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(job.script),
        "--root",
        str(job.result_root),
        "--out",
        str(job.out_file),
    ]
    print(f"[RUN] {job.name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return True


def main() -> None:
    args = parse_args()
    jobs = selected_jobs(args.jobs)
    if not jobs:
        raise SystemExit(f"No matching jobs for --jobs={args.jobs!r}")
    completed = 0
    for job in jobs:
        completed += int(run_job(job, strict=bool(args.strict)))
    print(f"[DONE] completed {completed}/{len(jobs)} plotting jobs")


if __name__ == "__main__":
    main()
