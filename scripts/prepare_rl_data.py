#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_OUTPUT_ROOT = PROJECT_ROOT / "output"
RAW_RESULTS_ROOT = PROJECT_ROOT / "results" / "reinforcement_learning" / "raw"
REFERENCE_FIGURES_ROOT = PROJECT_ROOT / "results" / "reinforcement_learning" / "reference_figures"
MANIFEST_PATH = PROJECT_ROOT / "results" / "reinforcement_learning" / "results_manifest.json"

ENVIRONMENTS = ("ip", "idp", "walker", "ant")
FILE_PATTERN = re.compile(r"(.+?)_(\d+)(_discrete)?\.npy$")
REFERENCE_FIGURES = (
    "ip.svg",
    "ip_discrete.svg",
    "idp_discrete.svg",
    "walker.svg",
    "walker_SNN.svg",
    "ant.svg",
    "ant_discrete.svg",
)


def copy_environment_results(env_name: str) -> dict[str, object]:
    source_dir = SOURCE_OUTPUT_ROOT / env_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source result directory: {source_dir}")

    target_dir = RAW_RESULTS_ROOT / env_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)

    manifest_entry: dict[str, object] = {
        "source_dir": str(source_dir),
        "local_dir": str(target_dir),
        "ann": {},
        "snn": {},
    }

    for file_path in sorted(target_dir.glob("*.npy")):
        match = FILE_PATTERN.fullmatch(file_path.name)
        if not match:
            continue

        prefix = match.group(1)
        seed = int(match.group(2))
        variant = "snn" if match.group(3) else "ann"
        array = np.load(file_path, allow_pickle=False)
        length = int(array.shape[0]) if array.ndim >= 1 else 0

        variant_bucket = manifest_entry[variant]
        prefix_bucket = variant_bucket.setdefault(
            prefix,
            {"file_count": 0, "seeds": [], "lengths": []},
        )
        prefix_bucket["file_count"] += 1
        prefix_bucket["seeds"].append(seed)
        prefix_bucket["lengths"].append(length)

    for variant in ("ann", "snn"):
        for prefix, stats in manifest_entry[variant].items():
            stats["seeds"] = sorted(stats["seeds"])
            stats["lengths"] = sorted(set(stats["lengths"]))

    return manifest_entry


def copy_reference_figures() -> list[str]:
    REFERENCE_FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for figure_name in REFERENCE_FIGURES:
        source_path = REPO_ROOT / figure_name
        if not source_path.is_file():
            continue
        target_path = REFERENCE_FIGURES_ROOT / figure_name
        shutil.copy2(source_path, target_path)
        copied.append(str(target_path))
    return copied


def main() -> None:
    RAW_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "project_root": str(PROJECT_ROOT),
        "repo_root": str(REPO_ROOT),
        "environments": {},
        "reference_figures": copy_reference_figures(),
    }

    for env_name in ENVIRONMENTS:
        manifest["environments"][env_name] = copy_environment_results(env_name)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
