from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


NOTEBOOK_NAME = "fig2.ipynb"
DATA_FILE_NAME = "wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Fig. 2 notebook and export SVG figures.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional path to the large pickle input file.",
    )
    return parser.parse_args()


def resolve_data_path(base_dir: Path, explicit_path: Path | None) -> Path:
    local_data = base_dir / DATA_FILE_NAME
    if local_data.exists():
        return local_data

    env_path = os.environ.get("FIG2_DATA_PATH")
    candidates = [explicit_path, Path(env_path) if env_path else None]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Missing {DATA_FILE_NAME}. Place it in {base_dir} or pass --data-path / set FIG2_DATA_PATH."
    )


def hardlink_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        if source.stat().st_size > 100_000_000:
            raise RuntimeError(
                "Could not create a hard link for the large Fig. 2 data file. "
                "Place the pickle file directly next to fig2.ipynb and rerun."
            ) from None
        shutil.copy2(source, destination)


def execute_notebook(notebook_path: Path, workdir: Path) -> None:
    with notebook_path.open("r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    preprocessor = ExecutePreprocessor(timeout=-1)
    preprocessor.preprocess(notebook, {"metadata": {"path": str(workdir)}})


def sync_results(workdir: Path, results_dir: Path) -> list[str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    exported_files: list[str] = []
    for file_path in sorted(workdir.glob("*.svg")):
        destination = results_dir / file_path.name
        shutil.copy2(file_path, destination)
        exported_files.append(file_path.name)
    return exported_files


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    notebook_path = base_dir / NOTEBOOK_NAME
    data_path = resolve_data_path(base_dir, args.data_path)
    results_dir = base_dir / "results"

    with tempfile.TemporaryDirectory(prefix="fig2_run_", dir=str(data_path.parent)) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        tmp_notebook = tmp_dir / NOTEBOOK_NAME
        tmp_data = tmp_dir / DATA_FILE_NAME

        shutil.copy2(notebook_path, tmp_notebook)
        hardlink_or_copy(data_path, tmp_data)
        execute_notebook(tmp_notebook, tmp_dir)
        exported_files = sync_results(tmp_dir, results_dir)

    if not exported_files:
        raise RuntimeError("The Fig. 2 notebook finished without exporting any SVG files.")

    print(f"Exported {len(exported_files)} SVG files to {results_dir}")


if __name__ == "__main__":
    main()

