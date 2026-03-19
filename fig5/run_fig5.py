from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


NOTEBOOK_NAME = "fig5.ipynb"
DATA_FILE_NAME = "swER_all.jsonl"


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
    base_dir = Path(__file__).resolve().parent
    notebook_path = base_dir / NOTEBOOK_NAME
    data_path = base_dir / DATA_FILE_NAME
    results_dir = base_dir / "results"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing required input file: {data_path}")

    with tempfile.TemporaryDirectory(prefix="fig5_run_", dir=str(base_dir)) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        shutil.copy2(notebook_path, tmp_dir / NOTEBOOK_NAME)
        shutil.copy2(data_path, tmp_dir / DATA_FILE_NAME)
        execute_notebook(tmp_dir / NOTEBOOK_NAME, tmp_dir)
        exported_files = sync_results(tmp_dir, results_dir)

    if not exported_files:
        raise RuntimeError("The Fig. 5 notebook finished without exporting any SVG files.")

    print(f"Exported {len(exported_files)} SVG files to {results_dir}")


if __name__ == "__main__":
    main()

