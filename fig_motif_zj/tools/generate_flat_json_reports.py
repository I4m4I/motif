#!/usr/bin/env python3
import json
from pathlib import Path


def write_conversion_summary(root: Path) -> Path:
    projects = [
        root / "projects" / "our_multiregion_motif",
        root / "projects" / "clone_motif",
    ]
    out = root / "reports" / "flat_json_conversion_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write("# Flat JSON Conversion Summary\n\n")
        for p in projects:
            mani = p / "data" / "processed" / "flat_json" / "flat_json_manifest.json"
            rows = json.loads(mani.read_text(encoding="utf-8")) if mani.exists() else []
            mode = {}
            for r in rows:
                m = r.get("conversion_mode", "unknown")
                mode[m] = mode.get(m, 0) + 1

            f.write(f"## {p.name}\n")
            f.write(f"- files converted: {len(rows)}\n")
            for k in sorted(mode):
                f.write(f"- mode `{k}`: {mode[k]}\n")
            f.write("\n")

    return out


def write_flat_validation(root: Path) -> Path:
    files = sorted((root / "projects").glob("*/data/processed/flat_json/*.json"))
    violations = []
    summary = []

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            violations.append((p, f"json_error:{e}"))
            continue

        if not isinstance(data, list):
            violations.append((p, "top_not_list"))
            continue

        for i, row in enumerate(data[:2000]):
            if not isinstance(row, dict):
                violations.append((p, f"row_not_dict:{i}"))
                break
            for k, v in row.items():
                if isinstance(v, dict):
                    violations.append((p, f"nested_dict:{i}:{k}"))
                    break
            else:
                continue
            break

        summary.append((p, len(data)))

    out = root / "reports" / "flat_json_validation.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# Flat JSON Validation\n\n")
        f.write(f"Files checked: {len(files)}\n\n")
        f.write(f"Violations: {len(violations)}\n\n")

        if violations:
            f.write("## Violations\n")
            for p, m in violations:
                f.write(f"- `{p.relative_to(root)}`: {m}\n")
            f.write("\n")

        f.write("## File Sizes (rows)\n")
        for p, n in summary:
            f.write(f"- `{p.relative_to(root)}`: {n} rows\n")

    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    s1 = write_conversion_summary(root)
    s2 = write_flat_validation(root)
    print(s1)
    print(s2)


if __name__ == "__main__":
    main()
