#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

manifest="$ROOT/reports/reorg_manifest.tsv"
if [[ $DRY_RUN -eq 0 ]]; then
  : > "$manifest"
  printf "source\tdestination\n" >> "$manifest"
fi

is_clone_name() {
  local n="$1"
  [[ "$n" =~ (^clone|^Clone|^example_clone|^example_two_clones|clone[-_]|_clone|all_clones|clones_|^MOp_ALL_clones|^MOp_NZ_clone_vs_ION|^ION_real_vs_ER|^Clone_real_vs_ER|^uni_bi) ]]
}

move_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    echo "SKIP (exists): $(basename "$src") -> $dst"
    return
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY  $(basename "$src") -> ${dst#$ROOT/}"
  else
    mv "$src" "$dst"
    printf "%s\t%s\n" "${src#$ROOT/}" "${dst#$ROOT/}" >> "$manifest"
  fi
}

shopt -s nullglob
for src in "$ROOT"/*; do
  [[ -f "$src" ]] || continue
  base="$(basename "$src")"

  case "$base" in
    README.md|.DS_Store)
      continue
      ;;
  esac

  ext="${base##*.}"
  ext_lc="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"

  case "$ext_lc" in
    csv|json|npy|pkl|xlsx)
      if is_clone_name "$base"; then
        dst="$ROOT/projects/clone_motif/data/raw/$base"
      else
        dst="$ROOT/projects/our_multiregion_motif/data/raw/$base"
      fi
      move_file "$src" "$dst"
      ;;
    svg|png)
      if is_clone_name "$base"; then
        dst="$ROOT/projects/clone_motif/outputs/figures/$base"
      else
        dst="$ROOT/projects/our_multiregion_motif/outputs/figures/$base"
      fi
      move_file "$src" "$dst"
      ;;
    zip)
      if is_clone_name "$base"; then
        dst="$ROOT/projects/clone_motif/outputs/archives/$base"
      else
        dst="$ROOT/projects/our_multiregion_motif/outputs/archives/$base"
      fi
      move_file "$src" "$dst"
      ;;
    *)
      dst="$ROOT/projects/our_multiregion_motif/outputs/misc/$base"
      move_file "$src" "$dst"
      ;;
  esac

done

if [[ $DRY_RUN -eq 0 ]]; then
  echo "manifest: $manifest"
fi
