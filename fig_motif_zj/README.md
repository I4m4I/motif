# fig_motif

已整理为两个独立项目：

## 1) our 多脑区 motif
路径：`projects/our_multiregion_motif/`

主要 notebook：
- `projects/our_multiregion_motif/notebooks/fig1_new_region.ipynb`
- `projects/our_multiregion_motif/notebooks/figs6.ipynb`
- `projects/our_multiregion_motif/notebooks/heatmap.ipynb`
- `projects/our_multiregion_motif/notebooks/soma_distribution.ipynb`
- `projects/our_multiregion_motif/notebooks/supp_fig.ipynb`

## 2) clone motif
路径：`projects/clone_motif/`

主要 notebook：
- `projects/clone_motif/notebooks/fig1_clone.ipynb`

---

## 说明
- 已完成项目拆分 + 数据/输出归档
- 原始数据在各项目 `data/raw/`
- 统一扁平数据在各项目 `data/processed/flat_json/`
- 绘图只读取扁平 JSON，输出到各项目 `outputs/`

## 代码去重与整理工具
- 共享函数模块：`projects/shared/src/motif_common.py`
- 扁平 JSON 数据处理：`projects/shared/src/flat_json_pipeline.py`
- 扁平 JSON 绘图模块：`projects/shared/src/flat_json_plot.py`
- notebook 重复函数审计：`tools/notebook_duplicate_audit.py`
- 数据与输出归档脚本：`tools/reorganize_assets.sh`
- notebook 批量重构脚本：`tools/refactor_notebooks.py`
- 一键运行扁平 JSON 流水线：`tools/run_flat_pipeline.sh`
- 扁平 JSON 报告生成：`tools/generate_flat_json_reports.py`
- 审计报告输出：`reports/notebook_duplicate_functions.md`
- 归档清单输出：`reports/reorg_manifest.tsv`
- notebook 重构报告：`reports/notebook_refactor_report.md`
- 代码清理计划：`reports/code_cleanup_plan.md`
- 一层 JSON 校验报告：`reports/flat_json_validation.md`

## 当前执行方式
- 优先运行各项目 `notebooks/split/` 里的 `partXX` 文件（可维护性更高）
- 原始 notebook 保留在 `notebooks/`，用于对照与追溯

## 新流程（处理/绘图分离）
1. 数据处理（raw -> 一层 JSON）：
- `python3 projects/our_multiregion_motif/scripts/process_data.py`
- `python3 projects/clone_motif/scripts/process_data.py`
2. 绘图（只从一层 JSON 出图）：
- `python3 projects/our_multiregion_motif/scripts/plot_from_flat_json.py`
- `python3 projects/clone_motif/scripts/plot_from_flat_json.py`
