# Code Cleanup Plan (Notebook-first)

## 已完成
- 按项目拆分 notebook 目录结构
- 增加共享模块：`projects/shared/src/motif_common.py`
- 增加重复函数审计脚本：`tools/notebook_duplicate_audit.py`
- 输出重复函数报告：`reports/notebook_duplicate_functions.md`

## 当前重复热点（按审计）
- `map_region_to_group`（7）
- `group3`（7）
- `combination`（5）
- `sample_random_counts`（4）
- `truncate_colormap`（4）
- `analyze_and_plot`（3）
- `c_n_3`（3）
- `real_motif_for_submatrix`（3）

## 下一步建议（可继续自动化）
1. 把上述热点函数完全迁移到 `projects/shared/src/`，并在 notebook 中统一 import
2. 统一 I/O 路径：
   - 输入默认走 `projects/<project>/data/raw/`
   - 输出默认走 `projects/<project>/outputs/{figures,tables}/`
3. 为 notebook 加一个固定“Bootstrap Cell”，统一依赖、路径、随机种子
4. 对超长 notebook 按主题拆分为 2-4 个 notebook，避免单文件过长重复拷贝
