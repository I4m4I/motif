# clone_motif

clone motif 分析项目（clone network / clone-to-clone motif）。

## 目录
- `notebooks/fig1_clone.ipynb`
- `notebooks/split/*.ipynb`（自动拆分的分步骤版本）
- `scripts/process_data.py`（只做数据处理，输出一层 JSON）
- `scripts/plot_from_flat_json.py`（只从一层 JSON 绘图）
- `src/`（项目内代码）
- `../shared/src/motif_common.py`（跨 notebook 共用函数）
- `../shared/src/flat_json_pipeline.py`（扁平 JSON 处理）
- `../shared/src/flat_json_plot.py`（扁平 JSON 绘图）

## 建议环境
- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- networkx
- jupyterlab

## 运行建议
1. 从仓库根目录启动 Jupyter：`jupyter lab`
2. 打开 `projects/clone_motif/notebooks/fig1_clone.ipynb`
3. 每个 notebook 已自动注入 bootstrap（`AUTO_BOOTSTRAP_V2`）：
`from motif_common import ...` + 读写路径自动重定向到项目目录
4. 超长分析建议优先使用 `notebooks/split/` 下 `partXX` 顺序执行

## 处理与绘图分离（推荐）
1. 只做数据处理（raw -> flat json）：
`python3 projects/clone_motif/scripts/process_data.py`
2. 只做绘图（flat json -> figures/tables）：
`python3 projects/clone_motif/scripts/plot_from_flat_json.py`

## 数据与输出
- 输入数据：`data/raw/`
- 扁平 JSON：`data/processed/flat_json/`（全部为一层字段，`__` 作为展开分隔符）
- 输出图：`outputs/figures/`
- 输出表：`outputs/tables/`
- 历史压缩包：`outputs/archives/`
