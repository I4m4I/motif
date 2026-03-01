# motif

`motif` 是一个极简的 PyTorch 研究代码仓库，用于把三节点有向 motif 频率约束加到循环策略网络里，并在连续控制任务上结合 PPO 进行训练。

从代码结构看，这个仓库的核心思路是：

1. 用 `RNNCell` 构造 actor / critic。
2. 对 actor 的隐藏到隐藏权重 `weight_hh` 计算 13 类三节点有向 motif 的频率。
3. 在强化学习训练前，先做一段 motif regularization 预训练。
4. 再用 PPO 在 Gym 环境中优化策略。

仓库目前非常小，只包含 3 个文件：

```text
.
├── main.py       # 训练入口
├── tools.py      # motif regularizer、RNN actor/critic、PPO
└── run_batch.sh  # 多 GPU 批量运行脚本
```

## 功能概览

- 基于 `torch.nn.RNNCell` 的循环 actor / critic
- motif 频率正则项 `motifRegular`
- 可选的 motif 预训练阶段
- PPO 训练
- 连续隐藏状态与离散化隐藏状态两种评估输出
- 简单的多 GPU 批量调度脚本

## 环境依赖

仓库没有提供 `requirements.txt` 或 `environment.yml`，按代码推断至少需要：

- Python 3
- PyTorch
- NumPy
- Gym
- tqdm
- MuJoCo 相关依赖

如果你要运行 `Ant-v2`、`Walker2d-v2`、`Humanoid-v2` 这类环境，还需要旧版 Gym 对应的 MuJoCo 环境支持。

## 兼容性

这份代码使用的是旧版 Gym API：

- `env.seed(seed)`
- `obs = env.reset()`
- `obs, reward, done, info = env.step(action)`

如果你使用的是 Gymnasium 或新版 Gym，需要自行适配接口。

## 缺少的配置文件

`main.py` 会读取本地 `params.ini`：

```python
cfg_list.read('params.ini', encoding='utf-8')
cfg = cfg_list[args.env]
```

但这个文件当前并不在仓库里，因此仓库克隆后不能直接运行。根据代码，`params.ini` 至少需要为每个环境提供：

- `env_name`
- `epoch2`
- `output_name`

一个最小示例如下：

```ini
[ip]
env_name = InvertedPendulum-v2
epoch2 = 300
output_name = ip

[ant]
env_name = Ant-v2
epoch2 = 3000
output_name = ant
```

## 单次训练

### 1. Vanilla PPO

当 `--fre` 的 13 个值全为负数时，代码会自动跳过 motif 预训练：

```bash
python main.py \
  --env ip \
  --seed 1 \
  --prefix Vanilla \
  --cuda 0 \
  --epoch1 0 \
  --fre -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

### 2. 带 motif 预训练的运行

要启用 motif regularization，需要：

- 传入至少一个非负的 motif 目标频率
- 把 `--epoch1` 设为大于 0 的值

例如：

```bash
python main.py \
  --env ip \
  --seed 1 \
  --prefix MOP_E \
  --cuda 0 \
  --epoch1 100 \
  --fre -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.449035 -1
```

## `--fre` 参数说明

`--fre` 是一个长度为 13 的向量，对应 13 类 motif 的目标频率。

- 值 `>= 0`：把该 motif 纳入约束
- 值 `< 0`：忽略该 motif

代码逻辑中，如果 13 个值全部小于 0，会直接把 `epoch1` 置为 `0`，即不做 motif 预训练。

## 代码中的训练流程

`main.py` 的训练分两段：

### motif 预训练

对 actor 的循环矩阵 `ppo.actor_net.w` 施加正则：

```python
a = loss2.cal(ppo.actor_net.w)
lossa = 1e5 * a
```

这里 `loss2.cal(...)` 会计算当前循环权重诱导出的 motif 频率，并与 `--fre` 指定的目标频率做平方误差。

### PPO 训练

之后进入标准的 rollout + PPO 更新流程：

- 收集 2048 步经验
- 用 `critic_net` 估计 value
- 用 GAE 计算 advantage
- 按 PPO clipped objective 更新 actor

## 离散隐藏状态评估

除了标准前向传播，代码还实现了 `forward_discrete()`：

- 每一步 RNN hidden state 之后都做一次均匀量化
- 默认量化级数为 `128`

训练完成后会同时保存：

- 标准策略的回报曲线
- 离散化隐藏状态策略的回报曲线

## 输出结果

结果保存在：

```text
output/<output_name>/
```

文件名格式：

```text
<prefix>_<seed>.npy
<prefix>_<seed>_discrete.npy
```

例如：

- `output/ip/Vanilla_1.npy`
- `output/ip/MOP_E_1_discrete.npy`

## `run_batch.sh`

仓库自带了一个批量脚本：

```bash
bash run_batch.sh
```

它的作用是：

- 按 `datasets`
- `run_ids`
- `prefixes`
- `fre_lists`

生成任务列表，然后把任务按 stride 均匀分发到多张 GPU。

脚本中目前默认启用的是：

- 数据集：`ant`
- 前缀：`Vanilla`
- 10 个随机种子

如果你要跑 motif 版本，需要手动取消注释或修改 `prefixes` / `fre_lists`。

## 代码里已经出现的 motif 目标示例

`main.py` 和 `run_batch.sh` 的注释里已经给出了一些可直接复用的目标频率配置，例如：

### MOP-E

```text
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.449035 -1
```

### MOP

```text
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.249035 -1
```

### ORBI

```text
0.091003 0.287659 0.178217 0.107868 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

### ORBI-E

```text
0.19003 0.287659 0.278217 0.107868 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

## 主要参数

`main.py` 中最重要的参数有：

- `--env`：`params.ini` 中的节名
- `--seed`：随机种子
- `--prefix`：输出文件名前缀
- `--cuda`：GPU 编号
- `--fre`：13 维 motif 目标频率
- `--epoch1`：motif 预训练轮数
- `--numOfNeuron`：RNN hidden size，默认 `512`
- `--batchSize`：PPO mini-batch 大小，默认 `64`
- `--amplitude`：权重二值化近似的放大系数
- `--bias`：边存在与否的阈值
- `--discrete`：是否额外评估离散 hidden state 版本

## 已知问题与注意事项

- 仓库当前缺少 `params.ini`，需要手动补。
- 没有依赖锁定文件，环境需要自行配置。
- 代码默认直接使用 `cuda:<id>`，没有 CPU fallback。
- `run_batch.sh` 只覆盖了一个环境和一组默认实验设置。
- 仓库使用旧版 MuJoCo / Gym 环境命名，现代环境里可能无法直接运行。

## 适合谁用

这个仓库更像一个最小研究原型，适合：

- 快速复现 motif regularization 的核心实现
- 修改循环权重上的图结构约束
- 在 PPO 上做小规模实验

如果你想要一个开箱即用、依赖完整、可直接复现整套实验结果的工程版本，这个仓库本身还不够完整，至少需要先补齐配置文件和运行环境。
