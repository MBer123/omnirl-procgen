# OmniRL-Procgen

[English](README.md) | 简体中文

**SPPO: A Practical Recipe for Strong PPO in Procgen**

OmniRL-Procgen 用于在统一实验设置下比较 PPO 与 SPPO 在程序生成游戏环境中的表现。项目基于 [OmniRL](https://github.com/MBer123/omnirl) 提供的分布式强化学习框架；框架架构、运行时和公共接口请参阅 OmniRL 仓库。

## 方法

SPPO 是一套面向视觉 On-policy 强化学习的 PPO 改进方案。它将表征学习、归一化和面向数据增强的优化方法适配到 PPO，并针对 actor 与 critic 采用不同的处理方式。

| 组成部分 | 在 SPPO 中的作用 |
| --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347) | 提供 clipped surrogate objective 和基础 Actor-Critic 更新。 |
| [SPR](https://arxiv.org/abs/2007.05929) | 使用 action-conditioned transition model 和 EMA target network，预测未来多步 latent representation。 |
| [RMSNorm](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html) | 改进 [IMPALA-CNN](https://proceedings.mlr.press/v80/espeholt18a.html) backbone，在 latent projection 前对展平后的卷积特征进行归一化。 |
| Separate Heads | actor 和 critic 使用独立 head：actor 采用 [LayerNorm](https://arxiv.org/abs/1607.06450)，critic 采用 RMSNorm。 |
| [SADA](https://rlj.cs.umass.edu/2024/papers/Paper26.html)-actor | 借鉴 SADA 的思想，分别使用原始观测和 random-shift 观测计算 PPO actor loss，再对两者取平均。 |
| [DrAC](https://proceedings.neurips.cc/paper_files/paper/2021/hash/2b38c2df6a49b97f706ec9148ce48d86-Abstract.html)-actor | 使用 KL divergence 约束增强后的 policy 与原始 policy 保持一致。 |
| [SVEA](https://proceedings.neurips.cc/paper/2021/hash/1e0f65eb20acbfb27ee05ddc000b50ec-Abstract.html)-critic | 借鉴 SVEA 的思想，让原始观测与增强观测预测的 value 回归到同一个 return target。 |
| RewardNorm | 使用 discounted return 的运行标准差缩放 reward，在多个 Learner 间同步统计量，并裁剪归一化后的 reward。 |

记 $g$ 为 random-shift augmentation，$`\hat{R}`$ 为使用归一化 reward 计算的 return target，$`\mathrm{sg}`$ 表示 stop-gradient。PPO clipped surrogate loss 定义为

```math
\begin{aligned}
\mathcal{L}_{\mathrm{PPO}}^{\mathrm{clip}}(x)
&=
-\mathbb{E}_t
\left[
\min\left(
r_t(x)\hat{A}_t,\,
\mathrm{clip}\left(r_t(x),1-\epsilon,1+\epsilon\right)\hat{A}_t
\right)
\right], \\
r_t(x)
&=
\frac{\pi_\theta(a_t \mid x)}
{\pi_{\mathrm{old}}(a_t \mid s_t)},
\qquad
x \in \{s_t,g(s_t)\},
\end{aligned}
```

其中，$`\hat{A}_t`$ 为归一化后的 advantage estimate，$`\epsilon`$ 为 PPO clipping range。actor 和 critic 的数据增强 loss 分别为

```math
\begin{aligned}
\mathcal{L}_{\mathrm{actor}}
&=
\frac{1}{2}\mathcal{L}_{\mathrm{PPO}}^{\mathrm{clip}}(s)
+
\frac{1}{2}\mathcal{L}_{\mathrm{PPO}}^{\mathrm{clip}}(g(s)), \\
\mathcal{L}_{\mathrm{critic}}
&=
\frac{1}{2}\lVert V(s)-\hat{R}\rVert_2^2
+
\frac{1}{2}\lVert V(g(s))-\hat{R}\rVert_2^2.
\end{aligned}
```

DrAC-actor 额外加入 policy consistency loss：

```math
\mathcal{L}_{\mathrm{DrAC}}
=
D_{\mathrm{KL}}
\left(
\mathrm{sg}[\pi(\cdot \mid s)]
\;\middle\|\;
\pi(\cdot \mid g(s))
\right).
```

最终优化目标为

```math
\mathcal{L}_{\mathrm{SPPO}}
=
\mathcal{L}_{\mathrm{actor}}
+ c_v\mathcal{L}_{\mathrm{critic}}
- c_H\mathcal{H}(\pi)
+ c_{\mathrm{SPR}}\mathcal{L}_{\mathrm{SPR}}
+ c_{\mathrm{DrAC}}\mathcal{L}_{\mathrm{DrAC}},
```

其中，各项 loss 的系数和 SPR prediction horizon 均由 Agent YAML 配置。

## 环境

Procgen Benchmark 由一组程序生成游戏环境组成，主要用于评估强化学习算法的样本效率和泛化能力。

所有实验均采用 `hard` distribution mode，并在各实验配置中分别指定训练与测试 Level 的范围。

| 环境 | 任务说明 |
| --- | --- |
| `bigfish` | 控制小鱼吞食体型更小的鱼，同时躲避体型更大的鱼。 |
| `bossfight` | 控制飞船与 Boss 战斗，在躲避攻击的同时寻找输出机会。 |
| `caveflyer` | 驾驶飞行器穿过洞穴，在避开墙壁、障碍物和敌人的同时抵达出口。 |
| `chaser` | 在类似 Pac-Man 的迷宫中收集目标，并应对追逐角色的敌人。 |
| `climber` | 在平台之间向上攀爬并收集奖励。 |
| `dodgeball` | 在房间内躲避敌人和投射物，并击败敌人完成 Level。 |
| `heist` | 探索迷宫，收集钥匙，打开对应的锁并取得宝石。 |
| `miner` | 挖掘地图、收集钻石，同时避开落石等危险。 |
| `starpilot` | 在横向卷轴射击场景中躲避敌人与子弹，并摧毁目标。 |

游戏规则、环境选项和原始实现详见 [openai/procgen](https://github.com/openai/procgen)。

## 安装

OmniRL-Procgen 需要 64 位 Python 3.10，因为 Procgen 0.10.7 不提供 Python 3.11 及以上版本的安装包。首先创建虚拟环境：

```bash
python -m venv .venv
```

在 Windows 上激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

在 Linux 和 macOS 上：

```bash
source .venv/bin/activate
```

安装 OmniRL-Procgen 和 Procgen 环境依赖：

```bash
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements/procgen.txt
```

Procgen 的依赖文件会引入 `base.txt`，因此可以在全新环境中直接安装。

如需使用 GPU 训练，请安装与操作系统和 CUDA 运行时相匹配的 PyTorch 版本，然后确认 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 快速开始

运行 OmniRL-Procgen 需要两份 YAML 配置：

- Runner 配置用于定义 Procgen 环境、Ray Worker 数量、设备、随机种子、检查点和日志。
- Agent 配置用于定义 PPO 或 SPPO，以及 `Learner`、`Memory`、`Policy` 和网络参数。

在单机上启动 BigFish 的 SPPO 训练：

```bash
python run.py --mode train --runner-config experiments/bigfish/sppo-0/train.yaml --agent-config experiments/bigfish/sppo-0/sppo.yaml
```

如果检测到可用的 Ray 集群，`run.py` 会直接连接；否则会启动本地 Ray 运行时。多节点运行时，先启动 Head 节点：

```bash
ray start --head --port=6379
```

让各 Worker 节点加入集群，然后在 Head 节点上启动 `run.py`：

```bash
ray start --address=<head-ip>:6379
```

实验配置遵循 `experiments/<environment>/<algorithm>-<seed>` 的目录结构。当 `runner.load_checkpoint` 为 `true` 时，重新执行相同的训练命令会从 `runner.ckpt_dir` 中的 `checkpoint.ckpt` 恢复训练。

### 输出

每次运行都会将以下文件写入 `runner.ckpt_dir`：

- `checkpoint.ckpt`：用于恢复训练的完整 `Learner` 状态和优化器状态。
- `agent_<version>.pth`：导出的策略权重。
- `log.csv`：运行时、评估和 `Learner` 指标。
- `run_config.yaml`：本次运行实际使用的 Runner 与 Agent 合并配置文件。

## 评估

在对应的 `test.yaml` 中设置 `runner.ckpt_dir` 和 `runner.eval_model`，然后运行：

```bash
python run.py --mode test --runner-config experiments/bigfish/sppo-0/test.yaml --agent-config experiments/bigfish/sppo-0/sppo.yaml
```

Procgen 环境模块会分别评估配置中的训练与测试 Level，并给出回报的均值、标准差、最大值和最小值。

如需观看 Agent 运行，请在测试配置中将 `env.render_mode` 设置为 `human`，并将 `runner.num_evaluators` 设置为 `1`。

## 实验

实验使用 PPO 和 SPPO，在九个 Procgen 环境上进行对比：

| 设置 | 值 |
| --- | --- |
| 算法 | PPO、SPPO |
| 环境 | BigFish、BossFight、CaveFlyer、Chaser、Climber、DodgeBall、Heist、Miner、StarPilot |
| 随机种子 | 0、1、2 |
| Distribution mode | `hard` |
| 训练 Level | 500 |
| 留出测试 Level | 100 |
| 每次运行的参数更新次数 | 3,052 |
| 总运行次数 | 54 |

配置位于 `experiments/<environment>/<algorithm>-<seed>`，检查点、日志和合并后的运行配置将写入 `results/<environment>/<algorithm>-<seed>`。

下图中，实线表示三个随机种子的训练分数均值，阴影区域表示一个标准差。

<p align="center">
  <img src="docs/images/bigfish.png" alt="PPO 与 SPPO 在 BigFish 上的实验结果" width="32%">
  <img src="docs/images/bossfight.png" alt="PPO 与 SPPO 在 BossFight 上的实验结果" width="32%">
  <img src="docs/images/caveflyer.png" alt="PPO 与 SPPO 在 CaveFlyer 上的实验结果" width="32%">
</p>

<p align="center"><em>BigFish（左）、BossFight（中）和 CaveFlyer（右）。</em></p>

<p align="center">
  <img src="docs/images/chaser.png" alt="PPO 与 SPPO 在 Chaser 上的实验结果" width="32%">
  <img src="docs/images/climber.png" alt="PPO 与 SPPO 在 Climber 上的实验结果" width="32%">
  <img src="docs/images/dodgeball.png" alt="PPO 与 SPPO 在 DodgeBall 上的实验结果" width="32%">
</p>

<p align="center"><em>Chaser（左）、Climber（中）和 DodgeBall（右）。</em></p>

<p align="center">
  <img src="docs/images/heist.png" alt="PPO 与 SPPO 在 Heist 上的实验结果" width="32%">
  <img src="docs/images/miner.png" alt="PPO 与 SPPO 在 Miner 上的实验结果" width="32%">
  <img src="docs/images/starpilot.png" alt="PPO 与 SPPO 在 StarPilot 上的实验结果" width="32%">
</p>

<p align="center"><em>Heist（左）、Miner（中）和 StarPilot（右）。</em></p>

## 注意事项

- 使用 CPU 训练时，请将 `device.learner` 设置为 `cpu`，并保持 `runner.num_learners` 为 `1`。
- 请根据可用硬件调整 `runner.num_samplers`、`runner.num_evaluators` 和 `runner.num_learners`。初次配置时，建议将三者总数控制在可用 CPU 线程数的 90% 以下。
- 训练与评估使用的 `env.num_stack` 应保持一致。
- 使用 `python -m unittest discover -s tests -v` 运行测试套件。
- OmniRL-Procgen 面向强化学习研究。基于多个随机种子的实验结果仅代表当前实验条件，不构成生产级保证。

## 许可证

Copyright 2026 Huang Chenbiao (MBer123).

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。
