# 本周组会：RNN Baseline 归因与下一步

## 汇报目的

本次组会聚焦回答一个问题：

> 为什么 RNN 在 Luna AR-HCF 数据上具有很高的 teacher-forced 单步预测精度，却无法稳定 autoregressive 递推；而原始论文的 RNN 在其超连续谱任务上可以成功递推？

本周可支撑的结论应保持克制：RNN 的实现与代码迁移在原始任务上有效；但当前 Luna 数据的局部自回归目标表示更难稳定递推。短程 RNN baseline 可以使用，而完整传播图预测仍应以直接输出全图的 CNN/Transformer 为主。

## 建议 PPT 页序（7 页，10--12 分钟）

### 第 1 页：问题与本周目标（约 1 分钟）

**标题：** `Can the RNN Autoregressively Model Luna AR-HCF Propagation?`

**页面内容：**

- 原始 RNN 论文：单步与 autoregressive 光谱演化预测均表现良好。
- 早期 Luna 结果：`stepwise R2` 约为 `0.994`，但完整 autoregressive `R2` 约为 `-0.410`。
- 本周工作：从代码、预处理、递推步数和数据动力学四方面进行受控归因。

**口头表述：**

“问题并不是 RNN 能否拟合下一个传播截面的光谱，实验表明它可以做到。真正的问题是：当模型必须把自身预测重新输入时，局部光谱窗口能否稳定地表征传播状态。”

### 第 2 页：受控归因矩阵（约 1.5 分钟）

**标题：** `Separate Code Effects from Data Effects`

建议使用下表：

| 训练/评估组合 | Stepwise R2 | Autoregressive R2 | 含义 |
|---|---:|---:|---|
| 原始 Keras RNN + 原始 SC 数据 | 0.9990 | 0.8559 | 原始工作流可以复现。 |
| PyTorch RNN + 原始 SC 数据 | 0.9989 | 0.9633 | PyTorch 迁移在原始任务上有效。 |
| 原始 Keras RNN + Luna，10 cm/51 z/251 lambda | 0.9743 | 0.3161 | 原始实现用于 Luna 时同样较弱。 |
| 原始 Keras RNN + Luna raw power + 原始 dBm | 0.9794 | 0.3071 | 仅匹配原始预处理无法解决问题。 |

**数据来源：** `rnn_visual_diagnostics/rollout_experiments/summary_metrics.csv`。

**口头表述：**

“这是最关键的归因结果。两种实现都能在原始 SC 数据上成功，而未经结构修改的 Keras 原始代码放到降维 Luna 数据上也出现递推退化。因此，代码迁移并不是主要原因。”

### 第 3 页：短程 Luna Baseline，什么是可用的（约 1.5 分钟）

**标题：** `A Usable RNN Baseline Exists Only at Short Horizon`

建议使用下表或画一条 R2 随递推步数变化的曲线：

| 前 10 cm 子任务 | 递推预测步数 | Autoregressive R2 | 最后截面 R2 |
|---|---:|---:|---:|
| 51 个 z 点，1000 个波长点 | 41 | 0.4873 | 0.3301 |
| 101 个 z 点，1000 个波长点 | 91 | 0.4339 | 0.2839 |
| 201 个 z 点，1000 个波长点 | 191 | 0.3716 | 0.2430 |

页脚注明：三组实验采用相同的 `per_sample_minmax` 目标、`features_z` 条件输入、直接 sigmoid 预测、低比例 scheduled sampling feedback、no-detach feedback 与梯度裁剪。

**口头表述：**

“在其他训练条件保持一致时，只增加递推反馈次数，性能就持续下降。这说明即使只看前 10 cm，rollout 长度也是一个实质性瓶颈。因此，51 点设置可以作为当前可汇报的 RNN baseline，但不应被表述为完整传播图预测方案。”

### 第 4 页：同一 dBm 标度下的稀疏性与早期重构（约 2 分钟）

**标题：** `Why the Same dBm Mapping Is Less RNN-Friendly for Luna`

**主图：**
`rnn_visual_diagnostics/data_dynamics_comparison/data_dynamics_ppt_overview.png`

重点指向：

- 目标值分布：Luna 有 `61.45%` 的值被截断为 0；原始 SC 为 `26.41%`。
- 相邻 z 截面平均变化：Luna 最大变化位于归一化传播位置 `0`；原始 SC 的峰值约在 `0.207`。
- Autoregressive R2 曲线：原始任务保持稳定，Luna 随递推逐步退化。

**建议在页面显著位置写出的限定：**

`传播轴经归一化后仅用于比较动力学位置；两个物理系统的实际长度尺度并不相同。`

**口头表述：**

“这并不意味着 dBm 标度在物理上错误，而是说明原始任务使用的全局最大值和 -55 dB 截断，在 Luna 光谱上形成了更稀疏的目标空间。同时，Luna 的最强平均重构恰好发生在输入端，也就是 autoregressive rollout 的起点。”

### 第 5 页：代表性目标演化图与局部可预测性（约 1.5 分钟）

**标题：** `Luna Trajectories Are More Diverse Under the Same Recurrent Representation`

**主图：**
`rnn_visual_diagnostics/data_dynamics_comparison/representative_targets_original_vs_luna.png`

**可选辅助图：**
`rnn_visual_diagnostics/data_dynamics_comparison/local_window_ambiguity_original_vs_luna.png`

**页面内容：**

- 所有热图的横轴为波长采样点编号，纵轴为传播步编号。
- 四行分别是按 early-z 变化强度排序的 P10/P50/P90/P99 样本；它们是统计代表样本，不是物理参数一一匹配的样本。
- 共享 PCA 局部窗口诊断：
  - 最近历史窗口距离中位数：Luna 为 `1.764`，原始 SC 为 `0.498`；
  - 相近历史窗口的下一步光谱差异中位数：Luna 为 `0.00893`，原始 SC 为 `0.00339`。

**口头表述：**

“将两类数据的历史窗口标准化后投影到同一个 PCA 空间，Luna 的窗口分布更分散；即使最近的历史窗口，其下一步也更不相似。这是描述局部递推可预测性的统计量，并不能单独证明唯一的物理因果机制。”

### 第 6 页：本周结论与模型定位（约 1.5 分钟）

**标题：** `Model Roles After This Week's Experiments`

建议采用三行结构图或表格：

| 模型 | 合适的角色 | 当前限制 |
|---|---|---|
| RNN | 局部/短程 autoregressive baseline；用于研究误差累积 | 长程递推会退化，尤其是 early-dense 全图。 |
| Temporal CNN | 直接预测传播图的工程强 baseline | 需要在相同 early-dense 数据与指标下和 Transformer 公平比较。 |
| Transformer | 论文主模型；通过直接全图预测建模全局 z 关系 | 窄 UV 结构仍存在过平滑。 |

**口头表述：**

“RNN 的结果仍然有价值：它提供了一个受控的反例，高单步 R2 不代表稳定的长程传播能力。对于完整的 `S(z, lambda)` 任务，CNN 和 Transformer 通过一次前向输出整张传播图，在机制上避免了递推反馈误差。”

### 第 7 页：建议决策与下周计划（约 1 分钟）

**标题：** `Recommended Next Decisions`

**建议明确提出以下决策：**

1. 将 `z <= 10 cm、51 个 z 点、1000 个波长点` 固定为当前可汇报的 RNN autoregressive baseline。
2. 暂停更多 dB floor 扫描：原始预处理已被严格匹配，但没有解决核心递推问题。
3. 启动 CNN 与 Transformer 在相同 early-dense 数据切分、相同 temporal metric 下的重新训练与评估。
4. RNN 保留为局部递推诊断模型；inverse design 继续建立在验证后的直接 forward surrogate 上。

**建议的后续实验：**

- CNN 与 Transformer 在同一 early-dense 数据集、同一目标表示和同一 temporal 指标下的公平对比。
- 仅在确有必要时再开展有限 RNN 消融：输入谱加物理特征的状态增强，或多步 latent-state RNN；但不要预设其一定可以解决完整 rollout。
- 用直接 surrogate 做带约束的参数优化，再以 Luna 仿真验证候选解。

## 图像清单

| 页面 | 文件 | 用途 |
|---|---|---|
| 第 4 页 | `rnn_visual_diagnostics/data_dynamics_comparison/data_dynamics_ppt_overview.png` | 六面板主证据图。 |
| 第 5 页 | `rnn_visual_diagnostics/data_dynamics_comparison/representative_targets_original_vs_luna.png` | 真实目标轨迹对比。 |
| 第 5 页 | `rnn_visual_diagnostics/data_dynamics_comparison/local_window_ambiguity_original_vs_luna.png` | 可选的局部可预测性定量支撑。 |
| 备用 | `rnn_visual_diagnostics/rawpower_dbm_failure/per_step_autoreg_r2_original_vs_luna.png` | 直接展示逐步 autoregressive R2。 |
| 备用 | `rnn_visual_diagnostics/rawpower_dbm_failure/sample_r2_distribution_original_vs_luna.png` | 展示 Luna 样本间失败程度不均匀。 |
| 备用 | `rnn_visual_diagnostics/rawpower_dbm_failure/target_value_hist_original_vs_luna.png` | 单独展示 dBm 处理后的稀疏性。 |

## 可能问题与简短回答

**为什么不直接采用原文预处理？**  已经严格测试：Luna 的 raw linear power 被送入原始 `dBm` 预处理流程后，原始 Keras RNN 的 autoregressive R2 仍只有 `0.3071`。所以预处理不一致不是唯一或主要原因。

**这是否证明 RNN 不能预测 AR-HCF 传播？**  不能。它表明当前“局部光谱窗口递推”的 RNN 形式在现有 early-dense Luna 表示上不适合长程 rollout；作为短程 baseline 仍然可以使用。

**为什么 PyTorch 在原始 SC 数据上的结果还高于原始 Keras？**  两套训练细节与 checkpoint 选择未做到逐项完全相同，因此这里只说明实现具备该任务能力，不将其表述为性能超越原文。

**将两个系统的 z 轴归一化比较是否合理？**  它不用于比较实际物理长度，只用于比较各自轨迹中变化发生的位置和递推动力学特征。

**为什么不无限继续优化 RNN？**  项目的核心目标是完整传播图的准确 forward prediction。直接预测全图的 CNN/Transformer 本身没有逐步反馈误差，更值得投入主要资源。

## 一句话收束

“本周的受控交叉验证表明，Luna RNN 的 rollout 瓶颈主要与 AR-HCF 数据在当前递推表示下的目标动力学有关，而不只是代码迁移或 dBm 预处理错误；下一步将其定位为短程诊断 baseline，并优先推进直接全图预测模型。”
