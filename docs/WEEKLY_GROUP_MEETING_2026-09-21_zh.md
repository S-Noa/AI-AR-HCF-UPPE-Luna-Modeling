# 本周项目进展汇报（2026-09-14 至 2026-09-21）

## 1. 本周一句话结论

本周完成了受控的 Simple/Complex RNN benchmark、Moderate 候选池和
5 cm constrained-RL 物理验证，并恢复了干净、可复现的云端 Git 工作区。
受控实验表明：RNN 在平滑 Simple 谱形上可以稳定完成 200 步递推；在更复杂
的 AR-HCF 谱形上，one-step 仍然很高，但完整 autoregressive 的稳定性显著下降。
目标表示与训练实现均会影响这一差距，因此下一步应补齐统一频域演化图和
Keras/PyTorch 实现对齐，而不是直接宣称单一物理原因。

## 2. 本周完成事项

| 工作线 | 已完成内容 | 当前结论 |
| --- | --- | --- |
| Luna RNN 诊断 | 完成 raw-power + original dBm 与原文 SC 数据的定量比较；新增固定 1/2/3/4-step 递推评估。 | Luna 早期强重构与长 rollout 的误差累积是主要难点；短步反馈本身不一定立即失稳。 |
| 受控复杂度 benchmark | 生成并筛选 Simple/Complex 数据；完成 2 数据集 x 2 表示 x 2 框架 x 3 seed，共 24 组训练。 | 谱形复杂度和归一化都会影响完整递推；Keras 与 PyTorch 在 Complex 上存在需继续审查的差异。 |
| Moderate bridge set | Smoke 图册评审后，完整生成 1600 条 Moderate 候选。 | 可作为 Simple 与 Complex 之间的中间谱形池；尚未定义中等复杂度筛选区间，也尚未训练。 |
| RL inverse design | 完成 raw4 50 cm 目标退化诊断；完成 z=5 cm constrained-RL smoke 与 10 条 Luna 验证。 | 单纯 UV fraction 会奖励近乎能量耗尽的解；约束 UV power 路线避免了该退化，但仍只是单 seed 可行性验证。 |
| 工程环境 | 云端旧 Git 工作区恢复为 clean clone；完成 Julia/Python 环境验证。 | 本地 GitHub main 现在是唯一源码基线，云端只保留代码 checkout 和运行产物。 |

## 3. RNN：从原文复现到受控复杂度实验

### 3.1 已完成的问题拆分

此前 Luna early-dense RNN 出现“teacher-forced stepwise 很高，但 full
autoregressive 为负”的现象。本周将问题拆成两层：

1. 原文数据与原文式预处理是否能被我们的 RNN 递推；
2. 在统一的 Luna 几何、序列长度和波长维度下，仅改变谱形复杂度时，递推性能如何变化。

原文 `SC_spec_251` 上，PyTorch RNN 可获得约 `autoregressive R2=0.9633`，
说明 PyTorch 训练/评估主链路能够在原文类型任务上稳定递推。此前的 common-dBm
诊断也表明，Luna 数据在原文 `-55 dB` 映射后更稀疏，且最大相邻 z 变化发生在输入端，
更不利于局部窗口递推。

### 3.2 受控 benchmark 设计

所有轨迹统一为 Ar、壁厚 `0.65 um`、长度 `0--5 cm`、200 个均匀 z 保存点、
`200--2500 nm` 的 251 个波长点、`window_size=10`。每类候选池为 1600 条，按
演化图复杂度选择 1300 条，其中 1250 条训练、50 条测试。

| 类别 | Energy | Duration | Pressure | Core diameter | 目的 |
| --- | --- | --- | --- | --- | --- |
| Simple | 0.3--1.0 uJ | 25--50 fs | 0.5--15 bar | 30--50 um | 平滑、近稳态谱形的可递推基线。 |
| Complex | 2.0--2.5 uJ | 10--16 fs | 10--25 bar | 18--25 um | 数值可运行的强非线性、快速重构谱形。 |
| Moderate | 0.7--1.6 uJ | 15--35 fs | 5--25 bar | 22--40 um | Simple 与 Complex 之间的桥接候选池。 |

Complex 的初始场强增加了 `0.5 x PPT` 表格上限保护；原先更极端的参数范围会在
5 cm 内自压缩到 PPT 插值表范围以外，因此不能把数值失败当作“复杂谱形样本”。

### 3.3 完整 24 组训练结果

训练统一使用 250-hidden LSTM、两层 250-hidden Dense、sigmoid 输出、MSE、
RMSprop 和 `50 + 30` epoch 调度。下表是 3 个随机 seed 的均值 +/- 标准差，
`AR R2` 为完整 200 点 rollout，`Final AR R2` 是终点谱。

| Framework | Data | Representation | Stepwise R2 | AR R2 | Final AR R2 |
| --- | --- | --- | ---: | ---: | ---: |
| Keras | Simple | original dBm | 1.0000 +/- 0.0000 | 0.9865 +/- 0.0018 | 0.9612 +/- 0.0059 |
| PyTorch | Simple | original dBm | 0.9999 +/- 0.0001 | 0.9889 +/- 0.0006 | 0.9692 +/- 0.0031 |
| Keras | Simple | per-sample min-max | 0.9996 +/- 0.0000 | 0.8492 +/- 0.0104 | 0.6818 +/- 0.0321 |
| PyTorch | Simple | per-sample min-max | 0.9994 +/- 0.0000 | 0.9638 +/- 0.0009 | 0.9123 +/- 0.0016 |
| Keras | Complex | original dBm | 0.9979 +/- 0.0000 | 0.1519 +/- 0.1065 | 0.0825 +/- 0.1211 |
| PyTorch | Complex | original dBm | 0.9774 +/- 0.0085 | 0.7913 +/- 0.0201 | 0.6972 +/- 0.0853 |
| Keras | Complex | per-sample min-max | 0.9903 +/- 0.0001 | 0.5514 +/- 0.2142 | 0.3222 +/- 0.2525 |
| PyTorch | Complex | per-sample min-max | 0.9729 +/- 0.0105 | 0.7887 +/- 0.0056 | 0.7078 +/- 0.0107 |

**可以在组会上报告的结论：**

- 对 Simple 数据，原版 Keras 和 PyTorch 都达到约 `0.99` 的完整递推 R2，说明
  “局部谱窗口递推”在平滑传播轨迹上可行。
- 对 Complex 数据，Keras 的完整 AR R2 大幅下降，且 original-dBm 的均值仅约
  `0.15`；per-sample min-max 可提升到约 `0.55`，说明目标表示会实质影响递推稳定性。
- PyTorch 在 Complex 上约为 `0.79`，明显高于 Keras。这是重要但尚未完全归因的
  框架/实现差异：两边虽对齐了主要网络和优化设置，仍需检查 LSTM 单元实现、
  batch/window 构造、checkpoint 选择时机及 rollout 细节。不能只把 Keras 的失败
  归因于物理复杂度。

### 3.4 固定 1--4 step 诊断

固定 horizon 评估从真实历史窗口起步，只在随后的 `k` 步使用模型输出反馈。Complex
任务中，PyTorch original-dBm 的 all-start R2 平均从 `0.9774`（1 step）到 `0.9752`
（4 step）；Keras original-dBm 从 `0.9979` 到 `0.9824`。这说明在 1--4 步范围内
局部预测仍然很好。

因此，full rollout 失败并不是“第 2 或第 3 步立刻崩溃”，而是小误差在 200 个保存点
上持续进入输入窗口后的长期累积。该诊断也解释了为什么 early-dense 1401 点任务比
200 点 controlled benchmark 更困难。

### 3.5 频域演化图材料

组会建议放三类图，而非只放 R2 表：

1. Simple/Complex 真实 target 对比：
   `rnn_visual_diagnostics/rnn_complexity_benchmark/visualizations/representative_simple_complex_targets.png`。
2. Simple 原版 dBm 的 target、stepwise、autoregressive 演化图和最终谱：
   `rnn_visual_diagnostics/rnn_complexity_benchmark/visualizations/pytorch_simple_original_dbm_seed123/`。
3. Moderate 最终 16 分位图册（本周已同步本地）：
   `rnn_visual_diagnostics/rnn_complexity_benchmark/moderate_visualizations/moderate_target_quantile_gallery.png`。

Complex 的 Keras/PyTorch 完整频域演化图尚未按同一模板批量导出；这应作为下一步，
否则不能仅凭表格判断谱形层面的失败模式。

## 4. Moderate 数据集进展

Moderate smoke 图册确认其主流谱形不是 Simple 中几乎静止的 1030 nm 窄泵浦，且没有
Complex 的强碎裂主导行为。因此已完成独立的 1600 条候选生成，并将最终图册同步到本地。

下一步不是直接训练，而是根据 `complexity_score` 与图册共同定义中等复杂度区间：
排除接近 Simple 的低分位样本和接近 Complex 的高分位样本，再导出 Moderate 的
1250/50 split。这样 Moderate 才是“中间物理复杂度”的受控实验，而非混合数据集。

## 5. RL inverse design：从退化奖励到可验证闭环

### 5.1 已完成结果

- Raw4 final-spectrum forward surrogate：`Final_LogPower_R2=0.9831`，
  `UV_LogPower_R2=0.9585`；UV-power-proxy Spearman 为 `0.9438`。
- z=5 cm raw4 forward surrogate：`Final_LogPower_R2=0.9855`，
  `UV_LogPower_R2=0.9647`；UV-power-proxy Spearman 为 `0.9804`。
- 50 cm 的纯 UV fraction 奖励出现退化：20 个去重 SAC 候选中 11 条可运行，
  但能量传输仅 `2.17e-20--3.02e-13`，即模型学到的是“耗尽总能量后短波占比高”，
  不是可用的 UV 输出。
- 改为 z=5 cm constrained UV-power reward 后，单 seed smoke 的 10 条候选均可由
  Luna 运行；Luna UV fraction 为 `0.0300--0.0623`，能量传输为
  `6.09e-7--8.44e-7`，最大电子密度为 `2.78e17--3.22e19 m^-3`。

### 5.2 当前定位

这证明了“surrogate -> RL candidate -> Luna validation”的闭环能跑通，但不构成
最优设计结论。下一版 reward 应基于完整 `S(z,lambda)`：奖励局域、连续、可追踪的
UV 色散波，同时惩罚过低总输出功率、过强电离和宽带噪声状短波结构。

可用于汇报的 Luna 演化图：
`rl_inverse_design/raw4_z5cm_constrained_rl_smoke_v1/luna_validation/analysis/luna_spectral_evolution_montage.png`。

## 6. 云端工程恢复

云端旧 Git checkout 停留在历史提交，并积累了未提交脚本和环境目录。本周完成以下恢复：

1. 备份旧 worktree 的 patch、源码文件清单和未跟踪源码归档；
2. 新建 GitHub clean clone 并确认 38 个旧未跟踪源码文件均与 GitHub 内容一致；
3. 旧目录保留为 legacy，新 clean clone 接管标准路径
   `/mnt/AI-AR-HCF-UPPE-Luna-Modeling`；
4. 修复共享 Julia depot 的单线程 FFTW wisdom，并重建 PyCall/PyPlot；
5. 验证 `using Luna`、`data_preprocessing.py --help` 与 `train_mlp.py --help`；
6. 规定所有汇报/评审用可视化必须同步到本地。

详细记录见 `docs/CLOUD_WORKTREE_RECOVERY_2026-09-21.md`。

## 7. 下周优先级

1. **补齐 benchmark 可视化**：为 Complex 的 Keras 与 PyTorch 两种表示生成同一批
   target、stepwise、autoregressive、最终谱和逐步 R2 图，并同步本地。
2. **审查框架差异**：逐项比对 Keras 与 PyTorch 的 LSTM gate activation、初始化、
   batch 顺序、learning-rate schedule、best checkpoint 和 rollout 代码；确认是否是
   实现差异放大了 Complex 结果差距。
3. **导出 Moderate 子集**：先固定中等复杂度选择规则，再启动 Moderate RNN 对照，
   形成 Simple--Moderate--Complex 梯度证据。
4. **升级 RL reward**：从 final-spectrum proxy 迁移到 temporal clean-UV reward，
   并以 Luna 演化图、输出功率和电子密度共同验收，而不是仅优化 UV fraction。
5. **更新论文/开题叙事**：RNN 定位为局部传播与误差累积诊断 baseline；完整传播图的
   主路线仍是直接预测的 CNN/Transformer surrogate。

## 8. 建议 PPT 页序（6 页）

1. **本周目标与结论**：RNN 递推问题、受控 benchmark、RL 闭环三条线。
2. **为什么需要受控 benchmark**：原文 SC 可递推，Luna early-dense 易漂移；展示
   Simple/Complex 真实演化图。
3. **实验设计**：统一 0--5 cm、200 z、251 lambda；展示三类参数范围与复杂度逻辑。
4. **核心结果**：放 24 组聚合表，突出 Simple 与 Complex、dBm 与 min-max、Keras 与
   PyTorch 的差异。
5. **机制解释与 Moderate**：固定 1--4 step 曲线说明短步未崩、长 rollout 才累积；
   放 Moderate 分位图册，并说明下一步筛选。
6. **RL 与工程闭环 / 下一步**：50 cm reward 退化 -> 5 cm constrained smoke ->
   clean-UV temporal reward；最后列出下周五项任务。

## 9. 汇报时的建议表述

> We have separated immediate local prediction from long-horizon recurrence.
> On controlled simple trajectories, both implementations reproduce the
> original RNN behavior. On complex Luna trajectories, one-step accuracy stays
> high but full rollout becomes representation- and implementation-sensitive.
> Therefore, the next task is not to claim that RNNs universally fail, but to
> visualize the failure mode, align the two implementations, and construct a
> controlled Moderate regime between the two extremes.
