# 本周项目进展汇报（2026-09-14 至 2026-09-21）

## 1. 本周一句话结论

本周完成了受控的 Simple/Complex RNN benchmark、Moderate 候选池和
5 cm constrained-RL 物理验证，并恢复了干净、可复现的云端 Git 工作区。
受控实验表明：RNN 在平滑 Simple 谱形上可以稳定完成 200 步递推；在更复杂
的 AR-HCF 谱形上，one-step 仍然很高，但完整 autoregressive 的稳定性显著下降。
目标表示与训练实现均会影响这一差距，因此下一步应补齐统一频域演化图和
Keras/PyTorch 实现对齐，而不是直接宣称单一物理原因。Moderate 的
12 组 Keras/PyTorch 对照训练及其统一频域演化图导出已于本周末启动。

## 2. 项目整体工作逻辑

当前项目不是彼此独立地训练若干模型，而是围绕“高保真物理仿真--快速前向预测--
可信逆向设计”建立三层闭环：

```text
Luna/UPPE 高保真仿真
        |
        +--> 数据集、复杂度分层和物理可运行性判据
        |
        +--> 前向 surrogate
        |      |- CNN/Transformer：直接预测完整 S(z, lambda)，是主模型路线
        |      `- RNN：局部递推 baseline，用于分析误差累积与可递推性边界
        |
        `--> inverse design
               surrogate 作为快速环境 -> 优化/RL 给出候选 -> Luna 回代验证
               -> 将失败或高价值样本反馈到下一轮数据与模型
```

因此各工作线的职责不同：

| 工作线 | 回答的核心问题 | 当前定位 | 与其它线的关系 |
| --- | --- | --- | --- |
| 数据与 Luna | 哪些参数和物理状态可运行，真实传播图是什么 | 物理真值与最终裁判 | 为所有模型训练、评估和候选验证提供依据 |
| Transformer/CNN | 能否由参数直接快速预测最终谱和完整传播图 | 当前主前向模型 | 为逆向设计提供可微或快速的 surrogate 环境 |
| RNN | 局部谱窗口能否稳定递推，误差在哪个 horizon 开始累积 | baseline 与诊断工具 | 不替代主模型；解释直接 full-map 预测为何更稳健 |
| Controlled benchmark | 谱形复杂度、目标表示和框架实现分别造成多大影响 | RNN 归因实验 | 将 early-dense 的复杂问题拆成可控证据 |
| RL inverse design | 如何在可制造参数范围内自动搜索有价值工作点 | 逆向设计探索路线 | 必须建立在已验证 forward surrogate 与 Luna 回代之上 |

## 3. 本周完成事项

| 工作线 | 已完成内容 | 当前结论 |
| --- | --- | --- |
| Luna RNN 诊断 | 完成 raw-power + original dBm 与原文 SC 数据的定量比较；固定递推诊断已扩展到 1--10 step。 | Luna 早期强重构与长 rollout 的误差累积是主要难点；短步反馈本身不一定立即失稳。 |
| 受控复杂度 benchmark | 生成并筛选 Simple/Complex 数据；完成 2 数据集 x 2 表示 x 2 框架 x 3 seed，共 24 组训练。 | 谱形复杂度和归一化都会影响完整递推；Keras 与 PyTorch 在 Complex 上存在需继续审查的差异。 |
| Moderate bridge set | Smoke 图册评审后，完整生成 1600 条 Moderate 候选；按统一 complexity score 取中间 1300 条。 | 可作为 Simple 与 Complex 之间的中间谱形池；12 组 Keras/PyTorch 训练和统一图包正在运行。 |
| RL inverse design | 完成 raw4 50 cm 目标退化诊断；完成 z=5 cm constrained-RL smoke 与 10 条 Luna 验证。 | 单纯 UV fraction 会奖励近乎能量耗尽的解；约束 UV power 路线避免了该退化，但仍只是单 seed 可行性验证。 |
| 工程环境 | 云端旧 Git 工作区恢复为 clean clone；完成 Julia/Python 环境验证。 | 本地 GitHub main 现在是唯一源码基线，云端只保留代码 checkout 和运行产物。 |

## 4. RNN：从原文复现到受控复杂度实验

### 4.1 已完成的问题拆分

此前 Luna early-dense RNN 出现“teacher-forced stepwise 很高，但 full
autoregressive 为负”的现象。本周将问题拆成两层：

1. 原文数据与原文式预处理是否能被我们的 RNN 递推；
2. 在统一的 Luna 几何、序列长度和波长维度下，仅改变谱形复杂度时，递推性能如何变化。

原文 `SC_spec_251` 上，PyTorch RNN 可获得约 `autoregressive R2=0.9633`，
说明 PyTorch 训练/评估主链路能够在原文类型任务上稳定递推。此前的 common-dBm
诊断也表明，Luna 数据在原文 `-55 dB` 映射后更稀疏，且最大相邻 z 变化发生在输入端，
更不利于局部窗口递推。

### 4.2 受控 benchmark 设计

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

### 4.3 完整 24 组训练结果

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

### 4.4 固定 1--10 step 诊断

固定 horizon 评估从真实历史窗口起步，只在随后的 `k` 步使用模型输出反馈。已完成的
Complex 结果中，PyTorch original-dBm 的 all-start R2 平均从 `0.9774`（1 step）到
`0.9752`（4 step）；Keras original-dBm 从 `0.9979` 到 `0.9824`。这说明在最初的
1--4 步范围内局部预测仍然很好。评估代码及新启动的 Moderate/Keras/PyTorch 训练已将
horizon 扩展到 `1--10`，用于检查误差是平滑累积还是在中短程出现明显拐点。

因此，full rollout 失败并不是“第 2 或第 3 步立刻崩溃”，而是小误差在 200 个保存点
上持续进入输入窗口后的长期累积。该诊断也解释了为什么 early-dense 1401 点任务比
200 点 controlled benchmark 更困难。

### 4.5 频域演化图材料

组会建议放三类图，而非只放 R2 表：

1. Simple/Complex 真实 target 对比：
   `rnn_visual_diagnostics/rnn_complexity_benchmark/visualizations/representative_simple_complex_targets.png`。
2. Simple 原版 dBm 的 target、stepwise、autoregressive 演化图和最终谱：
   `rnn_visual_diagnostics/rnn_complexity_benchmark/visualizations/pytorch_simple_original_dbm_seed123/`。
3. Moderate 最终 16 分位图册（本周已同步本地）：
   `rnn_visual_diagnostics/rnn_complexity_benchmark/moderate_visualizations/moderate_target_quantile_gallery.png`。

Complex 的 Keras/PyTorch 完整频域演化图与 Moderate 的对应图包已在后台按同一模板
批量导出；结果完成后必须同步到本地，再判断谱形层面的失败模式。

## 5. Moderate 数据集进展

Moderate smoke 图册确认其主流谱形不是 Simple 中几乎静止的 1030 nm 窄泵浦，且没有
Complex 的强碎裂主导行为。因此已完成独立的 1600 条候选生成，并将最终图册同步到本地。

已采用可复现的选择规则：按共享的 `complexity_score` 排序后，剔除最低和最高各 150
条，保留中间 1300 条，并划分为 1250/50 train/test。这样 Moderate 是“中间物理复杂度”
的受控实验，而非混合数据集。其 12 组 Keras/PyTorch 训练正在后台进行。

## 6. RL inverse design：从退化奖励到可验证闭环

### 6.1 已完成结果

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

### 6.2 当前定位

这证明了“surrogate -> RL candidate -> Luna validation”的闭环能跑通，但不构成
最优设计结论。下一版 reward 应基于完整 `S(z,lambda)`：奖励局域、连续、可追踪的
UV 色散波，同时惩罚过低总输出功率、过强电离和宽带噪声状短波结构。

可用于汇报的 Luna 演化图：
`rl_inverse_design/raw4_z5cm_constrained_rl_smoke_v1/luna_validation/analysis/luna_spectral_evolution_montage.png`。

### 6.3 RL inverse design 的整体路线图与当前层级

RL 在本项目中不是直接替代 Luna，也不是直接从目标谱反推出唯一参数。它是建立在
forward surrogate 之上的连续参数搜索策略。完整闭环应按以下层级推进：

```text
L0  定义可设计变量和物理边界
    E, tau, pressure, diameter；wall thickness=0.65 um、length=5 cm 固定
                         |
L1  训练并验收 forward surrogate
    参数 -> 可解释的谱指标或 S(z, lambda)
                         |
L2  设计不会退化的 reward
    有用 UV 输出 + 输出功率下限 + 电离/可运行性约束
                         |
L3  训练策略并产生候选
    SAC 与 random/Sobol、DE、梯度优化使用同一边界和近似预算
                         |
L4  Luna/UPPE 回代验证
    检查最终谱、频域演化图、能量传输、电子密度和失败率
                         |
L5  完整传播图 clean-UV 目标与主动学习
    将有效/失败候选反馈到数据集和 surrogate，迭代奖励与模型
                         |
L6  实验验证
    在实际 AR-HCF 平台验证候选工作点
```

各层的具体含义如下：

| 层级 | 要解决的问题 | 当前状态 | 通过标准 |
| --- | --- | --- | --- |
| L0 参数域 | agent 可以改什么，哪些量必须由原始变量导出 | 已完成 | 仅优化 `E, tau, p, d`；不直接优化 beta2、gamma、N 等派生特征 |
| L1 前向模型 | reward 是否建立在足够可靠的快速预测上 | 已完成 z=5 cm gate | final/UV log-power R2 高，UV-power 与 total-power 排序能力可用 |
| L2 reward | reward 是否真正代表“可用 UV”而非数值漏洞 | 已发现并修正第一类漏洞 | 禁止单独使用 UV fraction；必须同时关注绝对 UV 功率与输出功率下限 |
| L3 策略搜索 | policy 能否在连续参数空间提出非重复候选 | 已完成单 seed SAC smoke | 与随机、DE、梯度 baseline 使用同一参数边界和 budget 比较 |
| L4 Luna 验证 | surrogate 候选是否在真实 UPPE 中可运行且物理合理 | 已完成 5 cm 单 seed smoke | 不只看 surrogate reward，必须看 HDF5 演化图、传输、ne 和失败样本 |
| L5 temporal reward | 是否得到干净、局域、可追踪的 UV 色散波 | 尚未开始，是下一阶段核心 | reward 由完整 `S(z,lambda)` 特征计算，而不是只拟合最终谱 |
| L6 实验 | 仿真候选是否可在实验中复现 | 未开始 | 设备参数、可制造性和测量结果一致 |

**当前准确定位：我们已完成 L0--L2，并完成 L3--L4 的单 seed、固定 z=5 cm
可行性 smoke。** 这说明技术闭环已通，但还没有证明 RL 优于其它优化器，也没有证明
它找到的是“干净 UV 色散波”的最优参数。下一份可用于论文或开题的结论至少需要：

1. 用完整传播图 surrogate 或可从传播图提取的指标定义 clean-UV reward；
2. 在 SAC、Random/Sobol、Differential Evolution、可微梯度优化之间做等预算比较；
3. 对每种方法的 top-k 候选统一进行 Luna 回代，并同时展示频域演化图和最终谱；
4. 汇报成功率、UV 绝对功率、能量传输、最大电子密度、目标谱形质量及计算预算；
5. 将高价值与失败候选纳入下一轮训练数据，形成主动学习闭环。

### 6.4 本周可直接使用的 RL 汇报叙事

> 我们先把 RL 放在已经验证的 forward surrogate 之后，而不是让 agent 直接调用昂贵的
> Luna。第一个 50 cm 的 UV-fraction reward 暴露出一个重要问题：UV 占比高并不等于有用
> UV 输出，因为模型可以通过几乎耗尽总能量来“刷高”该指标。我们随后把任务固定在 UV
> 形成较早的 5 cm，并把 reward 改为绝对 UV 功率加输出功率约束。单 seed 的 SAC 候选已能
> 全部被 Luna 回代运行，说明闭环可行；下一步是把 reward 升级为完整传播图上的 clean-UV
> 色散波质量，并与传统优化方法进行公平比较。

## 7. 云端工程恢复

云端旧 Git checkout 停留在历史提交，并积累了未提交脚本和环境目录。本周完成以下恢复：

1. 备份旧 worktree 的 patch、源码文件清单和未跟踪源码归档；
2. 新建 GitHub clean clone 并确认 38 个旧未跟踪源码文件均与 GitHub 内容一致；
3. 旧目录保留为 legacy，新 clean clone 接管标准路径
   `/mnt/AI-AR-HCF-UPPE-Luna-Modeling`；
4. 修复共享 Julia depot 的单线程 FFTW wisdom，并重建 PyCall/PyPlot；
5. 验证 `using Luna`、`data_preprocessing.py --help` 与 `train_mlp.py --help`；
6. 规定所有汇报/评审用可视化必须同步到本地。

详细记录见 `docs/CLOUD_WORKTREE_RECOVERY_2026-09-21.md`。

## 8. 下周优先级

1. **完成 Moderate matrix 与统一图包**：监控 12 组 Moderate 训练，批量导出
   Complex/Moderate 的 target、stepwise、autoregressive、最终谱和逐步 R2 图，并同步本地。
2. **审查框架差异**：逐项比对 Keras 与 PyTorch 的 LSTM gate activation、初始化、
   batch 顺序、learning-rate schedule、best checkpoint 和 rollout 代码；确认是否是
   实现差异放大了 Complex 结果差距。
3. **形成复杂度梯度结论**：汇总 Simple--Moderate--Complex 的三 seed 指标和频域演化图，
   检查 Moderate score 范围是否确实处于两端之间。
4. **升级 RL reward**：从 final-spectrum proxy 迁移到 temporal clean-UV reward，
   并以 Luna 演化图、输出功率和电子密度共同验收，而不是仅优化 UV fraction。
5. **更新论文/开题叙事**：RNN 定位为局部传播与误差累积诊断 baseline；完整传播图的
   主路线仍是直接预测的 CNN/Transformer surrogate。

## 9. 建议 PPT 页序（7 页）

1. **项目总工作流与本周结论**：Luna 真值 -> 前向 surrogate -> RNN 诊断与 RL 候选 -> Luna 回代。
2. **为什么需要受控 benchmark**：原文 SC 可递推，Luna early-dense 易漂移；展示
   Simple/Complex 真实演化图。
3. **RNN 实验设计**：统一 0--5 cm、200 z、251 lambda；展示三类参数范围与复杂度逻辑。
4. **RNN 核心结果**：放 24 组聚合表，突出 Simple 与 Complex、dBm 与 min-max、Keras 与
   PyTorch 的差异。
5. **机制解释与 Moderate**：固定 1--10 step 曲线说明短步未崩、长 rollout 才累积；
   放 Moderate 分位图册，并说明下一步筛选。
6. **RL 分层闭环**：L0--L6 路线图；突出当前仅完成 5 cm single-seed L3--L4 smoke，
   不是最终 inverse-design 结论。
7. **下一步与工程保障**：Complexity matrix 完成、Keras/PyTorch 对齐、temporal clean-UV
   reward、等预算优化器对照、Luna 回代与可视化同步。

## 10. 汇报时的建议表述

> We have separated immediate local prediction from long-horizon recurrence.
> On controlled simple trajectories, both implementations reproduce the
> original RNN behavior. On complex Luna trajectories, one-step accuracy stays
> high but full rollout becomes representation- and implementation-sensitive.
> Therefore, the next task is not to claim that RNNs universally fail, but to
> visualize the failure mode, align the two implementations, and construct a
> controlled Moderate regime between the two extremes.
