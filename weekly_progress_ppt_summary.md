# 过去一周项目进展汇报提纲

## 1. 本周主线

本周重点围绕 Luna RNN baseline 的 autoregressive 失败问题做归因分析。核心问题是：原文 RNN 在自己的数据上 autoregressive 表现很好，而我们在 Luna AR-HCF early-dense 数据上出现了“stepwise R2 很高、autoregressive R2 明显下降甚至为负”的现象。因此本周工作从单纯调参转向交叉验证：区分代码迁移问题、数据预处理问题和 Luna 任务本身复杂度问题。

## 2. Early-dense 数据整理

- 云端 `t0p6 / 0.65 um` early-dense 数据已可用。
- 原始目录中有两种 z 网格，不能直接混用：
  - `7780` 个样本：`z_len=1401`，前 `10 cm` 高密度采样；
  - `12` 个样本：旧格式，`z_len=696`，前 `0.5 cm` 高密度采样。
- 已建立筛选后的 symlink 目录，只使用一致的 `z_len=1401` 数据。
- 已完成 processed 数据：
  - `Train / Val / Test = 5446 / 1167 / 1167`
  - temporal target shape: `(N, 1401, 1000)`

PPT 建议图：放一张数据流示意图：raw early-dense HDF5 -> z-grid audit -> filtered z1401 subset -> processed temporal dataset。

## 3. Luna RNN Baseline 结果

RNN 采用窗口式 one-step 预测：

```text
S(z-window : z) -> S(z+1)
```

评估分成两种：

- Stepwise / teacher forcing：每一步输入真实历史谱；
- Autoregressive rollout：从初始谱开始，将模型预测反复喂回模型。

主要结果：

| 任务 | 波长点 | z 点 | Stepwise R2 | Autoregressive R2 | 结论 |
|---|---:|---:|---:|---:|---|
| full z1401 | 1000 | 1401 | 很高，约 0.99 | 负值 | 完整长程递推失败 |
| z10cm_201 | 1000 | 201 | 0.9671 | 0.2867 | 缩短递推后改善，但仍不强 |
| z10cm_101 | 1000 | 101 | 0.9446 | 0.4174 | 当前较稳定短程 baseline |
| z10cm_51 | 1000 | 51 | 0.9126 | 0.4296 | 当前最好短程 baseline |
| z10cm_101_lambda251 | 251 | 101 | 0.8543 | 0.4446 | 降维有帮助，但不根本 |

结论：RNN 可以作为 short-horizon/local propagation baseline，但目前不能作为完整 high-density propagation map 的主模型。

PPT 建议图：一页表格 + 一张 stepwise vs autoregressive 示意图，突出“one-step high does not imply stable rollout”。

## 4. 为什么 Autoregressive 会失败

主要原因是训练和推理分布不一致：

- 训练时输入窗口几乎总是真实谱；
- 推理时输入窗口逐步变成模型自己的预测谱；
- 小误差会被反复喂回并累积；
- Luna early-dense full task 需要约 `1391` 次递推，远长于原文很多任务；
- AR-HCF / UPPE / plasma / UV generation 的谱演化比标准 NLSE/GNLSE 数据更复杂。

因此 autoregressive R2 为负不是单步预测能力差，而是模型递推轨迹离开真实传播流形。

## 5. 与原文 RNN 的交叉验证

为了判断问题来自代码还是数据，本周建立了 attribution matrix：

```text
原文代码 + 原文数据
我们的 PyTorch 代码 + 原文数据
原文代码 + Luna 低维数据
我们的 PyTorch 代码 + Luna 低维数据
```

目前已完成的是“原文 Keras 代码 + Luna 低维数据”：

| Luna 低维数据 | Epoch | Stepwise R2 | Autoregressive R2 |
|---|---:|---:|---:|
| per_sample_minmax | 1 | 0.9600 | 0.3082 |
| per_sample_minmax | 3 | 0.9743 | 0.3161 |
| per_sample_minmax | 5 | 0.9764 | 0.2809 |
| rnn_paper_db | 1 | 0.9656 | -0.1442 |
| rnn_paper_db | 3 | 0.9780 | 0.0947 |
| rnn_paper_db | 5 | 0.9837 | -0.0537 |

判断：即使用原文 Keras 代码，Luna 低维数据上的 autoregressive 仍然不强。这说明失败不太可能只是 PyTorch 迁移代码错误，更可能与 Luna 数据的传播复杂性、归一化状态空间和递推长度有关。

## 6. 预处理是否仍可能有影响

有影响，但目前看不是唯一原因。

原文 `load_data.py` 的 `dBm` 处理逻辑是：

```text
raw positive spectrum
-> divide by global max
-> 10 log10
-> clip below -55 dB
-> map to [0, 1]
```

我们之前的 `rnn_paper_db` 已经模拟了这一路线，但它是在 Luna preprocessing 阶段完成的，不是把 raw Luna power 交给原文 `load_data.py` 处理。

因此今天新增了一个更严格的验证：

- 直接从 Luna HDF5 读取原始复频域场 `Eω`；
- 转成线性 power map；
- 导出为原文格式 `.mat`；
- 再让原文 `load_data(..., normalization='dBm')` 自己处理。

该任务已在云端后台运行：

```text
/mnt/Luna.jl-master/rnn_original_code_env/run_raw_power_dBm_luna_lambda251.sh
```

如果该实验仍只得到约 `0.3` 的 autoregressive R2，则可以基本确认：预处理差异不是主因。

## 7. 原文数据复现实验状态

原文 Zenodo 数据 `RNNnonlinear_v2.zip` 约 `6.2 GB`。之前下载在 `123 MB` 处中断，而且旧脚本只检查文件是否存在，导致无法自动续传。

今天已修复为：

- 检查 zip 完整性；
- 不完整则使用 `curl -C -` 断点续传；
- 外层循环自动重试；
- 下载完成后自动解压、转换并用我们的 PyTorch RNN 训练原文数据。

后台脚本：

```text
/mnt/Luna.jl-master/rnn_original_data/run_original_data_attribution_resumable.sh
```

PPT 表述建议：原文数据复现仍在进行，当前结论先基于“原文代码 + Luna 数据”和“我们的代码 + Luna 数据”的交叉结果。

## 8. 当前阶段结论

可以在组会上这样总结：

1. Early-dense 数据已经完成清洗和 RNN 数据构建，但混合 z 网格需要筛选。
2. Luna RNN 的 stepwise R2 很高，说明局部 one-step 学习没有问题。
3. Autoregressive 失败主要来自递推误差累积；完整 z1401 任务尤其困难。
4. 即使用原文 Keras 代码训练 Luna 低维数据，autoregressive 仍然较弱，说明问题更偏向数据/任务特性，而不是单纯代码迁移错误。
5. 目前可用 RNN baseline 应定位为 short-horizon baseline；完整传播图预测仍应以 Transformer/CNN direct full-map model 为主。

## 9. 下一步计划

- 等待 raw-power + original dBm 验证完成，最终判断预处理差异是否是主要瓶颈。
- 等待原文 Zenodo 数据下载完成，跑：
  - 原文代码 + 原文数据；
  - 我们的 PyTorch 代码 + 原文数据。
- 如果我们的代码在原文数据上成功，而 Luna 低维仍弱，则正式把结论写成：

```text
The RNN baseline is valid for local one-step prediction, but Luna AR-HCF propagation is too complex for stable long autoregressive rollout under the current setup.
```

- 后续主线回到 Transformer/CNN：
  - 用 early-dense 数据改善早期传播细节；
  - 做真实强度/global-log 版本；
  - 基于 forward surrogate 推进 inverse design / reinforcement learning。

## 10. 建议 PPT 页序

1. 本周问题：为什么 RNN stepwise 高但 autoregressive 失败？
2. Early-dense 数据状态：样本数、z 网格筛选、processed 数据。
3. Luna RNN baseline 结果表：full vs z10cm vs lambda251。
4. Autoregressive 失败机制：teacher forcing 与 rollout mismatch。
5. 原文 RNN 交叉验证设计：4 组 attribution matrix。
6. 已完成结果：原文 Keras 代码 + Luna 低维数据。
7. 预处理归因：raw-power + original dBm 正在运行。
8. 当前结论与下一步。
