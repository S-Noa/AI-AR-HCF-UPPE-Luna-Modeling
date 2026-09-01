# Transformer Temporal v3 与 Luna RNN 结果评估摘要

## Transformer temporal v3

结果目录：`Luna.jl-master/examples/simple_interface/models_t650_transformer_temporal_v3`

关键指标：

| 指标 | 数值 |
|---|---:|
| Final spectrum R2 | 0.9635 |
| Final UV R2 | 0.8916 |
| Temporal evolution R2 | 0.9650 |
| Temporal UV R2 | 0.8927 |
| Early-Z R2 | 0.9649 |
| Early UV R2 | 0.8942 |
| Restored epoch | 220 |
| Selection metric | `val_r2` |
| Training time | 225316.8 s |

判断：

- 模型已经能较好复现完整传播演化图的主结构，`Temporal_R2` 略高于最终光谱 `R2`。
- 紫外波段仍是主要短板，`UV_R2` 与 `Temporal_UV_R2` 均约为 0.89。
- 从 `temporal_evolution.png` 看，模型能捕捉主峰、宽带展宽和大尺度传播趋势，但对短波尖峰、细密条纹和局部干涉结构存在过平滑。
- 当前 v3 是按 `val_r2` 选择 checkpoint，不是按 `val_temporal_r2`。如果论文主线强调传播演化预测，后续训练/微调应优先使用 `--selection-metric val_temporal_r2`。

## Luna RNN baseline

结果目录：`Luna.jl-master/examples/simple_interface/results_luna_rnn_t650`

关键指标：

| 指标 | 数值 |
|---|---:|
| Stepwise MSE | 0.000481 |
| Stepwise R2 | 0.9944 |
| Autoregressive MSE | 0.1201 |
| Autoregressive R2 | -0.4099 |
| Window size | 10 |
| Hidden size | 250 |
| Training time | 264823.8 s |

判断：

- RNN 的 teacher-forced one-step prediction 很强，说明它能用真实历史窗口预测下一步光谱。
- Autoregressive rollout 失败，说明预测误差在递推过程中快速累积。
- 这个对照适合用于论文讨论：一步预测精度高并不等价于稳定的长程传播图预测。
- Transformer 直接从物理参数预测完整 `z x wavelength` 演化图，避免了 RNN 的逐步递推漂移问题。

## 推荐汇报口径

- Transformer 是本文主模型，重点展示完整传播演化图预测。
- RNN 是对照模型，用于说明递推式序列模型在长程传播预测中可能出现误差累积。
- 当前 Transformer 的主要改进方向不是单纯提高最终光谱 R2，而是减少传播图过平滑、提高 UV 尖峰和早期传播细节的保真度。
