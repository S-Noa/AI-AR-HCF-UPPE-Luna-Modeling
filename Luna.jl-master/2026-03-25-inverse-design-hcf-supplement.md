# Physics-Guided Inverse Design of Ultrafast Nonlinear Pulse Propagation in Gas-Filled Hollow-Core Fibers

## Supplementary Document - Implementation Guide

**Date:** 2026-05-10
**Version:** v1.0
**Status:** Implementation phase

***

## Appendix A: 项目实施关键步骤补充

### A.1 数据生成阶段（第2章补充）

**原文档描述：**

> 使用 Latin Hypercube Sampling 生成 \~10,000 个样本

**实际实施细节：**

1. **光纤长度固定化**：当前实现中，光纤长度根据模式自动固定：
   - 毛细管模式：`FIXED_FLENGTH = 1.0 m`
   - 反谐振模式：`FIXED_FLENGTH = 0.5 m`
   - 长度已从采样参数中移除
2. **管壁厚度处理**：反谐振模式下，管壁厚度为固定值（650nm/360nm/200nm/80nm），每种厚度生成独立数据集，训练独立模型（Scheme A架构）
3. **采样范围差异**：

| 参数       | 毛细管范围      | 反谐振范围      |
| -------- | ---------- | ---------- |
| Energy   | 1-100 uJ   | 0.3-3.0 uJ |
| Pressure | 0-20 bar   | 0.5-50 bar |
| Diameter | 100-300 um | 100-200 um |

1. **HDF5文件结构**：

```
sample_XXXXXX.h5
├── Ew/              # 复数电场 (freq, z)
├── z/               # 传播位置
├── stats/           # 统计信息
├── physics_features/ # 物理特征（beta2, gamma, N, L0等）
├── grid/            # 频率网格
└── prop_capillary_args/ # 输入参数
```

### A.2 数据预处理阶段（新增章节）

**原文档未包含此阶段，实际项目中为独立关键步骤：**

1. **光谱处理流程**：

```
Ew (复数电场)
  |
  |Ew|^2 (功率谱)
  |
波长裁剪 (200-2500nm)
  |
插值到500点均匀网格
  |
log10(功率 + epsilon)
  |
Min-Max归一化到[0,1]
```

1. **特征工程**：
   - 原始特征：energy, tau, pressure, diameter, wallthickness, beta2, gamma, N, L0, gamma\_K, P\_ratio, Aeff, neff（13维）
   - 多项式特征：energy\_pressure, diameter\_pressure（2维）
   - 总计：15维输入
2. **归一化策略（关键修正）**：
   - 输入特征：全局 Min-Max（基于训练集统计量）
   - 输出光谱：**Per-sample Min-Max**（每个样本独立归一化）
   - 原因：样本间光谱强度差异可达数个数量级，per-sample归一化使模型专注于学习相对形状

### A.3 模型架构变更（第3章补充）

**原文档架构（MLP Surrogate）：**

```
Dense(256) -> BN -> LeakyReLU -> Dense(512) -> BN -> LeakyReLU -> Dense(500)
```

**实际实施架构（三种模型）：**

| 模型           | 架构特点                                          | 参数量    | 适用场景        |
| ------------ | --------------------------------------------- | ------ | ----------- |
| StandardMLP  | 纯前馈，仅预测最终光谱                                   | \~400K | 快速基线        |
| TemporalMLP  | Encoder + TemporalDecoder(z条件) + FinalDecoder | \~600K | 时序演化预测      |
| TemporalLSTM | ParamEncoder + LSTM + SpectrumDecoder         | \~800K | **推荐，精度最高** |

**关键差异：**

- 输出层激活函数：原文档未指定，实际使用 **Sigmoid**（约束输出到\[0,1]）
- 损失函数：原文档 `L = L_spectral + alpha*L_energy + beta*L_RDW`，实际使用 `L = L_temporal + alpha*L_final + gamma*L_smoothness`
- energy\_loss 和 RDW\_loss 在实际中被移除（见A.4）

### A.4 损失函数修正说明

**原文档设计：**

```python
L = L_spectral + alpha*L_energy + beta*L_RDW
L_energy: ReLU(E_out_predicted - E_in)^2  # 能量守恒
L_RDW: MSE(detected_peak_lambda, lambda_RDW) * indicator(N>1)  # RDW位置约束
```

**实际实施及修正：**

1. **energy\_loss 移除原因**：
   - per-sample归一化后，光谱求和不再对应物理能量
   - `pred_energy = sum(pred_final)` in \[0, 500]，而 `input_energy` in \[0,1]（MinMax归一化后）
   - 尺度不匹配导致梯度方向错误
   - 能量信息已通过输入特征（energy, pressure）传递
2. **RDW\_loss 未实现原因**：
   - RDW峰值检测需要额外的峰值寻找算法
   - 在500点光谱上，RDW可能仅表现为1-2个点的局部极大值，检测不稳定
   - 增加训练复杂度，实际收益有限
3. **smoothness\_loss 新增**：

```python
L_smoothness = mean((pred_temporal[:, 1:, :] - pred_temporal[:, :-1, :])^2)
```

- 约束相邻z步之间的光谱变化平滑
- 防止时序预测出现突变

### A.5 训练超参数调整

| 参数            | 原文档建议             | 实际使用                                       | 调整原因             |
| ------------- | ----------------- | ------------------------------------------ | ---------------- |
| Batch size    | 256               | 128                                        | 内存限制，128更稳定      |
| Epochs        | 300-500           | 500                                        | 配合early stopping |
| Patience      | 未指定               | 20                                         | 防止过拟合            |
| Weight decay  | 未指定               | 1e-5                                       | L2正则化            |
| Dropout       | 0.1               | 0.1                                        | 保持一致             |
| Learning rate | 1e-3              | 1e-3                                       | 保持一致             |
| LR scheduler  | ReduceLROnPlateau | ReduceLROnPlateau(factor=0.5, patience=10) | 更激进的学习率衰减        |

<br />

***

## Appendix B: 实际操作注意事项

### B.1 环境配置

**Julia依赖：**

```julia
using Pkg
Pkg.add("Luna")        # 核心仿真包
Pkg.add("HDF5")        # 数据存储
Pkg.add("FFTW")        # FFT计算（注意wisdom缓存问题）
```

**Python依赖：**

```bash
pip install torch numpy h5py scikit-learn matplotlib joblib PyQt5
```

### B.2 常见问题及解决方案

| 问题                                          | 原因                     | 解决方案                                                  |
| ------------------------------------------- | ---------------------- | ----------------------------------------------------- |
| FFTW wisdom导入失败                             | 缓存文件损坏                 | 删除 `~/.julia/FFTWcache_*` 文件                          |
| HDF5 TypeBitfieldID错误                       | Julia特定类型不支持           | 使用 `safe_read_dataset()` 包装                           |
| 模型加载size mismatch                           | 输入维度不一致                | 从state\_dict推断维度，动态创建模型                               |
| R^2为负值                                      | 数据归一化不一致               | 确保train/val/test使用相同归一化                               |
| 训练loss下降但R^2不升                              | energy\_loss干扰         | 移除energy\_loss，调整alpha权重                              |
| 云端运行无日志输出                                   | stdout缓冲               | 添加 `flush(stdout)` after each print                   |
| evaluate\_models\_heatmap.py 模型加载失败         | 评估脚本与训练脚本架构不一致         | 确保evaluate\_models\_heatmap.py中的模型类与train\_mlp.py完全一致 |
| evaluate\_models\_heatmap.py 输入维度错误         | FEATURE\_NAMES与训练时不一致  | 确保包含wallthickness和多项式特征                               |
| evaluate\_models\_heatmap.py z\_positions越界 | z\_positions与光谱数据长度不一致 | 分别计算光谱和z\_positions的下采样索引                             |

### B.3 数据一致性检查清单

在运行训练前，验证以下数据一致性：

```python
# 检查1：输入维度匹配
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 15

# 检查2：输出范围
assert y_temporal_train.min() >= 0 and y_temporal_train.max() <= 1
assert y_temporal_val.min() >= 0 and y_temporal_val.max() <= 1

# 检查3：y_final与y_temporal一致性
for i in range(len(y_train)):
    assert np.allclose(y_train[i], y_temporal_train[i, -1, :])

# 检查4：无NaN/Inf
assert np.isfinite(X_train).all()
assert np.isfinite(y_temporal_train).all()
```

***

## Appendix C: 与原文档的差异总结

| 方面    | 原文档                      | 实际实施                         | 差异原因                    |
| ----- | ------------------------ | ---------------------------- | ----------------------- |
| 光纤长度  | 0.03-0.5m可变              | 1.0m(毛细管)/0.5m(反谐振)固定        | 简化参数空间                  |
| 管壁厚度  | 未提及                      | 4种固定厚度(650/360/200/80nm)     | 独立模型架构                  |
| 输入特征  | 15维(7+8)                 | 15维(13+2)                    | 移除chirp，新增wallthickness |
| 输出归一化 | 全局归一化                    | Per-sample归一化                | 样本间强度差异大                |
| 损失函数  | 含energy\_loss, RDW\_loss | 仅temporal+final+smoothness   | 实现简化                    |
| 模型架构  | 单一MLP                    | 三种架构(MLP/TemporalMLP/LSTM)   | 对比实验                    |
| 逆向设计  | Tandem网络                 | 仅梯度优化                        | 开发优先级                   |
| 激活函数  | 未指定                      | Sigmoid输出                    | 匹配归一化范围                 |
| 评估脚本  | 未提及                      | evaluate\_models\_heatmap.py | 模型对比可视化                 |

***

## Appendix D: 推荐实施路径

基于当前项目状态，推荐以下实施优先级：

**Phase 1（已完成）：**

- [x] 数据生成（Julia UPPE）
- [x] 数据预处理（Python）
- [x] 前向模型训练（三种架构）
- [x] GUI预测工具

**Phase 2（进行中）：**

- [ ] 模型性能优化（修复TemporalMLP的final R^2问题）
- [ ] 多厚度模型整合
- [ ] 云端部署脚本

**Phase 3（未来工作）：**

- [ ] Tandem逆向网络训练
- [ ] 压力梯度扩展（Direction B+）
- [ ] 实验验证（Phase C）

***

## Appendix E: evaluate\_models\_heatmap.py 输出图像说明

### E.1 输出文件列表

| 文件名                             | 尺寸           | DPI | 内容描述        |
| ------------------------------- | ------------ | --- | ----------- |
| model\_comparison\_heatmap.png  | 18x14 inches | 300 | 综合对比图（3行3列） |
| model\_comparison\_spectrum.png | 14x6 inches  | 300 | 最终光谱对比图     |
| model\_evaluation\_report.txt   | 文本           | N/A | 详细评估指标报告    |

### E.2 model\_comparison\_heatmap.png 详细内容

**第1行：时序演化热力图（3列）**

- 第1列：TemporalLSTM 预测的时序演化
- 第2列：TemporalMLP 预测的时序演化
- 第3列：Ground Truth（HDF5原始数据）
- X轴：波长（200-2500 nm）
- Y轴：传播距离 Z（m）
- 色图：'hot'（功率从低到高：黑->红->黄->白）

**第2行：差异热力图（3列）**

- 第1列：LSTM预测 - Ground Truth
- 第2列：TemporalMLP预测 - Ground Truth
- 第3列：LSTM预测 - TemporalMLP预测
- 色图：'RdBu\_r'（红色=正值，蓝色=负值，白色=零）

**第3行：最终光谱对比（1列，横跨3列）**

- 黑色虚线：Ground Truth 最终光谱
- 蓝色实线：TemporalLSTM 预测
- 橙色实线：TemporalMLP 预测
- X轴：波长（nm）
- Y轴：归一化对数功率

**附加信息：**

- 底部：模型性能指标（MSE, R^2）
- 左上角：样本参数（energy, tau, pressure, diameter）

### E.3 model\_comparison\_spectrum.png 详细内容

- 单一子图，仅展示最终光谱对比
- 与heatmap第3行内容相同，但更大更清晰
- 适合单独查看和论文插图

### E.4 Ground Truth 生成逻辑

**数据来源：** HDF5文件中的 `Ew` 数据集

**处理流程：**

```python
# 1. 读取复数电场 Ew
E_omega = f['Ew'][()]  # shape: (z_steps, freq_points)

# 2. 计算功率谱
power = abs(E_omega)^2

# 3. 转置为 (z, frequency)
if power.shape[0] < power.shape[1]:
    power = power.T

# 4. 对数变换
log_power = log10(power + 1e-12)

# 5. 全局归一化（注意：这里是全局而非per-sample）
p_min = log_power.min()
p_max = log_power.max()
log_power_norm = (log_power - p_min) / (p_max - p_min)

# 6. 插值到500波长点
if log_power_norm.shape[1] != 500:
    interp = interp1d(old_x, log_power_norm, axis=1, kind='linear')
    log_power_norm = interp(new_x)

# 7. 下采样到20个z步
if log_power_norm.shape[0] > 20:
    indices = linspace(0, n_z - 1, 20, dtype=int)
    temporal_evolution = log_power_norm[indices, :]
```

### E.5 Ground Truth 生成逻辑的问题分析

**问题1：全局归一化 vs Per-sample归一化**

| 方面    | Ground Truth（evaluate\_models\_heatmap.py）        | 模型训练（data\_preprocessing.py）                      |
| ----- | ------------------------------------------------- | ------------------------------------------------- |
| 归一化方式 | 全局 Min-Max                                        | Per-sample Min-Max                                |
| 计算公式  | `(data - global_min) / (global_max - global_min)` | `(data - sample_min) / (sample_max - sample_min)` |
| 影响    | 保留样本间绝对强度差异                                       | 每个样本独立归一化到\[0,1]                                  |

**后果：**

- Ground Truth 热力图显示的是全局归一化后的值
- 模型预测输出的是per-sample归一化后的值
- 两者在绝对数值上不可直接比较
- 但相对形状（峰值位置、光谱结构）仍可对比

**建议：** 如果需要进行严格的数值比较，应对Ground Truth也进行per-sample归一化，或在使用时进行反归一化。

**问题2：z\_positions下采样索引不一致（已修复）**

**原问题：**

```python
# 错误代码
indices = np.linspace(0, log_power_norm.shape[0] - 1, 20, dtype=int)
temporal_evolution = log_power_norm[indices, :]
z_positions = z_positions[indices]  # 错误：用光谱的索引访问z_positions
```

**修复后：**

```python
# 正确代码
n_z = log_power_norm.shape[0]
indices = np.linspace(0, n_z - 1, 20, dtype=int)
temporal_evolution = log_power_norm[indices, :]
if z_positions is not None and len(z_positions) > 20:
    z_indices = np.linspace(0, len(z_positions) - 1, 20, dtype=int)
    z_positions = z_positions[z_indices]  # 正确：使用z_positions自己的索引
```

**问题3：波长轴不匹配**

- HDF5中的波长轴来自 `grid/λ`，可能与训练时的500点均匀网格不同
- 模型预测始终使用200-2500 nm的500点均匀网格
- 如果HDF5中的波长范围不同，可视化时可能出现偏移

**建议：** 在可视化时明确标注波长轴的来源，或在预处理阶段统一波长网格。

***

*本文档为原始设计文档的补充说明，所有修改和补充均基于实际项目实施经验。如有冲突，以实际代码实现为准。*
