# 模型架构对比分析报告

## 时序MLP、LSTM与指导文档标准规范的差异对比

**分析日期:** 2026-04-29
**分析对象:** train_mlp.py 中的三种模型架构（StandardMLP / TemporalMLP / TemporalLSTM）
**参考标准:** 2026-03-25-inverse-design-hcf.md（项目设计文档）

---

## 一、文件状态核查

### 1.1 data_generation.jl 管壁厚度命令行参数功能

| 要求项 | 状态 | 说明 |
|--------|------|------|
| 命令行参数解析 | ✅ 已实现 | `--wall-thickness` / `-t` 参数 |
| 参数类型正确 | ✅ 已实现 | `parse(Float64, ARGS[i+1])` |
| 合理默认值 | ✅ 已实现 | 默认 `0.5` μm |
| 参数验证机制 | ✅ 已实现 | `validate_wall_thickness()` 函数，检查 >0，警告 <0.1 或 >5.0 |
| 替代硬编码值 | ✅ 已实现 | 使用 `WALL_THICKNESS_UM` 常量替代硬编码 |

**结论:** `data_generation.jl` v3.2 已完整实现所有要求功能。

### 1.2 train_mlp.py 架构选择命令行参数功能

| 要求项 | 状态 | 说明 |
|--------|------|------|
| 架构选择参数解析 | ✅ 已实现 | `--model` / `-m` 参数，choices=['mlp', 'temporal', 'lstm'] |
| 参数与架构映射 | ✅ 已实现 | `MODEL_TYPE` 映射到对应类 |
| 架构切换功能 | ✅ 已实现 | `if MODEL_TYPE == "lstm"` 等条件分支 |

**结论:** `train_mlp.py` 已完整实现所有要求功能。

---

## 二、损失函数定义对比分析

### 2.1 指导文档标准规范（Section 3.5）

```
L = L_spectral + α·L_energy + β·L_RDW

L_spectral:  MSE(predicted_log_spectrum, true_log_spectrum)
L_energy:    ReLU(E_out_predicted - E_in)²
L_RDW:       MSE(detected_peak_λ, λ_RDW_analytical) · 𝟙(N > 1)
```

**标准规范特征:**
- 三项组成：光谱MSE + 能量守恒 + RDW相位匹配
- `L_energy` 使用 `ReLU(E_out - E_in)²`，仅惩罚输出能量超过输入能量的情况
- `L_RDW` 仅在 `N > 1` 时激活，约束RDW峰位
- 建议超参数：`α = 0.1`, `β = 0.01`

### 2.2 当前实现（TemporalLoss类）

```python
L = L_temporal + α·L_final + β·L_energy + γ·L_smoothness

L_temporal:   MSE(pred_temporal, target_temporal)  [时序光谱损失]
L_final:      MSE(pred_final, target_final)         [最终时刻损失]
L_energy:     torch.relu(pred_energy - input_energy)²  [能量守恒]
L_smoothness: mean((pred_temporal[:,1:,:] - pred_temporal[:,:-1,:])²)  [平滑性]
```

**当前实现特征:**
- 四项组成：时序MSE + 最终时刻MSE + 能量守恒 + 时序平滑性
- 缺少 **RDW相位匹配损失** (`L_RDW`)
- 能量守恒实现与标准一致（使用 `ReLU`）
- 新增 **时序平滑性损失** (`L_smoothness`)，标准文档未提及
- 默认超参数：`α = 0.5`, `β = 0.1`, `γ = 0.01`（与建议值不同）

### 2.3 差异总结

| 对比维度 | 指导文档标准 | 当前实现 | 差异评估 |
|----------|-------------|----------|----------|
| 损失项数量 | 3项 | 4项 | 新增平滑性项 |
| RDW损失 | ✅ 有 `L_RDW` | ❌ 缺失 | **重要缺失** |
| 能量损失 | `ReLU(E_out - E_in)²` | `torch.relu(pred_energy - input_energy)²` | 一致 ✅ |
| 光谱损失 | `L_spectral` (单时刻) | `L_temporal` + `L_final` (多时序) | 因时序预测而扩展 |
| 平滑性损失 | 无 | 有 `L_smoothness` | 新增，合理 |
| α默认值 | 0.1 | 0.5 | **偏差较大** |
| β默认值 | 0.01 | 0.1 | **偏差较大** |
| γ默认值 | 无 | 0.01 | 新增参数 |

**关键问题:**
1. **RDW相位匹配损失缺失**: 标准文档将 `L_RDW` 作为重要物理约束，当前实现完全缺失此项
2. **超参数设置偏差**: α和β的默认值与文档建议相差5-10倍，可能影响训练稳定性
3. **能量计算方式**: 当前实现使用 `torch.sum(pred_final, dim=1)` 估算能量，而文档建议从光谱积分计算

---

## 三、网络架构设计对比分析

### 3.1 指导文档标准规范（Section 3.4）

```
Input: 15 features (7 raw + 8 physics)
  │
  ├── Normalize all inputs to [0, 1] via min-max scaling
  │
  Dense(256)  → BatchNorm → LeakyReLU(0.01)
  Dense(512)  → BatchNorm → LeakyReLU(0.01)
  Dense(512)  → BatchNorm → LeakyReLU(0.01) → Dropout(0.1)
  Dense(256)  → BatchNorm → LeakyReLU(0.01)
  Dense(500)  → Output (log-power spectrum)
```

**标准规范特征:**
- 输入维度：15（7原始参数 + 8物理特征）
- 层结构：256 → 512 → 512 → 256 → 500
- 激活函数：LeakyReLU(0.01)
- 正则化：BatchNorm + Dropout(0.1)
- 输出：500点log功率谱，**无Sigmoid激活**
- 总参数量：~800K

### 3.2 当前实现对比

#### 3.2.1 StandardMLP（标准MLP）

```python
nn.Linear(input_dim, 256)   → BatchNorm1d(256) → LeakyReLU(0.01)
nn.Linear(256, 512)         → BatchNorm1d(512) → LeakyReLU(0.01)
nn.Linear(512, 512)         → BatchNorm1d(512) → LeakyReLU(0.01) → Dropout(0.1)
nn.Linear(512, 256)         → BatchNorm1d(256) → LeakyReLU(0.01)
nn.Linear(256, output_dim)  → Output
```

**与标准对比:**
- ✅ 层结构完全一致（256→512→512→256→output）
- ✅ 激活函数一致（LeakyReLU 0.01）
- ✅ BatchNorm和Dropout位置一致
- ✅ 输出无Sigmoid（符合标准）

**结论:** StandardMLP 与指导文档标准规范 **完全一致**。

#### 3.2.2 TemporalMLP（时序MLP）

```python
# 编码器
nn.Linear(input_dim, 256)   → BatchNorm1d(256) → LeakyReLU(0.01)
nn.Linear(256, 512)         → BatchNorm1d(512) → LeakyReLU(0.01)
nn.Linear(512, 512)         → BatchNorm1d(512) → LeakyReLU(0.01) → Dropout(0.1)

# 时序解码器（每个z位置）
nn.Linear(512 + 1, 512)     → BatchNorm1d(512) → LeakyReLU(0.01)
nn.Linear(512, 512)         → BatchNorm1d(512) → LeakyReLU(0.01)
nn.Linear(512, output_dim)  → Sigmoid()

# 最终时刻解码器
nn.Linear(512, 256)         → BatchNorm1d(256) → LeakyReLU(0.01)
nn.Linear(256, output_dim)  → Sigmoid()
```

**与标准对比:**
- ⚠️ 编码器结构与标准MLP一致 ✅
- ❌ **输出层使用 Sigmoid 激活**，标准文档明确使用无激活的线性输出
- ⚠️ 架构为编码器-解码器结构，标准文档未明确时序扩展方式
- ⚠️ 时序解码器输入为 `512 + 1`（+1为z位置编码），属于合理扩展

**关键问题:**
1. **Sigmoid输出限制**: Sigmoid将输出限制在(0,1)，而标准文档使用线性输出预测log功率谱（可正可负）
2. **架构扩展未在文档中定义**: 时序MLP是文档中"Neural Split-Step"（Section 8A）的简化版，但实现方式不同

#### 3.2.3 TemporalLSTM（时序LSTM）

```python
# 参数编码器
nn.Linear(input_dim, 256)   → BatchNorm1d(256) → LeakyReLU(0.01)
nn.Linear(256, hidden_dim)  → BatchNorm1d(hidden_dim) → LeakyReLU(0.01)

# LSTM层
nn.LSTM(input_size=hidden_dim + 1, hidden_size=hidden_dim, 
        num_layers=2, batch_first=True, dropout=0.1)

# 光谱解码器
nn.Linear(hidden_dim, 512)  → BatchNorm1d(512) → LeakyReLU(0.01)
nn.Linear(512, 256)         → BatchNorm1d(256) → LeakyReLU(0.01)
nn.Linear(256, output_dim)  → Sigmoid()
```

**与标准对比:**
- ❌ **输出层使用 Sigmoid 激活**，与标准不一致
- ❌ **LSTM架构在指导文档中未提及**，属于额外扩展
- ⚠️ 文档Section 8A提出"Neural Split-Step"架构，但使用CNN/MLP处理非线性层，非LSTM
- ⚠️ LSTM的 `num_layers=2` 和 `dropout=0.1` 配置合理但无文档依据

**关键问题:**
1. **LSTM未在标准文档中定义**: 指导文档仅定义了标准MLP和Neural Split-Step两种架构，LSTM是额外添加的
2. **Sigmoid输出问题**: 与TemporalMLP相同，限制了输出范围
3. **与Neural Split-Step的差异**: 文档8A节建议使用物理嵌入的分步架构（精确色散层+学习非线性层），而LSTM是纯数据驱动的黑盒时序模型

### 3.3 架构差异总结

| 对比维度 | 指导文档标准 | StandardMLP | TemporalMLP | TemporalLSTM |
|----------|-------------|-------------|-------------|--------------|
| 层结构 | 256→512→512→256→500 | ✅ 一致 | ⚠️ 编码器一致 | ⚠️ 不同 |
| 激活函数 | LeakyReLU(0.01) | ✅ 一致 | ✅ 一致 | ✅ 一致 |
| 输出激活 | 无（线性） | ✅ 线性 | ❌ Sigmoid | ❌ Sigmoid |
| BatchNorm | 有 | ✅ 有 | ✅ 有 | ✅ 有 |
| Dropout | 0.1 | ✅ 0.1 | ✅ 0.1 | ✅ 0.1 |
| 时序处理 | 无（单输出） | N/A | 编码器-解码器 | LSTM |
| 物理嵌入 | 输入特征+损失 | 仅输入特征 | 仅输入特征 | 仅输入特征 |
| 文档依据 | Section 3.4 | ✅ 完全匹配 | ⚠️ 部分匹配 | ❌ 无直接依据 |

---

## 四、模型性能表现预期对比

### 4.1 指导文档性能目标（Section 3.6 / 5.1）

| 指标 | 目标值 |
|------|--------|
| 测试集 R² | > 0.95 |
| 分区域 R²（低N/高N/电离区） | > 0.90 |
| RDW峰位误差 | < 10 nm |
| RDW峰幅相对误差 | < 20% |
| 能量守恒误差 | \|E_out/E_in - true_ratio\| < 5% |

### 4.2 各架构性能预期分析

#### StandardMLP
- **优势**: 与文档标准完全一致，结构简单，训练稳定
- **劣势**: 仅预测最终光谱，无法捕捉演化过程
- **预期R²**: 0.93-0.97（取决于数据质量和训练充分度）
- **符合度**: ⭐⭐⭐⭐⭐（完全符合标准）

#### TemporalMLP
- **优势**: 可预测全过程光谱演化，编码器-解码器结构直观
- **劣势**: 
  - Sigmoid输出限制动态范围，可能影响RDW等弱信号预测精度
  - 缺少RDW损失约束，RDW峰位误差可能 > 10 nm
  - 时序解码器独立处理每个z位置，未显式建模时序依赖性
- **预期R²**: 0.90-0.95（最终时刻），时序一致性较好
- **符合度**: ⭐⭐⭐（部分符合，有显著偏差）

#### TemporalLSTM
- **优势**: LSTM天然建模时序依赖性，可能更好捕捉脉冲演化动态
- **劣势**:
  - Sigmoid输出限制动态范围
  - 缺少RDW损失约束
  - 黑盒时序模型，缺乏物理可解释性（与文档8A的Neural Split-Step理念相悖）
  - LSTM训练难度更高，可能收敛较慢
- **预期R²**: 0.91-0.96（最终时刻），时序平滑性较好
- **符合度**: ⭐⭐（偏差较大，无文档依据）

### 4.3 关键性能风险

1. **Sigmoid输出导致的动态范围损失**: 
   - 文档标准使用线性输出预测log功率谱（范围约 [-10, 0]）
   - Sigmoid限制输出到 (0,1)，无法表示负值
   - 影响：弱光谱成分（如RDW）的预测精度可能显著下降

2. **RDW损失缺失**:
   - 文档将RDW约束作为关键物理损失
   - 缺失导致模型不学习RDW相位匹配条件
   - 影响：RDW峰位误差可能远超 10 nm 目标

3. **超参数偏差**:
   - α=0.5（文档建议0.1）：过度强调最终时刻损失，可能忽略时序一致性
   - β=0.1（文档建议0.01）：能量守恒权重过高，可能主导训练

---

## 五、与Neural Split-Step架构的对比（Section 8A）

指导文档8A节提出了一种更先进的"Neural Split-Step"架构，与当前时序实现对比：

| 特征 | Neural Split-Step（文档8A） | TemporalMLP/TemporalLSTM（当前） |
|------|---------------------------|--------------------------------|
| 物理嵌入 | ✅ 精确色散层（零可学习参数） | ❌ 无物理嵌入 |
| 非线性处理 | 小型MLP学习Kerr+电离效应 | 大型网络学习全部映射 |
| 可解释性 | 每层对应物理传播步 | 黑盒时序模型 |
| FFT/IFFT | 使用可微分傅里叶变换 | 不使用 |
| 中间输出 | 每步光谱（科学价值） | 每步光谱 |
| 新颖性 | 高（Tier 2期刊） | 低-中（标准时序扩展） |
| 实现复杂度 | 中 | 低-中 |

**结论:** 当前时序实现是标准MLP的时序扩展，未达到文档8A推荐的物理嵌入架构水平。

---

## 六、改进建议

### 6.1 高优先级改进

1. **添加RDW相位匹配损失**
   ```python
   # 在TemporalLoss中添加
   def rdw_loss(self, pred_final, target_final, features):
       # 从features中提取N和理论λ_RDW
       N = features[:, 3]  # soliton order
       lambda_rdw = features[:, 5]  # predicted RDW wavelength
       
       # 检测预测光谱中的RDW峰
       pred_peak_lambda = detect_rdw_peak(pred_final)
       target_peak_lambda = detect_rdw_peak(target_final)
       
       # 仅在N > 1时激活
       mask = (N > 1.0).float()
       loss = torch.mean((pred_peak_lambda - lambda_rdw)² * mask)
       return loss
   ```

2. **移除Sigmoid输出激活（时序模型）**
   ```python
   # TemporalMLP.temporal_decoder 最后一层
   nn.Linear(512, output_dim)  # 移除 nn.Sigmoid()
   
   # TemporalMLP.final_decoder 最后一层
   nn.Linear(256, output_dim)  # 移除 nn.Sigmoid()
   
   # TemporalLSTM.spectrum_decoder 最后一层
   nn.Linear(256, output_dim)  # 移除 nn.Sigmoid()
   ```

3. **调整默认超参数**
   ```python
   # 与文档建议一致
   parser.add_argument('--alpha', type=float, default=0.1)
   parser.add_argument('--beta', type=float, default=0.01)
   ```

### 6.2 中优先级改进

4. **能量计算方式优化**
   - 当前：`torch.sum(pred_final, dim=1)`
   - 建议：从光谱积分计算实际能量（考虑波长网格）

5. **输入特征维度对齐**
   - 文档建议15维输入（7原始+8物理）
   - 当前实现使用 `data_preprocessing_raw.py` 提取的特征，需确认是否为15维

### 6.3 低优先级改进

6. **实现Neural Split-Step架构**
   - 按文档8A实现精确色散层+学习非线性层
   - 需要修改UPPE代码以支持非均匀压力分布（Phase A+）

7. **添加分区域评估**
   - 按低N/高N/电离区分别计算R²
   - 验证各区域是否达到 > 0.90 目标

---

## 七、总结

| 评估维度 | StandardMLP | TemporalMLP | TemporalLSTM |
|----------|-------------|-------------|--------------|
| 与文档一致性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 损失函数合规 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 架构设计合规 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 物理约束完整 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 预期性能达标 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 可解释性 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 实现复杂度 | 低 | 中 | 中高 |
| 推荐优先级 | 基准方法 | 可选扩展 | 实验性 |

**总体结论:**
- **StandardMLP** 完全符合指导文档标准规范，应作为基准方法
- **TemporalMLP** 和 **TemporalLSTM** 存在Sigmoid输出、RDW损失缺失、超参数偏差等问题，需要改进后才能达到文档要求的性能目标
- 当前时序实现是标准的数据驱动时序扩展，未达到文档8A推荐的物理嵌入架构水平
- 建议优先修复高优先级问题（移除Sigmoid、添加RDW损失、调整超参数），再评估时序模型的实际性能
