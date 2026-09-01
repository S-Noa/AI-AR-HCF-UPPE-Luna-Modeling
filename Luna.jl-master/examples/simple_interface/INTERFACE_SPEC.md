# 项目模块接口规范文档 v3.1

## 1. 概述

本文档定义了数据集生成、数据预处理和模型训练三个核心模块之间的数据接口规范，确保各模块能够独立运行且无缝对接。

---

## 2. 模块架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  data_generation.jl │ ──→ │ data_preprocessing.py │ ──→ │   train_mlp.py   │
│   (数据集生成)      │     │   (数据预处理)        │     │   (模型训练)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      输出 .h5 文件            输出 .npy 文件           输出 .pth 模型
```

---

## 3. 模块1: data_generation.jl

### 3.1 功能
生成光纤脉冲传播仿真数据集，输出HDF5格式文件。

### 3.2 配置参数（文件顶部常量）

| 常量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `SMALL_BATCH_MODE` | Bool | `false` | `true`=100样本，`false`=10000样本 |
| `FIBRE_MODEL` | Symbol | `:antiresonant` | `:capillary` 或 `:antiresonant` |
| `WALL_THICKNESS_UM` | Float64 | `0.5` | 反谐振模型管壁厚度（μm，固定值） |
| `CHIRP_ENABLED` | Bool | `true` | `true`=随机采样，`false`=固定为0 |
| `CHIRP_RANGE` | Tuple | `(-3.0, 3.0)` | 啁啾参数采样范围 |
| `BASE_OUTPUT_DIR` | String | `"training_data"` | 输出目录基础名 |

### 3.3 输出目录命名规则

```
{BASE_OUTPUT_DIR}_{模型标识}_{管壁厚度}_{啁啾标识}_{small}
```

| 配置 | 目录名示例 |
|------|-----------|
| 反谐振, t=0.5, 啁啾启用, 完整 | `training_data_ar_t0p5` |
| 反谐振, t=0.5, 啁啾禁用, 小批量 | `training_data_ar_t0p5_nochirp_small` |
| 毛细管, 啁啾启用, 完整 | `training_data_cap` |

### 3.4 HDF5文件结构

每个样本文件 `sample_XXXXXX.h5` 包含以下数据集：

```
sample_XXXXXX.h5
├── Eω/                  # 复数光谱数据 (n_freq, n_z) 或 (n_z, n_freq)
├── z/                   # 传播距离数组 (n_z,)
├── stats/               # 统计信息组
│   └── ...
├── physics_features/    # 物理特征组
│   ├── beta2            # 群速度色散 (Float64)
│   ├── gamma            # 非线性系数 (Float64)
│   ├── N                # 孤子阶数 (Float64)
│   ├── L0               # 色散长度 (Float64)
│   ├── gamma_K          # Keldysh参数 (Float64)
│   ├── P_ratio          # 功率比 (Float64)
│   ├── Aeff             # 有效模场面积 (Float64)
│   ├── neff             # 有效折射率 (Float64)
│   ├── wallthickness    # 管壁厚度 (m, Float64)
│   ├── fibre_model      # 模型类型 (String)
│   └── chirp_enabled    # 啁啾是否启用 (Bool)
├── meta/                # 元数据组
│   └── ...
├── prop_capillary_args/ # 原始仿真参数
│   ├── energy           # 脉冲能量 (J)
│   ├── τfwhm            # 脉宽 (s)
│   ├── ϕ                # 啁啾参数
│   ├── λ0               # 中心波长 (m)
│   ├── pressure         # 气压 (bar)
│   ├── flength          # 光纤长度 (m)
│   └── radius           # 纤芯半径 (m)
└── grid/                # 网格信息（如存在）
    ├── ω/ωo             # 角频率数组
    └── sidx             # 采样索引
```

### 3.5 输出文件清单

| 文件 | 说明 |
|------|------|
| `sample_XXXXXX.h5` | 样本数据文件（100或10000个） |
| `data_generation.log` | 运行日志 |
| `success.log` | 成功样本记录 |
| `fail.log` | 失败样本记录 |

---

## 4. 模块2: data_preprocessing.py

### 4.1 功能
读取HDF5仿真数据，提取特征和光谱，进行标准化处理，输出NumPy格式训练数据。

### 4.2 输入接口

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `INPUT_DIR` | str | `"/private/Luna.jl-master/training_data"` | HDF5文件目录 |
| `OUTPUT_DIR` | str | `"processed_data"` | 输出目录 |

**注意**: `INPUT_DIR` 需指向 `data_generation.jl` 生成的输出目录。

### 4.3 输出文件清单

| 文件 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `X_train.npy` | `(N_train, 14)` | float64 | 训练集输入特征 |
| `X_val.npy` | `(N_val, 14)` | float64 | 验证集输入特征 |
| `X_test.npy` | `(N_test, 14)` | float64 | 测试集输入特征 |
| `y_temporal_train.npy` | `(N_train, n_z, 500)` | float64 | 训练集时序光谱 |
| `y_temporal_val.npy` | `(N_val, n_z, 500)` | float64 | 验证集时序光谱 |
| `y_temporal_test.npy` | `(N_test, n_z, 500)` | float64 | 测试集时序光谱 |
| `y_train.npy` | `(N_train, 500)` | float64 | 训练集最终光谱（兼容） |
| `y_val.npy` | `(N_val, 500)` | float64 | 验证集最终光谱（兼容） |
| `y_test.npy` | `(N_test, 500)` | float64 | 测试集最终光谱（兼容） |
| `z_train.npy` | `(N_train,)` | object | 训练集z位置数组（变长） |
| `z_val.npy` | `(N_val,)` | object | 验证集z位置数组（变长） |
| `z_test.npy` | `(N_test,)` | object | 测试集z位置数组（变长） |
| `scaler_X.joblib` | - | - | MinMaxScaler对象 |
| `processing_params.json` | - | - | 处理参数记录 |

### 4.4 特征向量定义（14维）

```
特征索引 | 特征名        | 单位    | 来源
---------|---------------|---------|------------------
0        | energy        | J       | sim_params
1        | tau           | s       | sim_params
2        | pressure      | bar     | sim_params
3        | length        | m       | sim_params
4        | diameter      | μm      | sim_params (radius*2*1e6)
5        | wallthickness | μm      | physics_features
6        | beta2         | s²/m    | physics_features
7        | gamma         | 1/(W·m) | physics_features
8        | N             | -       | physics_features
9        | L0            | m       | physics_features
10       | gamma_K       | -       | physics_features
11       | P_ratio       | -       | physics_features
12       | Aeff          | m²      | physics_features
13       | neff          | -       | physics_features
```

### 4.5 数据标准化规则

- **输入特征X**: 使用 `sklearn.preprocessing.MinMaxScaler` 全局标准化到 `[0, 1]`
- **时序光谱y_temporal**: 每个样本独立Min-Max标准化
- **最终光谱y_final**: 取时序光谱最后一个时间步（未额外标准化）

### 4.6 数据划分比例

```
总数据 → 训练集(80%) + 临时集(20%)
临时集 → 验证集(50%) + 测试集(50%)

最终比例: 训练集(80%) / 验证集(10%) / 测试集(10%)
```

---

## 5. 模块3: train_mlp.py

### 5.1 功能
加载预处理后的数据，训练神经网络模型，支持三种架构选择。

### 5.2 配置参数

| 常量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `MODEL_TYPE` | str | `"lstm"` | `"mlp"` / `"temporal"` / `"lstm"` |
| `INPUT_DIR` | str | `"processed_data"` | 预处理数据目录 |
| `OUTPUT_DIR` | str | `"models"` | 模型输出目录 |

### 5.3 模型架构对比

| 架构 | 类名 | 输入 | 输出 | 适用场景 |
|------|------|------|------|----------|
| `mlp` | `StandardMLP` | `(batch, 14)` | `(batch, 500)` | 仅预测最终光谱，向后兼容 |
| `temporal` | `TemporalMLP` | `(batch, 14)` + z_pos | `(batch, n_z, 500)` | 时序MLP，全过程预测 |
| `lstm` | `TemporalLSTM` | `(batch, 14)` + z_pos | `(batch, n_z, 500)` | **推荐**，LSTM时序建模 |

### 5.4 模型前向接口

所有模型统一接口：

```python
def forward(self, x, z_positions=None):
    """
    参数:
        x: (batch, input_dim) - 输入特征
        z_positions: (batch, n_z) - z位置数组（时序模型需要）
    
    返回:
        temporal_output: (batch, n_z, output_dim) 或 None
        final_output: (batch, output_dim)
    """
```

### 5.5 损失函数

```
L_total = L_temporal + α·L_final + β·L_energy + γ·L_smoothness

其中:
- L_temporal: 时序光谱MSE（时序模型）
- L_final: 最终光谱MSE
- L_energy: 能量守恒约束
- L_smoothness: 时序平滑性约束
- α=0.5, β=0.1, γ=0.01
```

### 5.6 输出文件清单

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 验证集最佳模型权重 |
| `final_model.pth` | 最终训练模型权重 |
| `training_history.json` | 完整训练历史（每个epoch的loss、lr等） |
| `evaluation_metrics.json` | 测试集评估指标 |
| `model_config.json` | 模型配置信息 |

### 5.7 training_history.json 结构

```json
{
  "config": {
    "model_type": "lstm",
    "num_epochs": 500,
    "batch_size": 256,
    "use_temporal": true
  },
  "epochs": [
    {
      "epoch": 1,
      "train": {
        "total_loss": 0.1234,
        "temporal_loss": 0.0891,
        "final_loss": 0.0343,
        "energy_loss": 0.0000,
        "smoothness_loss": 0.0000
      },
      "val": {
        "total_loss": 0.0987,
        "temporal_loss": 0.0712,
        "final_loss": 0.0275,
        "energy_loss": 0.0000,
        "smoothness_loss": 0.0000,
        "r2_score": 0.8523
      },
      "learning_rate": 0.001,
      "epoch_time": 12.34
    }
  ],
  "best_epoch": 45,
  "best_val_loss": 0.0567,
  "total_epochs": 120,
  "early_stopped": true
}
```

---

## 6. 模块间数据流验证

### 6.1 数据格式一致性检查

| 检查项 | data_generation.jl | data_preprocessing.py | train_mlp.py | 状态 |
|--------|-------------------|----------------------|--------------|------|
| 输入特征维度 | - | 14维 | 接收14维 | ✅ |
| 光谱输出维度 | (n_freq, n_z) | (n_z, 500) | 接收(n_z, 500) | ✅ |
| 最终光谱维度 | - | (500,) | 接收(500,) | ✅ |
| z位置数据 | (n_z,) | object数组 | 填充后接收 | ✅ |
| 数据类型 | Float64 | Float64 | Float32 | ✅ |
| HDF5字段 | physics_features | 读取并解析 | - | ✅ |

### 6.2 关键接口对接点

**对接点1: HDF5 → 预处理**
- data_generation.jl 输出 `sample_XXXXXX.h5` 到 `{output_dir}/`
- data_preprocessing.py 从 `INPUT_DIR` 读取 `.h5` 文件
- **要求**: `INPUT_DIR` 必须指向 data_generation.jl 的 `output_dir`

**对接点2: .npy → 训练**
- data_preprocessing.py 输出 `.npy` 文件到 `processed_data/`
- train_mlp.py 从 `INPUT_DIR` 读取 `.npy` 文件
- **要求**: 两模块的 `INPUT_DIR`/`OUTPUT_DIR` 需匹配

### 6.3 运行顺序

```bash
# 步骤1: 生成数据集（Julia）
julia data_generation.jl
# 输出: training_data_ar_t0p5/*.h5

# 步骤2: 数据预处理（Python）
# 修改 data_preprocessing.py 中的 INPUT_DIR 指向上述目录
python data_preprocessing.py
# 输出: processed_data/*.npy

# 步骤3: 模型训练（Python）
python train_mlp.py
# 输出: models/*.pth, *.json
```

---

## 7. 兼容性说明

### 7.1 向后兼容

- `train_mlp.py` 的 `StandardMLP` 仅使用 `y_train/val/test.npy`（最终光谱）
- 当 `y_temporal_*.npy` 不存在时，自动回退到非时序训练模式
- `data_preprocessing.py` 始终输出 `y_*.npy` 以兼容旧模型

### 7.2 模型切换

修改 `train_mlp.py` 第27行即可切换模型：

```python
MODEL_TYPE = "lstm"    # 时序LSTM（推荐）
MODEL_TYPE = "temporal" # 时序MLP
MODEL_TYPE = "mlp"     # 标准MLP（仅最终光谱）
```

---

## 8. 附录: 标准MLP模型局限性分析

### 8.1 技术原因

`StandardMLP` 仅能预测最终时刻光谱，无法预测全过程演化：

1. **输入/输出设计限制**
   - 输入: `(batch, input_dim)` - 仅包含光纤参数，无时间/位置信息
   - 输出: `(batch, output_dim)` - 仅输出一维光谱向量
   - 无z位置输入接口，无法区分不同传播位置

2. **网络结构限制**
   - 纯全连接层堆叠，无序列建模能力
   - 无循环连接（RNN/LSTM）或注意力机制
   - 无法捕捉时间/空间依赖关系

3. **序列处理能力缺失**
   - 无状态传递机制，每个样本独立处理
   - 无法建模 `"输入参数 → 演化过程 → 输出光谱"` 的物理过程

### 8.2 对比: 时序模型如何解决

| 能力 | StandardMLP | TemporalLSTM |
|------|-------------|--------------|
| 输入 | 仅参数 | 参数 + z位置序列 |
| 序列建模 | ❌ | ✅ LSTM |
| 时序输出 | ❌ | ✅ (n_z, 500) |
| 物理过程建模 | ❌ | ✅ 逐步演化 |

---

## 9. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v3.0 | 2025-04 | 初始重构版本，支持时序数据 |
| v3.1 | 2025-04 | 添加配置化参数、动态目录命名、架构切换 |
