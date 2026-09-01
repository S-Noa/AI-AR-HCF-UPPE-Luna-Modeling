# 暑期计划汇报与下一阶段路线

## 1. 暑期计划完成情况总览

暑期计划围绕四条线展开：

1. 用 early-dense / Luna 数据重新检查 RNN baseline，确认递推模型在本任务上的真实表现。
2. 为开题综述做准备，整理非线性光纤、空芯光纤、超连续谱、AI 预测与逆向设计文献。
3. 搜索强电离或强 UV 候选样本，为后续物理分析和 inverse design 提供目标区域。
4. 重新梳理 Reinforcement Learning 与当前项目结合的逻辑，并和 Bayesian optimization、遗传算法、粒子群、微分进化、tandem network 等方法对比。

阶段性结论可以概括为：

```text
Forward surrogate 已经具备可用基础；
RNN baseline 说明 one-step accuracy 不等于 stable autoregressive rollout；
extreme sample search 给出了强 UV / 强非线性候选区域；
RL 更适合作为 forward surrogate 之上的 inverse-design layer，而不是替代 CNN/Transformer。
```

## 2. 开题综述规划：主线、章节、图表设计

### 2.1 综述主线

建议开题综述采用以下逻辑：

```text
应用需求
  -> 超连续谱与 UV 光源
  -> 气体填充空芯/反谐振光纤的优势
  -> UPPE/GNLSE 等高保真仿真方法
  -> 大规模参数扫描与逆向设计的计算瓶颈
  -> AI forward surrogate
  -> inverse design 与强化学习优化
  -> 本课题定位：面向 AR-HCF 脉冲传播的 forward prediction + inverse design
```

### 2.2 建议章节

1. 研究背景与应用需求  
   介绍超连续谱、UV 光源、OCT、高光谱成像、遥感、光谱学、气体传感等应用。

2. 空芯光纤与反谐振空芯光纤  
   说明低玻璃重叠、高损伤阈值、气压可调色散、宽带低损耗窗口等优势。

3. 超快非线性传播物理与数值模拟  
   介绍色散、Kerr 非线性、孤子动力学、色散波、电离和等离子体响应；比较 NLSE/GNLSE/UPPE。

4. AI 在超快光子学与光纤非线性中的 forward modeling  
   归纳 ANN、RNN、CNN、Transformer、PINN、physics-based deep learning 等路线。

5. 光纤与光子结构 inverse design  
   对比 grid/random search、Bayesian optimization、GA/PSO/DE、tandem network、invertible network、gradient-based surrogate optimization。

6. 强化学习用于 inverse design 的潜力  
   说明 RL 的 agent/action/environment/reward/policy 框架，以及它如何嵌入当前项目。

7. 本课题技术路线  
   使用 Luna/UPPE 生成数据，训练 CNN/Transformer forward surrogate，结合 extreme sample search 和 RL/其他优化方法做 inverse design，最后用 Luna/实验验证。

### 2.3 建议图

| 图号 | 图名 | 内容 |
|---|---|---|
| Fig. 1 | 研究背景总图 | 超连续谱/UV 光源应用、AR-HCF 结构、输入脉冲到输出宽带谱 |
| Fig. 2 | 物理传播机制图 | 色散、Kerr、孤子裂变、色散波、电离/等离子体共同作用 |
| Fig. 3 | 数值模拟与 AI 代理模型 | UPPE/Luna 生成数据，AI 学习 `parameters -> S(z, lambda)` |
| Fig. 4 | 模型结构对比 | RNN autoregressive、CNN direct map、Transformer z-token direct map |
| Fig. 5 | Extreme sample search | 参数空间、强电离判据、UV-rich 样本频域演化图 |
| Fig. 6 | Inverse design 方法谱系 | BO、GA、PSO、DE、gradient-through-surrogate、tandem/INN、RL |
| Fig. 7 | RL inverse design 闭环 | Agent 选参数，surrogate 快速预测，reward 评价，Luna 验证 |

## 3. 文献引用表格

说明：以下表格优先来自 `R:\Zotero` 中已有文件。部分条目暂以题名级信息记录，后续正式开题报告中可用 Zotero 导出 BibTeX 补全年份、作者、期刊、DOI。

| 编号 | 主题分类 | 文献/文件名 | 主要内容 | 与本项目关系 | 建议引用位置 |
|---:|---|---|---|---|---|
| 1 | 基础教材 | Agrawal - Nonlinear fiber optics | 非线性光纤基本理论、色散、SPM、孤子 | 解释 NLSE/GNLSE、孤子和超连续谱基础 | 传播物理基础 |
| 2 | Supercontinuum / review | Fibre supercontinuum generation Progress and perspectives | 超连续谱生成进展综述 | 总述 SC 应用与机制 | 引言、研究背景 |
| 3 | Supercontinuum / UV | High-energy pulse self-compression and ultraviolet generation through soliton dynamics in hollow capillary fibres | 空芯毛细管中高能脉冲自压缩与 UV 产生 | 说明气体空芯结构可产生 UV | 背景、目标应用 |
| 4 | Supercontinuum / UV | Bright Spatially Coherent Wavelength-Tunable Deep-UV Laser Source Using an Ar-Filled Photonic Crystal Fiber | Ar 填充 PCF 深紫外可调光源 | 与 Ar、UV dispersive wave 直接相关 | UV 产生背景 |
| 5 | Supercontinuum / UV | UV Continuum Generation in Ar-Filled Hollow-Core PCF | Ar 填充空芯 PCF 的 UV 连续谱 | 支撑本项目 UV 目标 | 背景、样本筛选 |
| 6 | Supercontinuum / UV | Zero-dispersion wavelength decreasing photonic crystal fibers for ultraviolet-extended supercontinuum generation | 通过色散设计扩展 UV 超连续谱 | 说明色散调控对 UV 扩展重要 | 物理机制 |
| 7 | Hollow-core fiber | A Review of Antiresonant Hollow-Core Fiber-Assisted Spectroscopy of Gases | 反谐振空芯光纤气体光谱应用综述 | 说明 AR-HCF 在气体/光谱学中的价值 | 背景 |
| 8 | Hollow-core fiber | Recent Advancement of Anti-Resonant Hollow-Core Fibers for Sensing Applications | AR-HCF 传感应用进展 | 支撑应用场景 | 应用背景 |
| 9 | Hollow-core fiber | Review on Hollow-Core Fiber Based Multi-Gas Sensing Using Raman Spectroscopy | 空芯光纤多气体 Raman 传感综述 | 说明空芯光纤与气体相互作用优势 | 应用背景 |
| 10 | Hollow-core fiber | Loss in hollow-core optical fibers mechanisms, scaling rules, and limits | 空芯光纤损耗机制和极限 | 解释结构参数、损耗和设计约束 | 光纤结构 |
| 11 | Hollow-core fiber | Hollow-Core Optical Fibers for Telecommunications and Data Transmission | 空芯光纤通信应用综述 | 泛化说明 HCF 平台价值 | 背景 |
| 12 | AR-HCF modeling | Poor-man's model of hollow-core anti-resonant fibers | AR-HCF 简化模型 | 与模式/损耗/反谐振窗口分析相关 | 模型选择 |
| 13 | HCF modeling | Accuracy of the capillary approximation for gas-filled kagome-style photonic crystal fibers | 毛细管近似在气体空芯 PCF 中的准确性 | 说明简化模型适用边界 | UPPE/模型讨论 |
| 14 | HCF modeling | Linear and nonlinear modeling of light propagation in hollow-core photonic crystal fiber | 空芯 PCF 线性和非线性传播建模 | 支撑 HCF 中非线性传播模型 | 传播建模 |
| 15 | Simulation | Open source, heterogeneous, nonlinear optics simulation | 开源非线性光学仿真框架 | 可作为 Luna/开源仿真背景扩展 | 数值仿真 |
| 16 | Simulation | Luna.jl / C. Brahms and J. C. Travers, Luna.jl | UPPE/GNLSE 开源仿真工具 | 当前数据生成核心工具 | 数据生成 |
| 17 | Simulation | Couairon and Mysyrowicz - Femtosecond filamentation in transparent media | UPPE/强场传播相关综述 | 解释 UPPE 对 broadband/ionization 的必要性 | 数值模型 |
| 18 | AI forward surrogate | Maximizing supercontinuum bandwidths in gas-filled hollowcore fibers using artificial neural networks | ANN 优化气体空芯光纤 SC 带宽 | 与本项目 forward/inverse surrogate 高相关 | AI 光纤设计 |
| 19 | AI forward surrogate | Predicting ultrafast nonlinear dynamics in fibre optics with a recurrent neural network | RNN 预测超快非线性传播 | 当前 Luna RNN baseline 的来源 | 相关工作、baseline |
| 20 | AI forward surrogate | Conditional Recurrent Neural Networks for broad applications in nonlinear optics | 条件 RNN 用于非线性光学 | 与给 RNN 加物理参数 conditioning 相关 | RNN 改进 |
| 21 | AI forward surrogate | Predicting nonlinear multi-pulse propagation in optical fibers via a lightweight convolutional neural network | CNN 预测多脉冲非线性传播 | 支撑 CNN surrogate 合理性 | CNN baseline |
| 22 | AI forward surrogate | Prediction of supercontinuum spectrum based on LSTM with multi-head attention mechanism | LSTM+attention 预测 SC 谱 | 与 RNN/attention 结合相关 | AI 方法综述 |
| 23 | AI forward surrogate | Predicting the Evolution of the Supercontinuum Generation With CNN-LSTM Model | CNN-LSTM 预测 SC 演化 | 与传播图预测任务接近 | 相关工作 |
| 24 | AI forward surrogate | Nonlinear autoregressive with external input neural network for predicting the nonlinear dynamics of supercontinuum generation in optical fibers | NARX 预测 SC 非线性动力学 | 与条件递推模型相关 | RNN/NARX 对比 |
| 25 | AI forward surrogate | Modelling the spectro-temporal evolution of a frequency comb in a nonlinear optical fibre with a feedforward neural network | 前馈网络建模频梳谱时演化 | 支撑直接映射路线 | Forward surrogate |
| 26 | AI forward surrogate | Predicting nonlinear reshaping of periodic signals in optical fibre with a neural network | 神经网络预测周期信号非线性重塑 | 非线性光纤 AI 建模例子 | 相关工作 |
| 27 | AI forward surrogate | Fast Predicting the Complex Nonlinear Dynamics of Mode-Locked Fiber Laser by a Recurrent Neural Network | RNN 快速预测锁模光纤激光动力学 | 说明 RNN 在光纤动力学中有效但任务不同 | RNN 背景 |
| 28 | AI forward surrogate | Accurate modeling of ultrafast nonlinear pulse propagation in multimode gain fiber | 多模增益光纤超快传播建模 | 支持复杂光纤系统 ML 建模 | 相关工作 |
| 29 | Transformer | NIPS-2017-attention-is-all-you-need-Paper | Transformer/self-attention 基础 | 当前 Transformer surrogate 的模型来源 | 方法 |
| 30 | Physics-based ML | Physics-based deep learning for modeling nonlinear pulse propagation in optical fibers | 物理约束深度学习建模非线性传播 | 与 physics-aware loss 和 surrogate 可信性相关 | 方法 |
| 31 | Physics-informed ML | Physics-informed neural networks A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations | PINN 基础文献 | 逆问题和物理约束学习背景 | 方法综述 |
| 32 | Physics-informed ML | Predicting Ultrafast Nonlinear Dynamics in Fiber Optics by Enhanced Physics-Informed Neural Network | 增强 PINN 预测光纤超快非线性动力学 | 可与纯数据驱动 surrogate 对照 | 相关工作 |
| 33 | Physics-informed ML | Physics-Informed Neural Network for Optical Fiber Parameter Estimation From the Nonlinear Schrodinger Equation | PINN 估计光纤参数 | inverse/parameter estimation 参考 | 逆问题 |
| 34 | Physics-informed ML | Physics-Informed Neural Networks for Fiber Laser Amplification and Its Associated Effects | PINN 用于光纤激光放大 | 物理约束 ML 在光纤中的应用 | 方法综述 |
| 35 | Physics-based ML | Physics-Based Deep Learning for Fiber-Optic Communication Systems | 光纤通信中的物理深度学习 | 支撑 physics-based/physics-informed 讨论 | 方法综述 |
| 36 | AI / photonics review | Machine Learning and Applications in Ultrafast Photonics | 机器学习在超快光子学中的应用 | 总述 AI+ultrafast photonics | 综述开头 |
| 37 | AI / photonics review | Advancements in ultrafast photonics confluence of nonlinear optics and intelligent strategies | 非线性光学与智能策略结合 | 支撑 AI+非线性光学趋势 | 引言 |
| 38 | AI / photonics review | Intelligent nanophotonics merging photonics and artificial intelligence at the nanoscale | AI 与纳米光子学综述 | 说明 AI photonics 大背景 | 引言 |
| 39 | AI / photonics review | Review for artificial intelligence-based micro-nanophotonic devices | AI 微纳光子器件综述 | 可放在 inverse design 背景 | 综述 |
| 40 | HCF inverse design | Artificial intelligence designer for optical Fibers Inverse design of a Hollow-Core Anti-Resonant fiber based on a tandem neural network | Tandem neural network 逆向设计 HCF | 与 AR-HCF inverse design 直接相关 | 逆向设计 |
| 41 | HCF inverse design | Use of machine learning to efficiently predict the confinement loss in anti-resonant hollow-core fiber | ML 预测 AR-HCF confinement loss | 结构参数到性能预测 | Forward/inverse design |
| 42 | HCF inverse design | Discovering extremely low confinement-loss anti-resonant fibers via swarm intelligence | 群智能搜索低损耗 AR-HCF | PSO/群智能参考 | 优化方法 |
| 43 | RL / HCF design | Design of Negative Curvature Hollow Core Fiber Based on Reinforcement Learning | RL 设计负曲率空芯光纤 | RL 用于 HCF 结构设计的直接参考 | RL 相关工作 |
| 44 | RL / nonlinear fiber | Machine learning multitarget optimization for ultrashort pulse nonlinear dynamics in optical fibers | DNN+DRL 优化超快光纤非线性动力学 | 当前 RL inverse design 的核心范例 | RL 范例 |
| 45 | Optimization | Spectral optimization of supercontinuum shaping using metaheuristic algorithms | 元启发式算法优化 SC shaping | GA/PSO/SA 与 RL 对照 | 优化方法 |
| 46 | Optimization | Programmable liquid-core fibers Reconfigurable local dispersion control for computationally optimized ultrafast supercontinuum generation | 可编程液芯光纤与计算优化 SC | 主动调控和优化闭环参考 | 优化/主动控制 |
| 47 | Optimization | Smart Control of Supercontinuum Generation by Machine Learning Towards Multiphoton Microscopy Applications | ML 控制 SC 以服务多光子显微 | 应用导向优化参考 | 应用与控制 |
| 48 | Optimization | Real-time fine-tuning ultrafast supercontinuum generation and pulse compression in hybrid nonlinear multipass cavity | 实时调谐 SC 和脉冲压缩 | 实时优化/实验闭环参考 | 控制优化 |
| 49 | Inverse design | Inverse Design of Broadband Dispersion Compensation Fiber Based on Deep Learning and Differential Evolution Algorithm | DL + DE 逆向设计色散补偿光纤 | DE 与神经网络结合案例 | 方法对比 |
| 50 | Inverse design | Machine learning aided inverse design for few-mode fiber weak-coupling optimization | ML 辅助少模光纤逆向设计 | 光纤 inverse design 案例 | 方法对比 |
| 51 | Inverse design | Inverse design of discrete Raman amplifiers using an invertible neural network for ultra-wideband optical transmission based on hollow core fibers | INN 逆向设计 Raman 放大器 | 可逆网络处理多解逆问题 | INN 对比 |
| 52 | Liquid-core / nonlinear | Liquid-Core Optical Fibers A Dynamic Platform for Nonlinear Photonics | 液芯光纤非线性光子学综述 | 作为非线性光纤平台横向比较 | 背景 |
| 53 | Liquid-core / SC | Two-octave supercontinuum generation in a water-filled photonic crystal fiber | 水填充 PCF 两倍频程 SC | 液芯 SC 参考 | SC 背景 |
| 54 | Liquid-core / SC | Supercontinuum generation in a water-core photonic crystal fiber | 水芯 PCF SC | 非气体光纤 SC 对照 | 背景 |
| 55 | Liquid-core / SC | Mid-IR supercontinuum generation in an integrated liquid-core optical fiber filled with CS2 | CS2 液芯光纤中红外 SC | 中红外拓展参考 | 应用背景 |
| 56 | Liquid-core / dynamics | Hybrid soliton dynamics in liquid-core fibres | 液芯光纤混合孤子动力学 | 孤子动力学横向参考 | 物理机制 |
| 57 | Liquid-core / dynamics | Tailoring soliton fission at telecom wavelengths using composite-liquid-core fibers | 复合液芯光纤调控孤子裂变 | 色散/非线性调控示例 | 物理机制 |
| 58 | Liquid-core / dynamics | Local distributed control of soliton fission in liquid-core optical fibers | 局域分布式控制孤子裂变 | 与 spatial/parameter control 和 inverse design 相关 | 优化背景 |
| 59 | Nonlinear dynamics | Soliton trapping of dispersive waves in tapered optical fibers | 锥形光纤中色散波捕获 | 解释色散波与孤子相互作用 | 物理机制 |
| 60 | Multimode nonlinear optics | Physics of highly multimode nonlinear optical systems | 高多模非线性光学综述 | 拓展复杂非线性传播背景 | 综述拓展 |
| 61 | Optical neural networks | Deep physical neural networks trained with backpropagation | 物理神经网络反向传播训练 | 与光子学 AI 大背景相关 | 可选背景 |
| 62 | Optical neural networks | Fully forward mode training for optical neural networks | 光学神经网络全前向训练 | 光学 AI 前沿参考 | 可选背景 |
| 63 | Imaging / UV application | Multi-scale tissue fluorescence mapping with fiber optic ultraviolet excitation and generative modeling | 光纤 UV 激发和生成式建模 | UV 应用和 generative ML 参考 | 应用背景 |
| 64 | Source technology | Mid-IR Picosecond Laser with Broad Wavelength Tunability and High Peak Power | 可调中红外皮秒激光源 | 说明不同波段脉冲应用差异 | 引言 |
| 65 | Sensing / HCF | Structure design and application of hollow core microstructured optical fiber gas sensor A review | 空芯微结构光纤气体传感综述 | HCF sensing 应用背景 | 引言 |

## 4. 强电离样本：参数区间、物理判据、频域演化图特征

### 4.1 源码默认搜索区间

当前 strong-UV / strong-ionization 搜索来自 `anti_resonant_simulation.jl` 的 `--search-extreme-samples` 模式。默认参数网格为：

```text
energy:         1.5--3.0 uJ, step 0.5 uJ
tau:            5--12 fs, step 1 fs
pressure:       10--50 bar, step 5 bar
diameter:       10--25 um, step 5 um
wall thickness: 0.65 um
length:         50 cm
gas:            Ar
chirp:          0
lambda0:        1030 nm
```

完整网格大小为：

```text
4 energy values x 8 tau values x 9 pressure values x 4 diameter values = 1152 candidates
```

由于候选生成顺序是：

```julia
for energy_uJ in energies, tau_fs in taus, pressure_bar in pressures, diameter_um in diameters
```

高编号段自然集中在高能量区域。例如 `candidate_id >= 865` 基本对应 `E = 3.0 uJ`。

### 4.2 强电离倾向的重点筛选区间

从物理直觉和已有 extreme candidate 统计看，强电离或强 UV 候选更可能落在：

```text
energy:   2.5--3.0 uJ
tau:      5--9 fs
pressure: 20--50 bar
diameter: 10--15 um, 可扩展到 20 um
wall:     0.65 um
length:   50 cm
```

原因：

- 高能量和短脉冲提高峰值功率和峰值电场。
- 小芯径降低有效模场面积 `Aeff`，增强非线性系数 `gamma`。
- 中高气压提高气体非线性，同时改变色散与相位匹配条件。
- T650 结构给出特定反谐振窗口和损耗结构，可能影响 UV 色散波形成。

### 4.3 源码中的强电离判据

当前代码不是直接用“电离率”作为唯一判断，而是先计算多个物理量：

```text
E0_over_dataset_limit
E0_over_Emax_ppt
gamma_K
P_ratio
N
gamma
Aeff
beta2
```

其中 strong-ionization candidate 的判断阈值为：

```text
EXTREME_STRONG_E_LIMIT_RATIO = 0.5
EXTREME_STRONG_PPT_RATIO    = 0.10
EXTREME_STRONG_GAMMA_K      = 2.0
```

代码逻辑可以概括为：

```text
若 E0 / dataset_field_limit 较大，
或 E0 / PPT barrier-suppression field 较大，
或 Keldysh parameter gamma_K 较小，
则标记为 strong-ionization candidate。
```

同时，`combined_score` 会把 ionization tendency 和 UV prior tendency 合并排序。UV prior 主要鼓励：

```text
更大的 soliton order N
更强的 gamma
更高能量
更短脉冲
更小芯径
```

### 4.4 频域演化图应具备的特点

强电离 / 强 UV 样本的频域演化图一般会出现：

- 早期传播段快速展宽，尤其在前 `0--10 cm` 内出现明显频谱重构。
- 泵浦附近出现快速谱裂变、红移/蓝移或多峰结构。
- UV/可见短波端出现亮带、窄峰或孤立色散波。
- 若电离太强，短波区域可能变成连续碎裂、噪声状、高纹理密度结构。
- 长波端可能出现拖尾或较宽的 continuum band。

筛选“干净 UV 色散波”时，建议优先寻找：

```text
UV 短波端存在局域、连续、可追踪的亮带；
亮带与主泵浦谱分离清晰；
不是整片短波区域都被噪声状强纹理填满；
0--1 cm、0--5 cm、0--10 cm 图中能看到形成过程；
最终谱在 200--700 nm 有明确峰值，但没有严重边界截断。
```

## 5. Inverse Design 方法谱系

### 5.1 Grid search / Random search

基本思路：

```text
直接枚举或随机采样输入参数 -> Luna/forward surrogate 预测 -> 选 reward 最高样本
```

优点：

- 实现最简单。
- 易并行。
- 不依赖梯度或模型结构。

缺点：

- 高维参数空间中效率很低。
- 难以精细定位最优区域。
- 对 Luna 这种昂贵仿真不友好。

适合当前项目中的用途：

```text
作为 baseline search；
用于构造初始 extreme candidate library；
用于验证更复杂 inverse design 方法是否真正更好。
```

### 5.2 Bayesian Optimization

基本思路：

```text
用高斯过程或 tree/parzen estimator 建立 surrogate of objective；
通过 acquisition function 选择下一组参数；
用 Luna 或 trained surrogate 评估 reward；
逐步更新优化器。
```

优点：

- 样本效率高。
- 适合 expensive black-box optimization。
- 对少量真实 Luna 验证很有价值。

缺点：

- 高维、多目标、强约束问题会变难。
- 每次优化通常针对一个目标重新跑。
- 对复杂谱图 reward 的建模不一定稳定。

适合当前项目中的用途：

```text
作为 RL 之前最重要的对照方法；
可用于少量 Luna-in-the-loop 的真实仿真优化。
```

### 5.3 Genetic Algorithm / Particle Swarm Optimization / Differential Evolution

基本思路：

```text
维护一组候选参数；
通过 mutation/crossover/swarm update/evolution step 改进；
用目标函数 reward 筛选更优候选。
```

优点：

- 黑箱友好。
- 不要求梯度。
- 能处理多峰目标和离散/连续混合参数。

缺点：

- 迭代次数通常较多。
- 每个新任务可能都要重新搜索。
- 如果每步都用 Luna，成本很高。

适合当前项目中的用途：

```text
用 trained CNN/Transformer surrogate 快速评估大量候选；
top candidates 再交给 Luna 验证；
作为 RL 的强 baseline。
```

### 5.4 Gradient-based optimization through surrogate

基本思路：

```text
冻结 forward surrogate；
把输入参数视为可优化变量；
通过反向传播优化目标谱或 UV reward。
```

优点：

- 速度最快。
- 能直接利用 PyTorch forward model 的可微性。
- 适合目标谱匹配、UV fraction 最大化等可微目标。

缺点：

- 容易陷入局部最优。
- 参数边界和离散壁厚需要额外处理。
- 如果 surrogate 在外推区不可靠，优化可能钻模型漏洞。

适合当前项目中的用途：

```text
作为 inverse design 的第一阶段实现；
快速得到候选参数；
配合 Luna 验证和主动学习修正 surrogate。
```

### 5.5 Tandem Neural Network

基本思路：

```text
训练 forward network；
再训练 inverse network 输出参数；
用 frozen forward network 检查 inverse output 是否生成目标谱。
```

优点：

- 光子逆设计中常见。
- 推理速度快。
- 可将目标谱直接映射到参数。

缺点：

- inverse design 常有一对多问题，同一个目标谱可能对应多组参数。
- 容易平均化多个可行解，输出物理上不理想的中间参数。
- 通常需要大量高质量目标-参数数据。

适合当前项目中的用途：

```text
作为后续 inverse model 的候选；
不建议作为第一阶段主路线。
```

### 5.6 Invertible Neural Network / Normalizing Flow

基本思路：

```text
学习参数和谱目标之间的可逆或条件概率映射；
给定目标后采样多个可能参数解。
```

优点：

- 更适合一对多 inverse problem。
- 可以输出多组候选参数。
- 能表达不确定性。

缺点：

- 训练复杂。
- 对数据量和目标表示要求高。
- 需要额外筛选和 forward validation。

适合当前项目中的用途：

```text
作为第二阶段 inverse model；
与 forward surrogate 结合做候选生成和筛选。
```

### 5.7 Reinforcement Learning

基本思路：

```text
Agent 观察当前参数/目标差距；
Action 调整 E, tau, pressure, length, diameter, wall thickness；
Environment 使用 CNN/Transformer surrogate 或 Luna 返回谱图；
Reward 评价 UV 色散波质量、带宽、平坦度和物理约束；
Agent 更新 policy。
```

优点：

- 适合多步参数调整。
- Reward 可以很灵活，能同时编码 UV fraction、peak location、cleanliness、bandwidth、field limit 等目标。
- 训练出 policy 后，对新目标可以快速给出候选。
- 可自然接入主动学习闭环。

缺点：

- 训练成本高于简单优化。
- Reward 设计很关键。
- 如果 environment 只用 surrogate，可能出现 model exploitation，需要 Luna 验证。

适合当前项目中的定位：

```text
CNN/Transformer forward surrogate = fast environment
RL agent = inverse-design policy
Luna/UPPE = final verifier
```

## 6. 如何体现 RL 的优势

### 6.1 与其他方法的对照逻辑

| 方法 | 它解决什么 | 主要短板 | RL 可体现的优势 |
|---|---|---|---|
| Grid / Random search | 粗略搜索参数空间 | 维度灾难、效率低 | RL 利用历史反馈学习策略，不是盲搜 |
| Bayesian optimization | expensive black-box 样本高效优化 | 高维、多目标、复杂 reward 困难 | RL 更适合多步调整和复杂奖励 |
| GA / PSO / DE | 黑箱多峰优化 | 每个目标往往重新搜索，迭代多 | RL policy 学好后可迁移到相似目标 |
| Gradient-through-surrogate | 快速可微优化 | 易卡局部最优，易钻 surrogate 漏洞 | RL 可探索多个路径和非凸区域 |
| Tandem network | 目标谱到参数的快速映射 | 一对多逆问题导致平均解 | RL 不要求唯一反解，只要求高 reward |
| INN / flow | 多解分布采样 | 训练复杂，仍需筛选 | RL 可把多目标和约束直接写进 reward |

### 6.2 当前项目中的 RL 实现构想

第一阶段建议不要直接用 Luna 作为 RL environment，因为单次仿真成本太高。更合理的路线是：

```text
Step 1: 训练可信 forward surrogate
        CNN / Transformer: parameters -> S(z, lambda), final spectrum

Step 2: 构造 reward
        UV fraction
        UV peak strength
        clean dispersive-wave score
        bandwidth / flatness
        field-limit penalty
        manufacturability constraints

Step 3: 训练 RL agent
        action = adjust E, tau, p, L, d, t
        environment = frozen surrogate

Step 4: Luna verification
        top-k candidates -> UPPE simulation

Step 5: Active learning
        failed or surprising candidates -> add to dataset -> retrain surrogate
```

### 6.3 推荐 reward 草案

可以先设计一个可解释的 reward：

```text
R = a * UV_fraction_200_700
  + b * UV_peak_score
  + c * bandwidth_score
  - d * noisy_UV_penalty
  - e * field_limit_penalty
  - f * boundary_peak_penalty
```

其中：

- `UV_fraction_200_700`：200--700 nm 积分占比。
- `UV_peak_score`：目标 UV 区域是否有明确峰值。
- `bandwidth_score`：有效谱宽。
- `noisy_UV_penalty`：惩罚短波端满屏碎裂纹理。
- `field_limit_penalty`：惩罚过强电场或强电离不稳定区域。
- `boundary_peak_penalty`：惩罚峰值贴在 200 nm 或 2500 nm 边界，避免插值/窗口截断伪最优。

### 6.4 参考文献怎么讲 Zhao et al.

可以在组会中这样说：

```text
Zhao et al. used a DNN forward predictor together with deep reinforcement learning for multitarget optimization of ultrashort pulse nonlinear dynamics in optical fibers. This is close to our intended logic: the expensive nonlinear propagation simulation is replaced by a fast learned environment during optimization, while the final candidates still need high-fidelity validation.
```

但要强调区别：

```text
Their examples focus on nonlinear fiber dynamics such as SSFS and SC optimization.
Our target is gas-filled antiresonant hollow-core fiber propagation with UPPE-level data, UV dispersive-wave generation, and full S(z, lambda) evolution-map prediction.
```

## 7. 明天组会 PPT 建议页序

### Slide 1：暑期计划与当前目标

内容：

- 暑期四条计划。
- 当前目标：从 forward prediction 推进到 inverse design。
- 一句话结论：RNN 做 baseline，CNN/Transformer 做 forward surrogate，extreme search 给目标区域，RL 作为下一阶段优化框架。

### Slide 2：开题综述主线

内容：

- 应用需求：SC、UV、OCT、spectroscopy、sensing。
- 物理平台：gas-filled AR-HCF。
- 技术瓶颈：UPPE 准确但慢。
- 解决思路：AI surrogate + inverse design。

图：

- 综述逻辑流程图。

### Slide 3：文献地图

内容：

- HCF/AR-HCF。
- SC/UV generation。
- UPPE/GNLSE。
- AI forward prediction。
- Inverse design / optimization。
- RL in photonics。

图：

- 文献分类圆环或树状图。

### Slide 4：RNN baseline 复盘

内容：

- 普通 Luna RNN：stepwise R2 高，但 autoregressive 失败。
- Conditional RNN：加入 features + z 后 autoregressive 改善但仍不足。
- 结论：递推模型在本任务上有误差累积问题。

图：

- Stepwise vs autoregressive 对比图。

### Slide 5：Forward surrogate 当前状态

内容：

- Transformer temporal v3：
  - Final R2 = 0.9635
  - Temporal R2 = 0.9650
  - Temporal UV R2 = 0.8927
- CNN 作为强 baseline。

图：

- Transformer/CNN 预测传播图示例。

### Slide 6：Extreme sample search

内容：

- 搜索区间：1.5--3.0 uJ、5--12 fs、10--50 bar、10--25 um。
- 重点强电离区域：2.5--3.0 uJ、5--9 fs、20--50 bar、10--15 um。
- 强电离判据：`E0`、`gamma_K`、`P_ratio`、`N` 等。

图：

- 参数空间示意 + 一个强 UV 样本频域演化图。

### Slide 7：什么是干净 UV 色散波

内容：

- 短波端局域亮带。
- 与主泵浦分离。
- 形成路径可追踪。
- 避免满屏噪声状碎裂和边界伪峰。

图：

- 好样本 vs 过强电离/不干净样本对比。

### Slide 8：Inverse design 方法谱系

内容：

- Grid/random。
- BO。
- GA/PSO/DE。
- Gradient-through-surrogate。
- Tandem network。
- INN/flow。
- RL。

图：

- 方法对比表或方法树。

### Slide 9：为什么考虑 RL

内容：

- RL 能处理复杂 reward。
- 能做多步参数调整。
- 能和 surrogate environment 结合。
- 可形成 active learning 闭环。

图：

- Agent -> surrogate -> reward -> policy update -> Luna validation。

### Slide 10：下一阶段计划

内容：

1. 完成 early-dense 数据整理与模型重训。
2. 推进真实强度 global-log 版本。
3. 筛选 clean UV dispersive wave 样本。
4. 实现 surrogate-assisted inverse design。
5. 对比 BO、GA/PSO/DE、gradient optimization 和 RL。

## 汇报收束话术

```text
这个暑期的核心工作不是单独追求某一个模型指标，而是把整个课题路线重新梳理清楚：先用 Luna/UPPE 建立可靠数据，用 CNN/Transformer 形成 forward surrogate；再通过 RNN baseline 说明递推预测的局限；然后用 extreme search 找到强 UV/强电离候选区域；最后把 inverse design 推进为下一阶段重点，其中 reinforcement learning 是一个值得深入比较的优化框架。
```
