**项目定位**

当前外层目录是 `Z:\H3I空芯光纤小组组会\Luna.jl-master`，真正的 Julia 包源码位于嵌套目录：

`Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master`

这是一个以 Julia 编写的非线性光学脉冲传播仿真包 `Luna`，核心能力是基于 UPPE/GNLSE 模拟波导和自由空间中的超快光场传播。仓库中还混入了本地扩展的反谐振空芯光纤数据生成、机器学习训练和 PyQt GUI 工具。

**1. 技术栈和主要依赖**

核心包：

- 语言：Julia `>= 1.9`
- 包名：`Luna`
- 版本：`0.6.2`
- 配置文件：[Project.toml](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\Project.toml:1)
- 主入口：[src/Luna.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Luna.jl:1)

主要 Julia 依赖按用途可分为：

- 数值计算与积分：`LinearAlgebra`、`Statistics`、`FFTW`、`QuadGK`、`Cubature`、`HCubature`、`NumericalIntegration`
- 特殊函数/求根/多项式：`SpecialFunctions`、`HypergeometricFunctions`、`Roots`、`FunctionZeros`、`Polynomials`
- 光学/物性数据：`CoolProp`、`PhysicalConstants`、`Unitful`
- 数据与文件：`HDF5`、`H5Zblosc`、`CSV`、`DelimitedFiles`
- 绘图与 Python 桥接：`PyPlot`、`PyCall`、`Conda`
- 并行与扫描：`Distributed`、`ArgParse`、`FileWatching`
- 优化与拟合：`Optim`、`BlackBoxOptim`、`FiniteDifferences`、`Dierckx`
- 文档：`Documenter`

本地扩展/实验脚本：

- Julia：`examples/simple_interface/data_generation.jl` 用 Luna 生成 HDF5 训练数据。
- Python：`h5py`、`numpy`、`scikit-learn`、`torch`、`matplotlib`、`PyQt5`。
- 机器学习模型：MLP、TemporalMLP、TemporalLSTM，定义在 [train_mlp.py](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\examples\simple_interface\train_mlp.py:101)。
- GUI：PyQt5 桌面应用 [hcf_predictor_gui.py](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\examples\simple_interface\hcf_predictor_gui.py:48)。

**2. 目录结构与模块职责**

核心目录：

```text
Luna.jl-master/
  Project.toml
  README.md
  src/
    Luna.jl
    Interface.jl
    Grid.jl
    Fields.jl
    Modes.jl
    LinearOps.jl
    Nonlinear.jl
    NonlinearRHS.jl
    RK45.jl
    Output.jl
    Processing.jl
    Plotting.jl
    Stats.jl
    Scans.jl
    PhysData.jl
    Capillary.jl
    Antiresonant.jl
    RectModes.jl
    StepIndexFibre.jl
    SimpleFibre.jl
    Ionisation.jl
    Raman.jl
    SFA.jl
    Tools.jl
    Utils.jl
    data/
  test/
  docs/
  examples/
```

核心模块职责：

- [src/Luna.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Luna.jl:51)：顶层模块，顺序 `include` 所有子模块，导出高层 API，并实现通用 `setup` 和 `run` 管线。
- [src/Interface.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Interface.jl:1)：高层用户接口，包含 `prop_capillary`、`prop_gnlse` 和 `Pulses` 子模块。
- [src/Grid.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Grid.jl:1)：时间/频率网格、包络网格、自由空间横向网格。
- [src/Fields.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Fields.jl:1)：输入光场，如高斯脉冲、sech 脉冲、数据驱动脉冲、噪声、时空场。
- [src/Modes.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Modes.jl:1)：模式抽象、模式集合、空间投影与模式耦合。
- `Capillary.jl`、`Antiresonant.jl`、`RectModes.jl`、`StepIndexFibre.jl`、`SimpleFibre.jl`：不同波导/光纤模型。
- `LinearOps.jl`：线性传播算子、色散、损耗、传播相位。
- `Nonlinear.jl`：Kerr、Raman、等离子体等非线性响应构造。
- `NonlinearRHS.jl`：非线性右端项和频域/时域变换器。
- [src/RK45.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\RK45.jl:1)：自适应 Runge-Kutta 传播求解器。
- [src/Output.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Output.jl:15)：内存输出、HDF5 输出、扫描结果保存、缓存续算。
- [src/Processing.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Processing.jl:1)：仿真结果后处理，频谱、时域场、能量、脉宽等。
- [src/Plotting.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Plotting.jl:1)：基于 PyPlot 的可视化。
- [src/Stats.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Stats.jl:1)：传播过程中的统计量收集。
- [src/Scans.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Scans.jl:121)：参数扫描，支持本地、范围、批次、队列、Slurm、Condor、SSH 执行。
- `PhysData.jl`：气体、玻璃、金属、Sellmeier、非线性系数等物理数据。
- `Ionisation.jl`：ADK/PPT 等强场电离模型与缓存。
- `Raman.jl`：气体/介质 Raman 响应。
- `SFA.jl`：强场近似谱计算。
- `Tools.jl`：物理量辅助计算，如孤子阶数、非线性长度、RDW 波长等。
- `Utils.jl`：缓存路径、FFTW wisdom、Git 元信息、HDF5 字典工具。

本地实验目录：

- `examples/simple_interface/data_generation.jl`：反谐振/毛细管数据生成。
- `examples/simple_interface/data_preprocessing.py`：HDF5 到 `.npy` 训练数据预处理。
- `examples/simple_interface/train_mlp.py`：训练 MLP/TemporalMLP/TemporalLSTM。
- `examples/simple_interface/hcf_predictor_gui.py`：加载模型并做交互式预测。
- `examples/simple_interface/models*`、`processed_data*`、`training_data`、`random_samples_output`：模型、处理数据、样本和图像产物，体积较大，属于实验结果而非库源码。

**3. 数据流和路由设计**

这个项目没有 Web 路由。核心“路由”是 Julia 多重派发和高层接口分流。

核心仿真数据流：

```text
用户参数
  -> Interface.prop_capillary / prop_gnlse
  -> *_args 构造 grid、pulse、mode、linop、nonlinear response、output
  -> Luna.setup 按 grid/mode 类型选择变换器
  -> Luna.run
  -> RK45.solve_precon 自适应传播
  -> Output.MemoryOutput 或 Output.HDF5Output
  -> Processing / Plotting / Stats 后处理
```

关键入口：

- `prop_capillary` 在 [Interface.jl:358](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Interface.jl:358)，内部调用 `prop_capillary_args`，再调用 `Luna.run`。
- `prop_capillary_args` 在 [Interface.jl:373](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Interface.jl:373)，负责构造仿真组件。
- `prop_gnlse` 在 [Interface.jl:974](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Interface.jl:974)。
- 通用传播器 `run` 在 [Luna.jl:366](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Luna.jl:366)。

多重派发路由：

- `setup(grid::RealGrid, ...)`：场解析传播。
- `setup(grid::EnvGrid, ...)`：包络传播。
- `setup(..., modes::ModeCollection, ...)`：多模传播。
- `setup(..., Hankel.QDHT, ...)`：径向自由空间传播。
- `setup(..., Grid.FreeGrid, ...)`：完整 3D 自由空间传播。

输出路由：

- `filepath == nothing` -> `MemoryOutput`
- `filepath != nothing` 且无扫描 -> `HDF5Output`
- `scan != nothing` -> `ScanHDF5Output`

对应实现位于 [Interface.jl:879](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Interface.jl:879)。

扫描数据流：

```text
Scan 定义参数空间
  -> addvariable!
  -> runscan 根据 exec 类型分发
  -> 每个 scanidx 调用用户函数
  -> 每个样本写 HDF5 或汇总到 collected HDF5
```

执行模式包括 `LocalExec`、`RangeExec`、`BatchExec`、`QueueExec`、`SlurmExec`、`CondorExec`、`SSHExec`，分发函数在 [Scans.jl:275](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\src\Scans.jl:275) 起。

本地 ML 数据流：

```text
data_generation.jl
  -> sample_XXXXXX.h5，包含 Eω/z/stats/physics_features/prop_capillary_args/grid
  -> data_preprocessing.py
  -> X_train/val/test.npy, y_train/val/test.npy, temporal spectra, scaler, params
  -> train_mlp.py
  -> best_model.pth / final_model.pth / metrics.json / history.json
  -> hcf_predictor_gui.py 或 evaluate_models_heatmap.py
```

**4. 当前已实现的功能清单**

核心 Luna 已实现：

- 高层 HCF/毛细管传播接口 `prop_capillary`
- 高层 GNLSE 传播接口 `prop_gnlse`
- 模式平均传播
- 多模传播
- 场解析传播和包络传播
- 毛细管 Marcatili 模式
- 反谐振模式模型，包括 Zeisberger/Vincetti 风格实现
- 矩形波导模式
- 阶跃折射率光纤模式
- 简化光纤模式
- 自由空间径向传播
- 完整 3D 自由空间传播
- Kerr 非线性
- Raman 响应
- 强场电离和等离子体响应
- shot noise / 噪声模型
- HDF5 输出、内存输出、缓存续算
- 参数扫描：本地、批处理、文件队列、Slurm、Condor、SSH
- 统计量收集：能量、峰值功率、峰值强度、FWHM、电子密度、ZDW 等
- 后处理：谱强度、时域场、能量、脉宽、相干性等
- 绘图：传播二维图、频谱、时域图、统计图、spectrogram
- 文档系统：Documenter
- 较完整测试集：[test/runtests.jl](Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master\test\runtests.jl:1) 覆盖 Maths、PhysData、Capillary、Ionisation、Output、Multimode、Polarisation、Scans、Raman、Kerr、LinearOps、Modes、Freespace、Interface、GNLSE、Noise 等。

本地扩展已实现：

- 反谐振空芯光纤训练数据生成
- 拉丁超立方采样
- HDF5 样本结构校验
- 光谱和传播图批量可视化
- HDF5 到 ML 数据集预处理
- 特征提取：能量、脉宽、压力、长度、直径、壁厚、`beta2`、`gamma`、孤子阶数、`Aeff`、`neff` 等
- 模型训练：StandardMLP、TemporalMLP、TemporalLSTM
- 训练曲线、预测图、热力图评估
- PyQt5 GUI 单样本/批量预测和训练结果查看

**5. 需要特别注意的代码模式或技术债**

- 实际项目根目录有一层重复嵌套：外层 `Luna.jl-master` 内还有一个 `Luna.jl-master`。后续运行 Julia 包命令时应进入内层目录。
- 核心库大量使用 Unicode 变量名，如 `ω`、`λ`、`τ`、`β`。这符合 Julia 风格，但 Windows PowerShell 输出中出现了编码错乱，阅读中文/希腊字母文档时要注意终端编码。
- `src/Luna.jl` 在文件末尾执行预编译用的示例传播调用，这会增加加载/预编译成本。
- 全局设置 `Luna.settings` 控制 FFTW planning 和线程，属于全局可变状态；并行或多任务场景需要小心。
- `Output.HDF5Output` 默认带缓存续算机制，复用同一路径时会受 `cachehash` 和已有 HDF5 结构影响。
- `Scans.QueueExec` 使用文件队列和 `global` 状态，跨进程并发依赖锁文件和 HDF5 状态，失败恢复需要谨慎。
- README 明确说低层接口示例“不主动维护且不保证可运行”，不要把 `examples/low_level_interface` 当作稳定 API。
- 源码里存在若干 TODO：例如 `Capillary.jl` 中吸收材料传输硬编码、`Interface.jl` 中 DataPulse 峰值功率、`NonlinearRHS.jl` 并行化、`PhysData.jl` 部分物性参数、`Processing.jl` 某些模式重建输入等。
- 本地 ML/GUI 代码与原 Luna 包耦合在 `examples/simple_interface`，并且包含模型权重、`.npy`、`.h5`、图片、zip、日志等生成产物；这会让仓库体积、职责边界和版本管理变得混乱。
- ML 文档与当前代码有版本差异：文档中提过能量损失/RDW 损失等设计，但当前 `train_mlp.py` 的 `TemporalLoss` 已移除 energy loss，只保留 temporal/final/smoothness。
- Python 训练模型输出使用 `Sigmoid` 限制到 `[0,1]`，这适合归一化后的 log spectrum，但如果目标数据不是严格归一化区间，会带来表达能力限制。
- 部分脚本包含删除旧图像/清理 FFTW 缓存等写操作函数；本次只读，没有执行这些脚本。