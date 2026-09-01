using HDF5
using Luna
using Luna.Plotting
using Luna.Output
using Luna.PhysData
using Luna.Antiresonant
using Luna.Grid
using Luna.Interface
using Luna.Stats
using Random
using Printf
using PyPlot

const VIS_λRANGE = (200e-9, 2500e-9)
const VIS_TRANGE = (-100e-15, 100e-15)
const VIS_DB_MIN = -40.0
const EARLY_HIGHRES_FLENGTH = 0.005
const EARLY_HIGHRES_SAVEN = 201

# ============================================================================
# 命令行参数解析
# ============================================================================

"""
    parse_command_line_args() -> Dict{String, Any}

解析命令行参数，支持以下参数：
  --myN, -N VALUE    孤子阶数阈值（浮点数），仅筛选 N > myN 的样本
  --num, -n VALUE    样本数量（正整数），从符合条件的样本中选取的数量
  --extreme-samples   使用 extreme_sample_XXX.h5 文件名模式
  --all              处理所有符合条件的样本，不随机抽样
  --data-dir PATH    输入 HDF5 样本目录
  --output-dir PATH  输出图片目录
  --z-zooms-cm LIST  生成 z=0 到指定 cm 的放大图，例如 50 10 5 1
  --skip-highres-rerun 跳过 0-0.5 cm 高密度重传播
  --help, -h         显示帮助信息

示例：
  julia visualize_hdf5.jl -N 1.5 -n 5
  julia visualize_hdf5.jl --myN 2.0 --num 10
  julia visualize_hdf5.jl -N 0.5 -n 3
  julia visualize_hdf5.jl --extreme-samples -n 5
  julia visualize_hdf5.jl --extreme-samples --all --z-zooms-cm 50 10 5 1 --skip-highres-rerun
"""
function parse_command_line_args()
    args = Dict{String, Any}()
    
    # 默认值
    args["myN"] = 0.0    # 默认阈值0.0，即不过滤
    args["num"] = 10     # 默认选取10个样本
    args["extreme-samples"] = false
    args["all"] = false
    args["data-dir"] = nothing
    args["output-dir"] = nothing
    args["z-zooms-cm"] = Float64[2.0, 10.0]
    args["skip-highres-rerun"] = false
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        
        if arg == "--myN" || arg == "-N"
            if i + 1 <= length(ARGS)
                try
                    myN = parse(Float64, ARGS[i+1])
                    args["myN"] = myN
                    i += 1
                catch
                    error("--myN/-N 需要一个有效的数值参数（浮点数）")
                end
            else
                error("--myN/-N 需要一个数值参数")
            end
        elseif arg == "--num" || arg == "-n"
            if i + 1 <= length(ARGS)
                try
                    num = parse(Int, ARGS[i+1])
                    if num <= 0
                        error("--num/-n 参数必须为正整数（当前值: $num）")
                    end
                    args["num"] = num
                    i += 1
                catch e
                    if isa(e, ErrorException)
                        rethrow(e)
                    else
                        error("--num/-n 需要一个有效的正整数参数")
                    end
                end
            else
                error("--num/-n 需要一个正整数参数")
            end
        elseif arg == "--extreme-samples"
            args["extreme-samples"] = true
        elseif arg == "--all"
            args["all"] = true
        elseif arg == "--data-dir"
            if i + 1 <= length(ARGS)
                args["data-dir"] = ARGS[i+1]
                i += 1
            else
                error("--data-dir 需要一个目录路径参数")
            end
        elseif arg == "--output-dir"
            if i + 1 <= length(ARGS)
                args["output-dir"] = ARGS[i+1]
                i += 1
            else
                error("--output-dir 需要一个目录路径参数")
            end
        elseif arg == "--z-zooms-cm"
            values = Float64[]
            while i + 1 <= length(ARGS) && !startswith(ARGS[i+1], "--")
                try
                    z_end = parse(Float64, ARGS[i+1])
                    if z_end <= 0
                        error("--z-zooms-cm 的每个值都必须大于 0")
                    end
                    push!(values, z_end)
                    i += 1
                catch e
                    if isa(e, ErrorException)
                        rethrow(e)
                    else
                        error("--z-zooms-cm 需要一个或多个数值参数")
                    end
                end
            end
            if isempty(values)
                error("--z-zooms-cm 至少需要一个数值参数")
            end
            args["z-zooms-cm"] = values
        elseif arg == "--skip-highres-rerun"
            args["skip-highres-rerun"] = true
        elseif arg == "--help" || arg == "-h"
            print_help()
            exit(0)
        else
            println("警告: 未知参数 '$arg'，已忽略")
        end
        
        i += 1
    end
    
    return args
end

function print_help()
    println("""
Luna.jl 随机样本可视化工具 v2.0 - 命令行参数说明

用法: julia visualize_hdf5.jl [选项]

选项:
  -N, --myN VALUE    孤子阶数阈值（浮点数），仅筛选 N > myN 的样本进行可视化
                     默认值: 0.0（不过滤，处理所有样本）
  -n, --num VALUE    样本数量（正整数），从符合条件的样本中随机选取的数量
                     默认值: 10
  --extreme-samples  Extreme 样本模式，搜索 extreme_sample_XXX.h5 文件
  --all              处理所有符合条件的样本，不随机抽样
  --data-dir PATH    输入 HDF5 样本目录
  --output-dir PATH  输出图片目录
  --z-zooms-cm LIST  生成 z=0 到指定 cm 的放大图，例如 50 10 5 1
                     默认值: 2 10
  --skip-highres-rerun
                     跳过 0-0.5 cm 高密度重传播图
  -h, --help         显示此帮助信息

示例:
  julia visualize_hdf5.jl -N 1.5 -n 5
    筛选孤子阶数 N > 1.5 的样本，从中随机选取 5 个进行可视化

  julia visualize_hdf5.jl --myN 2.0 --num 10
    筛选孤子阶数 N > 2.0 的样本，从中随机选取 10 个进行可视化

  julia visualize_hdf5.jl -N 0.5 -n 3
    筛选孤子阶数 N > 0.5 的样本，从中随机选取 3 个进行可视化

  julia visualize_hdf5.jl --extreme-samples -n 5
    使用 extreme_sample_XXX.h5 文件名模式，从 extreme search 输出中随机选取 5 个样本

  julia visualize_hdf5.jl --extreme-samples --all --data-dir ../../extreme_search_t650_all_runnable --output-dir extreme_samples_output/all_zzooms --z-zooms-cm 50 10 5 1 --skip-highres-rerun
    处理所有 extreme 样本，并生成 0-50/0-10/0-5/0-1 cm 四组传播图

  julia visualize_hdf5.jl
    使用默认值（myN=0.0, num=10），不过滤，随机选取 10 个样本

注意事项:
  - 孤子阶数 N 从 HDF5 文件的 physics_features/N 数据集中读取
  - 如果样本缺少 N 值，该样本将被跳过
  - 如果符合条件的样本少于 num，将处理所有符合条件的样本
""")
end

# ============================================================================
# 工具函数
# ============================================================================

function check_and_create_dir(dirpath)
    if !isdir(dirpath)
        mkdir(dirpath)
        println("创建目录: $dirpath")
    end
end

function safe_savefig(filepath; dpi=150)
    dir = dirname(filepath)
    if !isempty(dir) && dir != "." && !isdir(dir)
        mkpath(dir)
    end
    PyPlot.savefig(filepath, dpi=dpi, bbox_inches="tight")
end

function read_wavelength_axis_nm(file, freq_count)
    if !haskey(file, "grid")
        return Float64[]
    end
    grid = file["grid"]

    for key in ("λ", "lambda")
        if haskey(grid, key)
            λ = vec(read(grid[key]))
            if length(λ) == freq_count
                return Float64.(λ .* 1e9)
            end
        end
    end

    for key in ("ω", "omega", "w")
        if haskey(grid, key)
            ω = vec(read(grid[key]))
            if length(ω) == freq_count
                return Float64.(2 * pi .* PhysData.c ./ ω .* 1e9)
            end
        end
    end

    if haskey(grid, "sidx") && haskey(grid, "ωo")
        sidx = vec(read(grid["sidx"]))
        ωo = vec(read(grid["ωo"]))
        if !isempty(sidx)
            idx = minimum(sidx) == 0 ? sidx .+ 1 : sidx
            if maximum(idx) <= length(ωo) && minimum(idx) >= 1
                ω = ωo[idx]
                if length(ω) == freq_count
                    return Float64.(2 * pi .* PhysData.c ./ ω .* 1e9)
                end
            end
        end
    end

    return Float64[]
end

function read_final_spectrum_and_wavelength(filepath)
    h5open(filepath, "r") do file
        if !haskey(file, "Eω") || !haskey(file, "z")
            error("HDF5 file must contain Eω and z datasets")
        end

        Eω = read(file["Eω"])
        z = vec(read(file["z"]))
        ndims(Eω) == 2 || error("Eω must be a 2D array")
        isempty(z) && error("z dataset is empty")

        final_spectrum = size(Eω, 2) == length(z) ? vec(Eω[:, end]) : vec(Eω[end, :])
        λ_nm = read_wavelength_axis_nm(file, length(final_spectrum))
        length(λ_nm) == length(final_spectrum) || error("Could not reconstruct wavelength axis")
        return λ_nm, final_spectrum
    end
end

function plot_final_spectrum_db(h5_filepath, fig_filepath; λrange=VIS_λRANGE, dBmin=VIS_DB_MIN)
    λ_nm, final_spectrum = read_final_spectrum_and_wavelength(h5_filepath)
    power = abs2.(final_spectrum)

    λ_min_nm = λrange[1] * 1e9
    λ_max_nm = λrange[2] * 1e9
    valid = isfinite.(λ_nm) .& isfinite.(power) .& (λ_min_nm .<= λ_nm) .& (λ_nm .<= λ_max_nm)
    if count(valid) < 2
        error("Not enough finite spectral points in requested wavelength range")
    end

    λ_plot = λ_nm[valid]
    power_plot = power[valid]
    perm = sortperm(λ_plot)
    λ_plot = λ_plot[perm]
    power_plot = power_plot[perm]
    pmax = maximum(power_plot)
    pmax > 0.0 || error("Final spectrum has zero peak power")

    sed_db = 10 .* log10.(max.(power_plot ./ pmax, eps(Float64)))
    sed_db = max.(sed_db, dBmin)

    PyPlot.figure(figsize=(12, 6))
    PyPlot.plot(λ_plot, sed_db, color="#1f77b4", linewidth=1.3)
    PyPlot.xlabel("Wavelength (nm)", fontsize=12)
    PyPlot.ylabel("SED (dB, relative to final peak)", fontsize=12)
    PyPlot.title("Final Spectrum (relative dB)", fontsize=13)
    PyPlot.grid(true, alpha=0.3)
    PyPlot.xlim(λ_min_nm, λ_max_nm)
    PyPlot.ylim(dBmin, 1.0)
    PyPlot.tight_layout()
    safe_savefig(fig_filepath)
    PyPlot.close()
    return fig_filepath
end

function apply_distance_zoom_to_current_figure(zrange_cm)
    fig = PyPlot.gcf()
    zoomed_axes = 0
    for ax in fig.axes
        ylabel = lowercase(String(ax.get_ylabel()))
        if occursin("distance", ylabel) || occursin("cm", ylabel)
            ax.set_ylim(zrange_cm[1], zrange_cm[2])
            zoomed_axes += 1
        end
    end
    if zoomed_axes == 0
        @warn "No propagation distance axes found for zoom; applying z range to current axis only."
        PyPlot.gca().set_ylim(zrange_cm[1], zrange_cm[2])
    end
    return fig
end

function plot_propagation_zoom(output, filepath; λrange=VIS_λRANGE,
                               zrange_cm=(0.0, 2.0), dBmin=VIS_DB_MIN)
    Plotting.prop_2D(output, :λ, dBmin=dBmin, λrange=λrange, trange=VIS_TRANGE)
    fig = apply_distance_zoom_to_current_figure(zrange_cm)
    fig.suptitle("Propagation Zoom: z = $(zrange_cm[1])-$(zrange_cm[2]) cm")
    safe_savefig(filepath)
    PyPlot.close()
    return filepath
end

function plot_spectral_propagation_zoom(output, filepath; λrange=VIS_λRANGE,
                                        zrange_cm=(0.0, 2.0), dBmin=VIS_DB_MIN)
    z_cm = output["z"] .* 1e2
    specx, Iω = Plotting.getIω(output, :λ, specrange=λrange)
    specx_nm = specx .* 1e9

    Iω_plot = ndims(Iω) == 3 ? dropdims(sum(Iω, dims=2), dims=2) : Iω
    Imax = maximum(Iω_plot)
    Imax > 0.0 || error("Spectral propagation has zero peak intensity")
    sed_db = 10 .* log10.(max.(Iω_plot ./ Imax, eps(Float64)))
    sed_db = max.(sed_db, dBmin)

    PyPlot.figure(figsize=(7, 5))
    im = PyPlot.pcolormesh(specx_nm, z_cm, transpose(sed_db), shading="auto")
    im.set_clim(dBmin, 0)
    cb = PyPlot.colorbar(im)
    cb.set_label("SED (dB)")
    PyPlot.xlabel("Wavelength (nm)")
    PyPlot.ylabel("Distance (cm)")
    PyPlot.xlim(λrange[1] * 1e9, λrange[2] * 1e9)
    PyPlot.ylim(zrange_cm[1], zrange_cm[2])
    PyPlot.title("Spectral Evolution Zoom: z = $(zrange_cm[1])-$(zrange_cm[2]) cm")
    PyPlot.tight_layout()
    safe_savefig(filepath)
    PyPlot.close()
    return filepath
end

function z_zoom_suffix_cm(z_end_cm)
    if isapprox(z_end_cm, round(z_end_cm); atol=1e-9)
        return string(Int(round(z_end_cm)), "cm")
    end
    label = replace(@sprintf("%.3g", z_end_cm), "." => "p")
    return string(label, "cm")
end

function numeric_h5_value(value)
    if value isa Number
        return Float64(real(value))
    elseif value isa AbstractArray
        isempty(value) && return nothing
        return numeric_h5_value(value[begin])
    elseif value isa AbstractString
        return tryparse(Float64, strip(value))
    elseif value isa Vector{UInt8}
        return tryparse(Float64, strip(String(value)))
    else
        return tryparse(Float64, strip(string(value)))
    end
end

function read_group_numeric(group, key)
    if !haskey(group, key)
        return nothing
    end
    try
        return numeric_h5_value(read(group[key]))
    catch
        return nothing
    end
end

function read_highres_params(file_path)
    params = Dict{String, Float64}()
    h5open(file_path, "r") do fid
        if haskey(fid, "input")
            inp = fid["input"]
            for (key, outkey) in (
                ("energy", "energy_uJ"),
                ("tau", "tau_fs"),
                ("pressure", "pressure_bar"),
                ("diameter", "diameter_um"),
                ("wallthickness", "wallthickness_um"),
                ("chirp", "chirp"),
            )
                value = read_group_numeric(inp, key)
                if value !== nothing
                    params[outkey] = value
                end
            end
            if haskey(params, "wallthickness_um") && params["wallthickness_um"] < 1e-3
                params["wallthickness_um"] *= 1e6
            end
        end

        if haskey(fid, "prop_capillary_args")
            pca = fid["prop_capillary_args"]
            energy = read_group_numeric(pca, "energy")
            τfwhm = read_group_numeric(pca, "τfwhm")
            pressure = read_group_numeric(pca, "pressure")
            radius = read_group_numeric(pca, "radius")
            if energy !== nothing && !haskey(params, "energy_uJ")
                params["energy_uJ"] = energy * 1e6
            end
            if τfwhm !== nothing && !haskey(params, "tau_fs")
                params["tau_fs"] = τfwhm * 1e15
            end
            if pressure !== nothing && !haskey(params, "pressure_bar")
                params["pressure_bar"] = pressure
            end
            if radius !== nothing && !haskey(params, "diameter_um")
                params["diameter_um"] = radius * 2e6
            end
        end

        if haskey(fid, "physics_features") && !haskey(params, "wallthickness_um")
            pf = fid["physics_features"]
            wall = read_group_numeric(pf, "wallthickness")
            if wall !== nothing
                params["wallthickness_um"] = wall < 1e-3 ? wall * 1e6 : wall
            end
        end
    end

    required = ("energy_uJ", "tau_fs", "pressure_bar", "diameter_um", "wallthickness_um")
    for key in required
        if !haskey(params, key)
            return nothing
        end
    end
    if !haskey(params, "chirp")
        params["chirp"] = 0.0
    end
    return params
end

function prop_antiresonant_highres(radius, gas, pressure, wallthickness_m;
                                   λlims=VIS_λRANGE, trange=500e-15,
                                   λ0=1030e-9, τfwhm, energy,
                                   ϕ=Float64[], pulseshape=:sech,
                                   saveN=EARLY_HIGHRES_SAVEN)
    grid = Grid.RealGrid(EARLY_HIGHRES_FLENGTH, λ0, λlims, trange)
    mode = Antiresonant.ZeisbergerMode(radius, gas, pressure;
                                       wallthickness=wallthickness_m,
                                       model=:full, loss=true)
    density = z -> PhysData.density(gas, pressure, PhysData.roomtemp)
    resp = Interface.makeresponse(grid, gas, false, true, true,
                                  true, false, true, true,
                                  Dict{Symbol, Any}(), 0.0, PhysData.roomtemp)
    inputs = Interface.makeinputs(mode, λ0, nothing, τfwhm, nothing, ϕ, nothing, energy,
                                  pulseshape, :linear, nothing)
    inputs, noise_field = Interface.makenoise(grid, mode, inputs, true, Random.GLOBAL_RNG)
    linop, Eω, transform, FT = Interface.setup(grid, mode, density, resp, inputs,
                                               false, 1e-3, Val(true);
                                               noise_field)
    stats = Stats.default(grid, Eω, mode, linop, transform; gas=gas)
    output = Interface.makeoutput(grid, saveN, stats, nothing, nothing, nothing, nothing)
    Luna.run(Eω, grid, linop, transform, FT, output; status_period=30)
    return output
end

function run_highres_from_params(params)
    energy = params["energy_uJ"] * 1e-6
    τfwhm = params["tau_fs"] * 1e-15
    pressure = params["pressure_bar"]
    radius = params["diameter_um"] / 2 * 1e-6
    wallthickness_m = params["wallthickness_um"] * 1e-6
    τ0 = τfwhm / (2 * log(1 + sqrt(2)))
    gdd = get(params, "chirp", 0.0) * τ0^2
    ϕ_vec = Float64[0.0, 0.0, gdd]
    return prop_antiresonant_highres(radius, :Ar, pressure, wallthickness_m;
                                     λ0=1030e-9, τfwhm=τfwhm, energy=energy,
                                     ϕ=ϕ_vec, λlims=VIS_λRANGE,
                                     trange=500e-15, saveN=EARLY_HIGHRES_SAVEN)
end

function find_sample_files(data_dir, pattern=r"^sample_(\d{6})\.h5$")
    files = []
    if isdir(data_dir)
        for f in readdir(data_dir)
            m = match(pattern, f)
            if m !== nothing
                push!(files, joinpath(data_dir, f))
            end
        end
    end
    return sort(files)
end

"""
    get_soliton_order(file_path) -> Union{Float64, Nothing}

从 HDF5 文件中读取孤子阶数 N 值。
返回 N 值，如果读取失败或不存在则返回 nothing。
"""
function get_soliton_order(file_path)
    try
        h5open(file_path, "r") do fid
            if haskey(fid, "physics_features")
                pf = fid["physics_features"]
                if haskey(pf, "N")
                    value = read(pf, "N")
                    # 处理可能为复数的情况
                    if typeof(value) <: Complex
                        return real(value)
                    else
                        return Float64(value)
                    end
                end
            end
        end
    catch e
        println("  警告: 读取 $file_path 的 N 值失败: $e")
    end
    return nothing
end

"""
    filter_samples_by_N(files, myN) -> Vector{String}

根据孤子阶数阈值 myN 筛选样本文件。
仅保留 N 值严格大于 myN 的样本。
"""
function filter_samples_by_N(files, myN)
    filtered = String[]
    skipped = 0
    
    for file_path in files
        N = get_soliton_order(file_path)
        if N === nothing
            skipped += 1
            continue
        end
        if N > myN
            push!(filtered, file_path)
        end
    end
    
    if skipped > 0
        println("  注意: $skipped 个样本因缺少 N 值被跳过")
    end
    
    return filtered
end

function print_sample_parameters(file_path)
    println("\n" * "="^60)
    println("样本参数信息: $(basename(file_path))")
    println("="^60)
    
    try
        h5open(file_path, "r") do fid
            if haskey(fid, "grid")
                grid = fid["grid"]
                if haskey(grid, "λ")
                    λ = read(grid, "λ")
                    @printf("  波长范围: %.2f nm - %.2f nm\n", λ[1]*1e9, λ[end]*1e9)
                    @printf("  波长点数: %d\n", length(λ))
                end
                if haskey(grid, "t")
                    t = read(grid, "t")
                    @printf("  时间范围: %.2f fs - %.2f fs\n", t[1]*1e15, t[end]*1e15)
                    @printf("  时间点数: %d\n", length(t))
                end
            end
            
            z = read(fid, "z")
            @printf("  传播距离: %.4f m (%.2f cm)\n", z[end], z[end]*100)
            @printf("  采样点数: %d\n", length(z))
            
            if haskey(fid, "Eω")
                Eω = read(fid, "Eω")
                @printf("  Eω 数据形状: (%d, %d)\n", size(Eω)...)
            end
            
            if haskey(fid, "physics_features")
                pf = fid["physics_features"]
                println("\n  物理特征:")
                for key in ["beta2", "gamma", "N", "L0", "gamma_K", "P_ratio", "Aeff", "neff"]
                    if haskey(pf, key)
                        value = read(pf, key)
                        if typeof(value) <: Complex
                            @printf("    %-10s: %.4e + %.4e im\n", key, real(value), imag(value))
                        else
                            @printf("    %-10s: %.4e\n", key, value)
                        end
                    else
                        @printf("    %-10s: 未定义\n", key)
                    end
                end
            end
            
            if haskey(fid, "energy")
                energy = read(fid, "energy")
                @printf("\n  初始能量: %.4e J (%.4f μJ)\n", energy[1], energy[1]*1e6)
                @printf("  最终能量: %.4e J (%.4f μJ)\n", energy[end], energy[end]*1e6)
                @printf("  能量损失: %.2f%%\n", (1 - energy[end]/energy[1])*100)
            end
        end
    catch e
        println("  读取参数失败: $e")
    end
    
    println("="^60)
end

function visualize_file(file_path, output_dir; z_zoom_ends_cm=Float64[2.0, 10.0],
                        skip_highres_rerun=false)
    sample_name = replace(basename(file_path), ".h5" => "")
    prefix = "$(sample_name)_"
    
    try
        output = Output.HDF5Output(file_path)
        z_max = maximum(output["z"])
        
        println("\n  绘制光谱图...")
        try
            fig = Plotting.spec_1D(output, z_max, :λ, λrange=VIS_λRANGE)
            filename = joinpath(output_dir, "$(prefix)spectrum_plot.png")
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            println("    ✓ 已保存: $(basename(filename))")
            PyPlot.close()
        catch e
            println("    ✗ 绘制光谱图失败: $e")
        end

        println("\n  绘制最终光谱 dB 图...")
        try
            filename = joinpath(output_dir, "$(prefix)spectrum_plot_db.png")
            plot_final_spectrum_db(file_path, filename; λrange=VIS_λRANGE, dBmin=VIS_DB_MIN)
            println("    ✓ 已保存: $(basename(filename))")
        catch e
            println("    ✗ 绘制最终光谱 dB 图失败: $e")
        end
        
        println("\n  绘制频域演化图...")
        try
            fig = Plotting.prop_2D(output, :λ, λrange=VIS_λRANGE,
                                   trange=VIS_TRANGE, dBmin=VIS_DB_MIN)
            filename = joinpath(output_dir, "$(prefix)spectral_evolution.png")
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            println("    ✓ 已保存: $(basename(filename))")
            PyPlot.close()
        catch e
            println("    ✗ 绘制频域演化图失败: $e")
        end

        for z_end_cm in z_zoom_ends_cm
            suffix = z_zoom_suffix_cm(z_end_cm)

            println("\n  绘制传播演化局部放大图 (z=0-$z_end_cm cm)...")
            try
                filename = joinpath(output_dir, "$(prefix)spectral_evolution_z0_$(suffix).png")
                plot_propagation_zoom(output, filename; λrange=VIS_λRANGE,
                                      zrange_cm=(0.0, z_end_cm), dBmin=VIS_DB_MIN)
                println("    ✓ 已保存: $(basename(filename))")
            catch e
                println("    ✗ 绘制传播演化 z=0-$z_end_cm cm 局部放大图失败: $e")
            end

            println("\n  绘制单独频域传播局部放大图 (z=0-$z_end_cm cm)...")
            try
                filename = joinpath(output_dir, "$(prefix)spectral_propagation_z0_$(suffix).png")
                plot_spectral_propagation_zoom(output, filename; λrange=VIS_λRANGE,
                                               zrange_cm=(0.0, z_end_cm), dBmin=VIS_DB_MIN)
                println("    ✓ 已保存: $(basename(filename))")
            catch e
                println("    ✗ 绘制单独频域传播 z=0-$z_end_cm cm 局部放大图失败: $e")
            end
        end

        if !skip_highres_rerun
            println("\n  绘制高密度早期传播图 (z=0-0.5 cm, 201 points)...")
            try
                params = read_highres_params(file_path)
                if params === nothing
                    println("    ⚠ 缺少重传播所需输入参数，跳过高密度早期传播图")
                else
                    highres_output = run_highres_from_params(params)
                    filename = joinpath(output_dir, "$(prefix)spectral_evolution_z0_0p5cm_highres.png")
                    plot_propagation_zoom(highres_output, filename; λrange=VIS_λRANGE,
                                          zrange_cm=(0.0, 0.5), dBmin=VIS_DB_MIN)
                    println("    ✓ 已保存: $(basename(filename))")

                    filename = joinpath(output_dir, "$(prefix)spectral_propagation_z0_0p5cm_highres.png")
                    plot_spectral_propagation_zoom(highres_output, filename; λrange=VIS_λRANGE,
                                                   zrange_cm=(0.0, 0.5), dBmin=VIS_DB_MIN)
                    println("    ✓ 已保存: $(basename(filename))")
                end
            catch e
                println("    ✗ 绘制高密度早期传播图失败: $e")
            end
        end
        
        println("\n  绘制统计信息图...")
        try
            fig = Plotting.stats(output)
            filename = joinpath(output_dir, "$(prefix)stats_plot.png")
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            println("    ✓ 已保存: $(basename(filename))")
            PyPlot.close()
        catch e
            println("    ✗ 绘制统计信息图失败: $e")
        end
        
        return true
    catch e
        println("\n  ✗ 处理文件失败: $e")
        return false
    end
end

# ============================================================================
# 主函数
# ============================================================================

function main()
    # 解析命令行参数
    cli_args = parse_command_line_args()
    myN = cli_args["myN"]
    num = cli_args["num"]
    extreme_samples = cli_args["extreme-samples"]
    process_all = cli_args["all"]
    z_zoom_ends_cm = cli_args["z-zooms-cm"]
    skip_highres_rerun = cli_args["skip-highres-rerun"]
    
    data_dir = cli_args["data-dir"] === nothing ? joinpath(dirname(dirname(@__DIR__)), "extreme_search_t650_all_runnable") : cli_args["data-dir"]
    output_dir = cli_args["output-dir"] === nothing ? joinpath(@__DIR__, "extreme_samples_output") : cli_args["output-dir"]
    sample_pattern = extreme_samples ? r"^extreme_sample_(\d{3})\.h5$" : r"^sample_(\d{6})\.h5$"
    
    println("="^70)
    println("Luna.jl 随机样本可视化工具 v2.0")
    println("="^70)
    println("数据目录: $data_dir")
    println("输出目录: $output_dir")
    println("样本模式: $(extreme_samples ? "extreme_sample_XXX.h5" : "sample_XXXXXX.h5")")
    println("孤子阶数阈值 (myN): $myN")
    println("目标样本数量 (num): $num")
    println("处理全部样本 (--all): $process_all")
    println("z 放大图终点 (cm): $(join(z_zoom_ends_cm, ", "))")
    println("跳过高密度重传播: $skip_highres_rerun")
    println()
    
    all_files = find_sample_files(data_dir, sample_pattern)
    @printf("找到 %d 个符合格式的样本文件\n", length(all_files))
    
    if isempty(all_files)
        println("错误: 未找到任何符合格式的样本文件")
        return
    end
    
    # 根据孤子阶数 N 筛选样本
    if myN > 0.0
        println("\n正在筛选 N > $myN 的样本...")
        filtered_files = filter_samples_by_N(all_files, myN)
        @printf("筛选后剩余 %d 个样本 (原 %d 个)\n", length(filtered_files), length(all_files))
        
        if isempty(filtered_files)
            println("错误: 没有样本的孤子阶数 N 大于 $myN")
            return
        end
        
        selected_pool = filtered_files
    else
        println("\n孤子阶数阈值 myN=0.0，不过滤，处理所有样本")
        selected_pool = all_files
    end
    
    # 从符合条件的样本中选取样本
    if process_all
        selected_files = selected_pool
        num_samples = length(selected_files)
    else
        num_samples = min(num, length(selected_pool))
        Random.seed!(rand(UInt32))
        selected_files = shuffle(selected_pool)[1:num_samples]
    end
    
    check_and_create_dir(output_dir)
    
    println(process_all ? "\n选中的全部样本:" : "\n随机选中的样本:")
    for (i, fp) in enumerate(selected_files)
        N_val = get_soliton_order(fp)
        N_str = N_val !== nothing ? @sprintf(" (N=%.3f)", N_val) : " (N=unknown)"
        println("  [$i] $(basename(fp))$N_str")
    end
    
    success_count = 0
    for (index, file_path) in enumerate(selected_files)
        println("\n" * "="^70)
        println("处理样本 [$index/$num_samples]: $(basename(file_path))")
        println("="^70)
        
        print_sample_parameters(file_path)
        success = visualize_file(file_path, output_dir;
                                 z_zoom_ends_cm=z_zoom_ends_cm,
                                 skip_highres_rerun=skip_highres_rerun)
        if success
            success_count += 1
        end
    end
    
    println("\n" * "="^70)
    println("可视化任务完成!")
    println("="^70)
    @printf("成功处理: %d/%d 个样本\n", success_count, num_samples)
    
    if isdir(output_dir)
        files = readdir(output_dir)
        @printf("生成图像文件数: %d\n", length(files))
        println("\n输出目录: $output_dir")
    end
end

main()
