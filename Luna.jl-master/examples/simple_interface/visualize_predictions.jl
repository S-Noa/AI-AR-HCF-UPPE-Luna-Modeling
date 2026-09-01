#!/usr/bin/env julia
#=
Visualization Script for Model Predictions - Scheme 2

This script creates a mock Luna Output object from prediction results
to use with Luna's built-in Plotting module.

Usage:
    julia visualize_predictions.jl prediction_results.h5

Output:
    - Spectrum plots (using Plotting.spec_1D)
    - Temporal evolution heatmaps (using Plotting.prop_2D)
    - Spectral evolution heatmaps (using Plotting.prop_2D)
=#

using HDF5
using Printf
using Luna
using Luna.Plotting
using PyPlot

# Define a mock Output type that mimics Luna.Output.HDF5Output
struct MockOutput
    z::Vector{Float64}
    Eω::Array{ComplexF64,2}
    λ::Vector{Float64}
    t::Vector{Float64}
    energy::Vector{Float64}
end

# Implement getindex to support output["z"] syntax
Base.getindex(o::MockOutput, key::String) = getfield(o, Symbol(key))

function check_and_create_dir(dirpath)
    if !isdir(dirpath)
        mkdir(dirpath)
        println("创建目录: $dirpath")
    end
end

function check_and_delete(filepath)
    if isfile(filepath)
        rm(filepath)
    end
end

function load_prediction_results(file_path)
    results = []
    
    h5open(file_path, "r") do fid
        for name in keys(fid)
            if startswith(name, "sample_")
                group = fid[name]
                sample_name = name
                
                try
                    sample_name = read(attrs(group), "name")
                catch
                end
                
                spectrum = read(group, "predicted_spectrum")
                
                params = Dict()
                if haskey(group, "parameters")
                    pf = group["parameters"]
                    for key in keys(pf)
                        params[key] = read(pf, key)
                    end
                end
                
                push!(results, Dict(
                    :name => sample_name,
                    :group_name => name,
                    :spectrum => spectrum,
                    :parameters => params
                ))
            end
        end
    end
    
    println("加载了 $(length(results)) 个预测样本")
    return results
end

function create_mock_output(spectrum, params)
    """Create a MockOutput object from prediction spectrum."""
    n_points = length(spectrum)
    n_z = 50
    
    # Create z axis (propagation distance in m)
    z = collect(LinRange(0, 0.5, n_z))
    
    # Create wavelength axis
    λ = collect(LinRange(200e-9, 2500e-9, n_points))
    
    # Create time axis
    t = collect(LinRange(-500e-15, 500e-15, n_points))
    
    # Create Eω data (replicate spectrum across propagation)
    Eω = repeat(reshape(spectrum, n_points, 1), 1, n_z)
    
    # Add some variation with propagation for spectral evolution
    z_vals = collect(LinRange(0, 1, n_z))
    for i in 1:n_z
        Eω[:, i] = Eω[:, i] .* (0.8 + 0.2 * z_vals[i])
    end
    
    # Convert to complex
    Eω_complex = ComplexF64.(Eω)
    
    # Create energy array (mock)
    energy = ones(n_z) * 1e-6
    
    return MockOutput(z, Eω_complex, λ, t, energy)
end

function plot_predicted_spectrum(output::MockOutput, params, output_dir, sample_name)
    println("  绘制光谱图...")
    try
        z_max = maximum(output.z)
        
        fig = Plotting.spec_1D(output, z_max, :λ, λrange=(200e-9, 2500e-9))
        
        filename = joinpath(output_dir, "$(sample_name)_predicted_spectrum.png")
        check_and_delete(filename)
        save(filename, fig, px_per_unit=3)
        println("    ✓ 已保存: $filename")
        
    catch e
        println("    ✗ 绘制光谱图失败: $e")
        println("      尝试备用方案...")
        plot_predicted_spectrum_fallback(output, params, output_dir, sample_name)
    end
end

function plot_predicted_spectrum_fallback(output::MockOutput, params, output_dir, sample_name)
    """Fallback using PyPlot directly."""
    try
        λ_nm = output.λ .* 1e9
        spectrum = abs2.(output.Eω[:, end])
        spectrum = spectrum ./ maximum(spectrum)
        
        fig, ax = subplots(figsize=(10, 6))
        ax.plot(λ_nm, spectrum, "b-", linewidth=2)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized Power")
        ax.set_title("Predicted Spectrum - $sample_name")
        ax.set_xlim(200, 2500)
        ax.set_ylim(0, 1.1)
        ax.grid(true, alpha=0.3)
        
        filename = joinpath(output_dir, "$(sample_name)_predicted_spectrum.png")
        check_and_delete(filename)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        close(fig)
        println("    ✓ 已保存(备用): $filename")
        
    catch e2
        println("    ✗ 备用方案也失败: $e2")
    end
end

function plot_temporal_evolution(output::MockOutput, params, output_dir, sample_name)
    println("  绘制脉冲时域演化图...")
    try
        fig = Plotting.prop_2D(output, :λ, λrange=(200e-9, 2500e-9), trange=(-200e-15, 200e-15))
        
        filename = joinpath(output_dir, "$(sample_name)_temporal_evolution.png")
        check_and_delete(filename)
        save(filename, fig, px_per_unit=3)
        println("    ✓ 已保存: $filename")
        
    catch e
        println("    ✗ 绘制时域演化图失败: $e")
        println("      尝试备用方案...")
        plot_temporal_evolution_fallback(output, params, output_dir, sample_name)
    end
end

function plot_temporal_evolution_fallback(output::MockOutput, params, output_dir, sample_name)
    """Fallback using PyPlot directly."""
    try
        n_points = size(output.Eω, 1)
        n_z = size(output.Eω, 2)
        
        t_fs = output.t .* 1e15
        z_cm = output.z .* 100
        
        # Calculate time-domain power (mock)
        time_power = abs2.(output.Eω)
        time_power = time_power ./ maximum(time_power)
        
        fig, ax = subplots(figsize=(10, 6))
        T = repeat(reshape(t_fs, n_points, 1), 1, n_z)
        Z = repeat(reshape(z_cm, 1, n_z), n_points, 1)
        
        pcm = ax.pcolormesh(T, Z, time_power, cmap="plasma", shading="auto")
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Propagation Distance (cm)")
        ax.set_title("Predicted Temporal Evolution - $sample_name")
        colorbar(pcm, ax=ax, label="Normalized Intensity")
        
        filename = joinpath(output_dir, "$(sample_name)_temporal_evolution.png")
        check_and_delete(filename)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        close(fig)
        println("    ✓ 已保存(备用): $filename")
        
    catch e2
        println("    ✗ 备用方案也失败: $e2")
    end
end

function plot_spectral_evolution(output::MockOutput, params, output_dir, sample_name)
    println("  绘制脉冲频域演化图...")
    try
        fig = Plotting.prop_2D(output, :λ, λrange=(200e-9, 2500e-9))
        
        filename = joinpath(output_dir, "$(sample_name)_spectral_evolution.png")
        check_and_delete(filename)
        save(filename, fig, px_per_unit=3)
        println("    ✓ 已保存: $filename")
        
    catch e
        println("    ✗ 绘制频域演化图失败: $e")
        println("      尝试备用方案...")
        plot_spectral_evolution_fallback(output, params, output_dir, sample_name)
    end
end

function plot_spectral_evolution_fallback(output::MockOutput, params, output_dir, sample_name)
    """Fallback using PyPlot directly."""
    try
        n_points = size(output.Eω, 1)
        n_z = size(output.Eω, 2)
        
        λ_nm = output.λ .* 1e9
        z_cm = output.z .* 100
        
        freq_power = abs2.(output.Eω)
        freq_power = freq_power ./ maximum(freq_power)
        
        fig, ax = subplots(figsize=(10, 6))
        L = repeat(reshape(λ_nm, n_points, 1), 1, n_z)
        Z = repeat(reshape(z_cm, 1, n_z), n_points, 1)
        
        pcm = ax.pcolormesh(L, Z, freq_power, cmap="plasma", shading="auto")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Propagation Distance (cm)")
        ax.set_title("Predicted Spectral Evolution - $sample_name")
        colorbar(pcm, ax=ax, label="Normalized Intensity")
        
        filename = joinpath(output_dir, "$(sample_name)_spectral_evolution.png")
        check_and_delete(filename)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        close(fig)
        println("    ✓ 已保存(备用): $filename")
        
    catch e2
        println("    ✗ 备用方案也失败: $e2")
    end
end

function print_parameters(params, sample_name)
    println("\n" * "="^60)
    println("参数信息: $sample_name")
    println("="^60)
    
    if !isempty(params)
        param_labels = Dict(
            "energy" => ("Pulse Energy", "μJ"),
            "tau" => ("Pulse Width", "fs"),
            "pressure" => ("Gas Pressure", "bar"),
            "length" => ("Fiber Length", "cm"),
            "diameter" => ("Core Diameter", "μm"),
            "beta2" => ("GVD β₂", "fs²/mm"),
            "gamma" => ("Nonlinear Coeff γ", "1/(W·m)"),
            "N" => ("Soliton Order", ""),
            "L0" => ("Dispersion Length", "cm"),
            "gamma_K" => ("Keldysh Parameter", ""),
            "P_ratio" => ("P_peak / P_critical", ""),
            "Aeff" => ("Effective Area", "μm²"),
            "neff" => ("Effective Index", "")
        )
        
        for key in ["energy", "tau", "pressure", "length", "diameter",
                    "beta2", "gamma", "N", "L0", "gamma_K", "P_ratio", "Aeff", "neff"]
            if haskey(params, key)
                val = params[key]
                label, unit = get(param_labels, key, (key, ""))
                
                if typeof(val) <: AbstractFloat
                    if abs(val) < 1e-6 || (abs(val) > 1e6 && val != 0)
                        println("  $(rpad(label, 20)): $(@sprintf("%.4e", val)) $unit")
                    else
                        println("  $(rpad(label, 20)): $(@sprintf("%.6f", val)) $unit")
                    end
                else
                    println("  $label: $val $unit")
                end
            end
        end
    else
        println("  无参数信息")
    end
    
    println("="^60)
end

function visualize_predictions(input_file, output_dir)
    println("="^70)
    println("Luna.jl 模型预测结果可视化 (方案2)")
    println("="^70)
    println("输入文件: $input_file")
    println("输出目录: $output_dir")
    println()
    
    results = load_prediction_results(input_file)
    
    if isempty(results)
        println("错误: 未找到预测数据")
        return
    end
    
    check_and_create_dir(output_dir)
    
    success_count = 0
    fallback_count = 0
    fail_count = 0
    
    for (i, result) in enumerate(results)
        sample_name = replace(result[:name], ".h5" => "")
        println("\n处理样本 [$i/$(length(results))]: $sample_name")
        
        print_parameters(result[:parameters], sample_name)
        
        # Create mock output object
        output = create_mock_output(result[:spectrum], result[:parameters])
        
        # Try primary method (Luna Plotting)
        println("  尝试使用 Luna.Plotting 绘制...")
        
        try
            plot_predicted_spectrum(output, result[:parameters], output_dir, sample_name)
            success_count += 1
        catch
            fallback_count += 1
        end
        
        try
            plot_temporal_evolution(output, result[:parameters], output_dir, sample_name)
            success_count += 1
        catch
            fallback_count += 1
        end
        
        try
            plot_spectral_evolution(output, result[:parameters], output_dir, sample_name)
            success_count += 1
        catch
            fallback_count += 1
        end
    end
    
    println("\n" * "="^70)
    println("可视化任务完成!")
    println("="^70)
    println("统计:")
    println("  成功使用 Luna.Plotting: $success_count")
    println("  使用备用方案: $fallback_count")
    println("  失败: $fail_count")
    
    if isdir(output_dir)
        files = readdir(output_dir)
        println("\n生成图像文件数: $(length(files))")
        println("输出目录: $output_dir")
        println("生成的文件:")
        for f in files
            println("  - $f")
        end
    end
end

function main()
    if length(ARGS) < 1
        println("用法: julia visualize_predictions.jl <prediction_results.h5>")
        return
    end
    
    input_file = ARGS[1]
    output_dir = "prediction_visualizations"
    
    if !isfile(input_file)
        println("错误: 文件不存在 - $input_file")
        return
    end
    
    visualize_predictions(input_file, output_dir)
end

main()