#!/usr/bin/env julia
"""
Anti-resonant Hollow-Core Fibre Single Simulation Script

基于 data_generation.jl 的反谐振光纤模型实现，提供单次光传输模拟功能。

Usage:
    julia anti_resonant_simulation.jl [options]

Options:
    -t, --wall-thickness VALUE  管壁厚度 (μm, 默认: 0.65)
    -d, --diameter VALUE        纤芯直径 (μm, 默认: 30)
    -e, --energy VALUE          脉冲能量 (μJ, 默认: 1.5)
    -τ, --tau VALUE             脉冲宽度 (fs, 默认: 15)
    -p, --pressure VALUE        气压 (bar, 默认: 10)
    -c, --chirp VALUE           啁啾参数 (默认: 0)
    -o, --output FILE           输出文件路径 (默认: anti_resonant_simulation.h5)
    -h, --help                  显示帮助信息

Example:
    julia anti_resonant_simulation.jl -t 0.65 -d 30 -e 1.5 -p 10
    julia anti_resonant_simulation.jl --wall-thickness 0.36 --energy 1.0 --pressure 30
"""

using Luna
using Luna.PhysData
using Luna.Modes
using Luna.Grid
using Luna.Antiresonant
using Dates
using HDF5
using Random
using PyPlot
using Printf
import FFTW
import Luna.Capillary

# ============================================================================
# 通用工具函数
# ============================================================================

# Keep this consistent with data_generation.jl. It is a numerical safety limit,
# not a marker for "strong ionization" samples.
const EXTREME_DATASET_E_LIMIT = 8.625348297553e10
const EXTREME_STRONG_E_LIMIT_RATIO = 0.5
const EXTREME_STRONG_PPT_RATIO = 0.10
const EXTREME_STRONG_GAMMA_K = 2.0

function manual_diff(y, x)
    n = length(y)
    d = similar(y, n - 1)
    for i in 1:(n - 1)
        d[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    end
    return d
end

function barrier_suppression_field(ionpot, Z)
    Ip_au = ionpot / PhysData.au_energy
    ns = Z / sqrt(2 * Ip_au)
    return Z^3 / (16 * ns^4) * PhysData.au_Efield
end

function find_resonance_wavelengths(wallthickness_um, ncl=1.45, nco=1.0;
                                     λ_min_nm=50.0, λ_max_nm=3000.0, max_order=20)
    wt_m = wallthickness_um * 1e-6
    resonance_orders = collect(1:max_order)
    λ_res_nm = 2.0 .* wt_m .* sqrt(ncl^2 - nco^2) ./ resonance_orders .* 1e9
    λ_res_nm = λ_res_nm[λ_min_nm .< λ_res_nm .< λ_max_nm]
    return sort(λ_res_nm, rev=true)
end

function find_anti_resonant_windows(wallthickness_um, ncl=1.45, nco=1.0;
                                     λ_min_nm=50.0, λ_max_nm=3000.0, margin_nm=10.0)
    λ_res = find_resonance_wavelengths(wallthickness_um, ncl, nco;
                                        λ_min_nm=λ_min_nm, λ_max_nm=λ_max_nm)
    windows = Vector{Tuple{Float64, Float64}}()

    if isempty(λ_res)
        push!(windows, (λ_min_nm, λ_max_nm))
        return windows, λ_res
    end

    if λ_res[1] + margin_nm < λ_max_nm
        push!(windows, (λ_res[1] + margin_nm, λ_max_nm))
    end

    for i in 2:length(λ_res)
        lo = λ_res[i] + margin_nm
        hi = λ_res[i - 1] - margin_nm
        if (hi - lo) > (2 * margin_nm)
            push!(windows, (lo, hi))
        end
    end

    if λ_res[end] - margin_nm > λ_min_nm
        push!(windows, (λ_min_nm, λ_res[end] - margin_nm))
    end

    return windows, λ_res
end

function interpolate_nan(x, y, nan_mask, good_idx)
    result = copy(y)
    for i in 1:length(x)
        if nan_mask[i]
            prev_idx = findlast(j -> j < i && !nan_mask[j], 1:length(x))
            next_idx = findfirst(j -> j > i && !nan_mask[j], 1:length(x))
            if prev_idx !== nothing && next_idx !== nothing
                t = (x[i] - x[prev_idx]) / (x[next_idx] - x[prev_idx])
                result[i] = y[prev_idx] + t * (y[next_idx] - y[prev_idx])
            end
        end
    end
    return result
end

function safe_savefig(filepath)
    dir = dirname(filepath)
    if !isempty(dir) && dir != "." && !isdir(dir)
        mkpath(dir)
    end
    if isfile(filepath)
        rm(filepath; force=true)
        println("  已删除旧文件: $(filepath)")
    end
    PyPlot.savefig(filepath, dpi=300, bbox_inches="tight")
end

function safe_rm(filepath)
    if isfile(filepath)
        rm(filepath; force=true)
        println("已删除旧文件: $(filepath)")
    end
end

function parse_bool_arg(value, name)
    v = lowercase(strip(value))
    if v in ("true", "1", "yes", "y")
        return true
    elseif v in ("false", "0", "no", "n")
        return false
    else
        error("$name requires true or false")
    end
end

function compute_window_dispersion(mode, λ_min_nm, λ_max_nm; min_points=30,
                                      debug=false)
    if λ_max_nm - λ_min_nm < 10.0
        return Float64[], Float64[], Float64[], Float64[]
    end

    n_points = max(min_points, round(Int, (λ_max_nm - λ_min_nm) / 0.3))
    n_points = min(n_points, 5000)

    λs = range(λ_min_nm * 1e-9, λ_max_nm * 1e-9, length=n_points)
    ωs = 2π * PhysData.c ./ λs

    neffs = zeros(ComplexF64, length(ωs))
    for i in 1:length(ωs)
        neffs[i] = Modes.neff(mode, ωs[i])
    end

    real_neff = real.(neffs)
    nan_mask = isnan.(real_neff) .| isinf.(real_neff)
    n_nan = count(nan_mask)

    if debug
        println("    窗口 [$(round(λ_min_nm,digits=1)), $(round(λ_max_nm,digits=1))] nm:")
        println("      总点数: $n_points, NaN/Inf: $n_nan")
        nonnan_idx = findall(.!nan_mask)
        if !isempty(nonnan_idx)
            println("      neff 范围: [$(minimum(real_neff[nonnan_idx])), $(maximum(real_neff[nonnan_idx]))]")
        end
    end

    valid = .!isnan.(real_neff) .& .!isinf.(real_neff)
    valid_idx = findall(valid)
    if length(valid_idx) < (min_points ÷ 2)
        if debug
            println("      跳过: 有效点不足 ($(length(valid_idx)) < $(min_points÷2))")
        end
        return Float64[], Float64[], Float64[], Float64[]
    end

    ωs_clean = ωs[valid_idx]
    real_neff_clean = real_neff[valid_idx]
    λs_clean_m = λs[valid_idx]

    c = PhysData.c
    βs = ωs_clean .* real_neff_clean / c

    if length(βs) < 20
        return Float64[], Float64[], Float64[], Float64[]
    end

    n = length(βs)
    n_eval = max(50, n ÷ 2)
    eval_ω = range(ωs_clean[1], ωs_clean[end], length=n_eval)
    eval_λ = 2π * c ./ eval_ω .* 1e9

    dβdω_fd = manual_diff(βs, ωs_clean)
    ω_mid_fd = (ωs_clean[1:(end - 1)] .+ ωs_clean[2:end]) ./ 2

    β2_fd = manual_diff(dβdω_fd, ω_mid_fd)
    ω_final_fd = (ω_mid_fd[1:(end - 1)] .+ ω_mid_fd[2:end]) ./ 2
    λ_final_fd = 2π * c ./ ω_final_fd .* 1e9
    D_fd = -2π * c ./ (ω_final_fd .^ 2) .* β2_fd .* 1e9

    if debug
        println("      FD 方法: D 范围 [$(minimum(D_fd)), $(maximum(D_fd))]")
    end

    n_poly = min(4, n - 1)
    if n_poly >= 3
        ω_center = (ωs_clean[1] + ωs_clean[end]) / 2
        ω_norm = (ωs_clean .- ω_center) ./ (ωs_clean[end] - ωs_clean[1])
        V = zeros(n, n_poly + 1)
        for j in 0:n_poly
            V[:, j + 1] = ω_norm .^ j
        end
        coeffs = V \ βs

        dβdω_poly = zeros(n_eval)
        β2_poly = zeros(n_eval)
        for j in 1:n_eval
            ωn = (eval_ω[j] - ω_center) / (ωs_clean[end] - ωs_clean[1])
            dβ = 0.0
            d2β = 0.0
            for p in 1:n_poly
                dβ += p * coeffs[p + 1] * ωn^(p - 1)
                if p >= 2
                    d2β += p * (p - 1) * coeffs[p + 1] * ωn^(p - 2)
                end
            end
            dβdω_poly[j] = dβ / (ωs_clean[end] - ωs_clean[1])
            β2_poly[j] = d2β / (ωs_clean[end] - ωs_clean[1])^2
        end

        β2_ps2_per_km_poly = β2_poly .* 1e27
        D_poly = -2π * c ./ (eval_ω .^ 2) .* β2_poly .* 1e9

        if debug
            println("      POLY 方法: D 范围 [$(minimum(D_poly)), $(maximum(D_poly))]")
            println("      POLY 方法: β₂ 范围 [$(minimum(β2_ps2_per_km_poly)), $(maximum(β2_ps2_per_km_poly))] ps²/km")
        end

        D_use = D_poly
        β2_use = β2_ps2_per_km_poly
        λ_use = eval_λ
        ω_loss = eval_ω
    else
        β2_fd_ps2_per_km = β2_fd .* 1e27
        D_use = D_fd
        β2_use = β2_fd_ps2_per_km
        λ_use = λ_final_fd
        ω_loss = ω_final_fd
    end

    n_loss = length(ω_loss)
    loss_dbm = zeros(Float64, n_loss)
    for i in 1:n_loss
        loss_dbm[i] = Modes.dB_per_m(mode, ω_loss[i])
    end

    if debug
        println("      最终 D 范围: [$(minimum(D_use)), $(maximum(D_use))]")
        println("      最终 β₂ 范围: [$(minimum(β2_use)), $(maximum(β2_use))] ps²/km")
        loss_finite = loss_dbm[isfinite.(loss_dbm)]
        if !isempty(loss_finite)
            println("      损耗范围: [$(minimum(loss_finite)), $(maximum(loss_finite))] dB/m")
        end
    end

    finite_D = isfinite.(D_use) .& isfinite.(β2_use) .& isfinite.(loss_dbm)
    return λ_use[finite_D], D_use[finite_D], β2_use[finite_D], loss_dbm[finite_D]
end

# ============================================================================
# 可视化辅助函数
# ============================================================================

function plot_dispersion_curve(mode, gas, pressure, wallthickness_um, output_prefix="dispersion")
    windows, λ_res_nm = find_anti_resonant_windows(wallthickness_um)

    all_λ = Float64[]
    all_D = Float64[]
    all_β2 = Float64[]
    all_loss = Float64[]

    println("  反谐振窗口划分 ($(length(windows)) 个):")
    for (i, (λ_lo, λ_hi)) in enumerate(windows)
        println("    窗口 $i: [$(round(λ_lo,digits=1)), $(round(λ_hi,digits=1))] nm")
    end

    for (λ_lo, λ_hi) in windows
        λs_w, Ds_w, β2s_w, losss_w = compute_window_dispersion(mode, λ_lo, λ_hi; debug=true)
        if !isempty(λs_w)
            append!(all_λ, λs_w)
            append!(all_D, Ds_w)
            append!(all_β2, β2s_w)
            append!(all_loss, losss_w)
        end
    end

    if !isempty(all_λ)
        perm = sortperm(all_λ)
        all_λ .= all_λ[perm]
        all_D .= all_D[perm]
        all_β2 .= all_β2[perm]
        all_loss .= all_loss[perm]
    end

    if isempty(all_λ)
        error("No finite dispersion data available for anti-resonant fibre plotting.")
    end

    gas_str = gas isa Symbol ? string(gas) : string(gas)
    n_effective_windows = length(λ_res_nm) + 1
    λ0_nm = 1030.0

    try
        # ====================================================================
        # Figure 1: 色散系数 D (ps/nm/km)
        # ====================================================================
        PyPlot.figure(figsize=(16, 8))
        PyPlot.plot(all_λ, all_D, color="#d62728", linewidth=1.0)
        PyPlot.xlabel("波长 (nm)", fontsize=12)
        PyPlot.ylabel("D (ps/nm/km)", fontsize=12)
        PyPlot.title("Dispersion Parameter D — Anti-resonant Fibre (WT=$(wallthickness_um) µm, " *
                     "$(gas_str) @ $(round(pressure, digits=1)) bar)  " *
                     "[$(n_effective_windows) windows, $(length(all_λ)) pts]",
                     fontsize=13)
        PyPlot.grid(true, alpha=0.3)
        PyPlot.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
        PyPlot.axvline(x=λ0_nm, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

        valid_abs_D = sort(abs.(all_D[isfinite.(all_D)]))
        if length(valid_abs_D) > 10
            clip_idx_D = min(round(Int, length(valid_abs_D) * 0.95), length(valid_abs_D))
            y_range_D = max(valid_abs_D[clip_idx_D] * 1.2, 1e-10)
            label_y = y_range_D * 0.7
            text_y = -y_range_D * 0.7
        else
            label_y = 0.5
            text_y = -0.5
        end
        PyPlot.text(λ0_nm + 5, label_y, "λ₀=1030 nm", fontsize=9, color="gray")

        for λr in λ_res_nm
            if 50 < λr < 3000
                PyPlot.axvline(x=λr, color="blue", linestyle="--",
                               linewidth=0.6, alpha=0.55)
                wt_m = wallthickness_um * 1e-6
                m_order = round(Int, 2 * wt_m * sqrt(1.45^2 - 1.0^2) * 1e9 / λr)
                PyPlot.text(λr + 2, text_y,
                           "m=$m_order",
                           fontsize=7.5, color="blue", alpha=0.6, rotation=90)
            end
        end

        if length(valid_abs_D) > 10
            PyPlot.ylim(-y_range_D, y_range_D)
        end

        PyPlot.xlim(50, 3000)
        PyPlot.tight_layout()
        local d_png_path = "$(output_prefix)_dispersion_D.png"
        safe_savefig(d_png_path)
        PyPlot.close()
        println("  色散系数 D 图像已保存: $(d_png_path)")

        # ====================================================================
        # Figure 2: 群速度色散参数 β₂ (ps²/km) — 范围限定 [-10, +10]
        # ====================================================================
        PyPlot.figure(figsize=(16, 8))

        # compute_window_dispersion returns beta2 in ps^2/km for plotting.
        # The HDF5 physics feature "beta2" remains in SI units (s^2/m).
        β2_ps2_per_km = all_β2
        b2_in_range = isfinite.(β2_ps2_per_km) .& (-10.0 .<= β2_ps2_per_km .<= 10.0)
        if any(b2_in_range)
            λ_b2 = all_λ[b2_in_range]
            β2_b2 = β2_ps2_per_km[b2_in_range]
            PyPlot.plot(λ_b2, β2_b2, color="#2ca02c", linewidth=1.0)
            n_pts_in = length(λ_b2)
        else
            PyPlot.plot(NaN, NaN, color="#2ca02c", linewidth=1.0)
            n_pts_in = 0
        end

        PyPlot.xlabel("波长 (nm)", fontsize=12)
        PyPlot.ylabel("β₂ (ps²/km)", fontsize=12)
        PyPlot.title("Group Velocity Dispersion β₂ — Anti-resonant Fibre (WT=$(wallthickness_um) µm, " *
                     "$(gas_str) @ $(round(pressure, digits=1)) bar)  " *
                     "[$(n_effective_windows) windows, $(length(all_λ)) pts, $n_pts_in in [-10,10]]",
                     fontsize=13)
        PyPlot.grid(true, alpha=0.3)
        PyPlot.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
        PyPlot.axvline(x=λ0_nm, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

        PyPlot.ylim(-10.0, 10.0)
        yticks_b2 = [-10, -5, 0, 5, 10]
        PyPlot.yticks(yticks_b2, [string(t) for t in yticks_b2])
        PyPlot.minorticks_on()

        label_y_b2 = 7.0
        text_y_b2 = -8.5
        PyPlot.text(λ0_nm + 5, label_y_b2, "λ₀=1030 nm", fontsize=9, color="gray")

        for λr in λ_res_nm
            if 50 < λr < 3000
                PyPlot.axvline(x=λr, color="blue", linestyle="--",
                               linewidth=0.6, alpha=0.55)
                wt_m = wallthickness_um * 1e-6
                m_order = round(Int, 2 * wt_m * sqrt(1.45^2 - 1.0^2) * 1e9 / λr)
                PyPlot.text(λr + 2, text_y_b2,
                           "m=$m_order",
                           fontsize=7.5, color="blue", alpha=0.6, rotation=90)
            end
        end

        PyPlot.xlim(50, 3000)
        PyPlot.tight_layout()
        local b2_png_path = "$(output_prefix)_dispersion_beta2.png"
        safe_savefig(b2_png_path)
        PyPlot.close()
        println("  群速度色散 β₂ 图像已保存: $(b2_png_path)  (范围内点数: $n_pts_in)")

        # ====================================================================
        # Figure 3: 光纤损耗曲线 (dB/m)
        # ====================================================================
        loss_finite_mask = isfinite.(all_loss) .& (all_loss .> 0)
        if any(loss_finite_mask)
            λ_loss = all_λ[loss_finite_mask]
            loss_plot = all_loss[loss_finite_mask]

            PyPlot.figure(figsize=(16, 8))
            PyPlot.semilogy(λ_loss, loss_plot, color="#9467bd", linewidth=1.0)
            PyPlot.xlabel("波长 (nm)", fontsize=12)
            PyPlot.ylabel("Confinement Loss (dB/m)", fontsize=12)
            PyPlot.title("Confinement Loss — Anti-resonant Fibre (WT=$(wallthickness_um) µm, " *
                         "$(gas_str) @ $(round(pressure, digits=1)) bar)  " *
                         "[$(n_effective_windows) windows, $(length(λ_loss)) pts]",
                         fontsize=13)
            PyPlot.grid(true, alpha=0.3, which="both")
            PyPlot.axvline(x=λ0_nm, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

            loss_mid = exp((log(minimum(loss_plot)) + log(maximum(loss_plot))) / 2)
            label_y_loss = loss_mid
            text_y_loss = minimum(loss_plot) * 1.5
            PyPlot.text(λ0_nm + 5, label_y_loss, "λ₀=1030 nm", fontsize=9, color="gray")

            for λr in λ_res_nm
                if 50 < λr < 3000
                    PyPlot.axvline(x=λr, color="blue", linestyle="--",
                                   linewidth=0.6, alpha=0.55)
                    wt_m = wallthickness_um * 1e-6
                    m_order = round(Int, 2 * wt_m * sqrt(1.45^2 - 1.0^2) * 1e9 / λr)
                    PyPlot.text(λr + 2, text_y_loss,
                               "m=$m_order",
                               fontsize=7.5, color="blue", alpha=0.6, rotation=90)
                end
            end

            PyPlot.xlim(50, 3000)
            PyPlot.tight_layout()
            local loss_png_path = "$(output_prefix)_loss.png"
            safe_savefig(loss_png_path)
            PyPlot.close()
            println("  损耗曲线已保存: $(loss_png_path)")
        else
            println("  警告: 无有效损耗数据（全部为 NaN/Inf/零/负值），跳过损耗曲线")
        end

        println("  反谐振窗口: $n_effective_windows 个, 有效数据点: $(length(all_λ))")
    catch e
        try
            PyPlot.close("all")
        catch
        end
        error("Dispersion plotting failed: $(sprint(showerror, e))")
    end
end

function plot_input_pulse(grid, τfwhm, energy, λ0, ϕ_vec, pulseshape, output_prefix="pulse")
    try
        if pulseshape == :sech
            pf = Fields.SechField(; λ0, energy, τfwhm, ϕ=ϕ_vec)
        elseif pulseshape == :gauss
            pf = Fields.GaussField(; λ0, energy, τfwhm, ϕ=ϕ_vec)
        else
            println("  警告: 不支持的脉冲形状 $(pulseshape)")
            return
        end

        Et = Fields.make_Et(pf, grid)
        t = grid.t
        ω0 = PhysData.wlfreq(λ0)

        It = abs2.(Et)
        It = It ./ maximum(It)

        dt = t[2] - t[1]
        n_orig = length(Et)
        n_fft = nextpow(2, n_orig)
        Et_padded = zeros(Float64, n_fft)
        Et_padded[1:n_orig] .= Et
        Eω_fft = FFTW.rfft(Et_padded)
        freqs = FFTW.rfftfreq(n_fft, 1 / dt)
        freq0_THz = ω0 / (2π) * 1e-12
        freqs_THz = freqs * 1e-12

        fig, (ax1, ax2) = PyPlot.subplots(1, 2, figsize=(14, 5))

        ax1.plot(t .* 1e15, It, color="#1f77b4", linewidth=1.2)
        ax1.set_xlabel("Time (fs)", fontsize=12)
        ax1.set_ylabel("Normalized Intensity", fontsize=12)
        ax1.set_title("Time Domain", fontsize=12)
        ax1.grid(true, alpha=0.3)

        Eω_intensity = abs2.(Eω_fft)
        Eω_intensity = Eω_intensity ./ maximum(Eω_intensity)
        ax2.plot(freqs_THz .- freq0_THz, Eω_intensity, color="#ff7f0e",
                 linewidth=1.2)
        ax2.set_xlabel("Frequency Offset (THz)", fontsize=12)
        ax2.set_ylabel("Normalized Spectral Intensity", fontsize=12)
        ax2.set_title("Frequency Domain", fontsize=12)
        ax2.grid(true, alpha=0.3)

        PyPlot.suptitle("Input Pulse at λ₀ = $(round(λ0*1e9, digits=1)) nm, " *
                        "τ = $(round(τfwhm*1e15, digits=1)) fs, " *
                        "E = $(round(energy*1e6, digits=3)) μJ",
                        fontsize=13)
        PyPlot.tight_layout()
        local png_path = "$(output_prefix)_pulse.png"
        #local pdf_path = "$(output_prefix)_pulse.pdf"
        safe_savefig(png_path)
        #safe_savefig(pdf_path)
        PyPlot.close()
        println("  脉冲图像已保存: $(png_path)")
    catch e
        println("  警告: 脉冲图像绘制失败: $(e)")
    end
end

# ============================================================================
# 命令行参数解析
# ============================================================================

# ============================================================================
# Anti-resonant fibre model export
# ============================================================================

function derivative_same_length(y, x)
    n = length(y)
    n == length(x) || error("derivative input arrays must have identical length")
    n >= 3 || error("at least 3 points are required for numerical derivatives")

    dy = similar(y, Float64)
    dy[1] = (y[2] - y[1]) / (x[2] - x[1])
    dy[end] = (y[end] - y[end - 1]) / (x[end] - x[end - 1])
    for i in 2:(n - 1)
        dy[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    end
    return dy
end

function make_export_wavelength_grid(wallthickness_um;
                                     λ_min_nm=50.0, λ_max_nm=3000.0,
                                     base_step_nm=1.0,
                                     resonance_step_nm=0.1,
                                     resonance_window_nm=2.0)
    base_step_nm > 0 || error("base_step_nm must be positive")
    resonance_step_nm > 0 || error("resonance_step_nm must be positive")
    λ_min_nm < λ_max_nm || error("λ_min_nm must be smaller than λ_max_nm")

    λ_points = collect(λ_min_nm:base_step_nm:λ_max_nm)
    if λ_points[end] < λ_max_nm
        push!(λ_points, λ_max_nm)
    end

    resonances = find_resonance_wavelengths(wallthickness_um;
                                            λ_min_nm=λ_min_nm,
                                            λ_max_nm=λ_max_nm)
    for λr in resonances
        lo = max(λ_min_nm, λr - resonance_window_nm)
        hi = min(λ_max_nm, λr + resonance_window_nm)
        append!(λ_points, collect(lo:resonance_step_nm:hi))
    end

    λ_sorted = sort(unique(round.(λ_points; digits=9)))
    all(diff(λ_sorted) .> 0) || error("export wavelength grid is not strictly increasing")
    return λ_sorted, resonances
end

function export_path_set(filepath)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    ext = lowercase(splitext(filepath)[2])

    if isempty(ext) && (isdir(filepath) || endswith(filepath, "/") || endswith(filepath, "\\"))
        outdir = filepath
        base = "fiber_model"
    elseif isempty(ext)
        outdir = dirname(filepath)
        base = basename(filepath)
        outdir = isempty(outdir) ? "." : outdir
    else
        outdir = dirname(filepath)
        base = splitext(basename(filepath))[1]
        outdir = isempty(outdir) ? "." : outdir
    end

    mkpath(outdir)
    prefix = joinpath(outdir, "$(base)_$(timestamp)")
    return Dict(
        "prefix" => prefix,
        "mat" => prefix * ".mat",
        "csv" => prefix * ".csv",
        "metadata" => prefix * "_metadata.json",
        "matlab" => joinpath(outdir, "read_fiber_model_matlab.m")
    )
end

function finite_ratio(arrays...)
    total = 0
    finite = 0
    for arr in arrays
        total += length(arr)
        finite += count(isfinite, arr)
    end
    return total == 0 ? 0.0 : finite / total
end

json_escape(s) = replace(replace(string(s), "\\" => "\\\\"), "\"" => "\\\"")

function json_value(v)
    if v isa AbstractString || v isa Symbol
        return "\"" * json_escape(v) * "\""
    elseif v isa Bool
        return v ? "true" : "false"
    elseif v isa Number
        if isnan(v)
            return "\"NaN\""
        elseif isinf(v)
            return v > 0 ? "\"Infinity\"" : "\"-Infinity\""
        else
            return string(v)
        end
    elseif v isa AbstractVector
        return "[" * join([json_value(x) for x in v], ", ") * "]"
    else
        return "\"" * json_escape(v) * "\""
    end
end

function write_metadata_json(path, metadata)
    open(path, "w") do io
        println(io, "{")
        keys_sorted = sort(collect(keys(metadata)); by=string)
        for (i, key) in enumerate(keys_sorted)
            comma = i == length(keys_sorted) ? "" : ","
            println(io, "  \"$(json_escape(key))\": $(json_value(metadata[key]))$comma")
        end
        println(io, "}")
    end
end

function write_fiber_model_csv(path, data)
    columns = ["lambda_nm", "D_ps_nm_km", "beta2_ps2_km", "beta3_ps3_km",
               "loss_dB_km", "n_eff_real", "n_eff_imag", "A_eff_um2", "gamma_W_km"]
    open(path, "w") do io
        println(io, join(columns, ","))
        n = length(data["lambda_nm"])
        for i in 1:n
            row = [data[col][i] for col in columns]
            println(io, join([@sprintf("%.16e", v) for v in row], ","))
        end
    end
end

function write_matlab_reader_script(path)
    script = raw"""
% Read and validate anti-resonant fibre model export.
% Set mat_file to a generated fiber_model_YYYYmmdd_HHMMSS.mat file.
mat_file = 'fiber_model_YYYYmmdd_HHMMSS.mat';

lambda_nm = h5read(mat_file, '/lambda_nm');
D_ps_nm_km = h5read(mat_file, '/D_ps_nm_km');
beta2_ps2_km = h5read(mat_file, '/beta2_ps2_km');
loss_dB_km = h5read(mat_file, '/loss_dB_km');
resonant_wavelengths = h5read(mat_file, '/resonant_wavelengths');

figure('Color', 'w');
subplot(2,1,1);
plot(lambda_nm, D_ps_nm_km, 'LineWidth', 1.2);
hold on;
yl = ylim;
for k = 1:numel(resonant_wavelengths)
    xline(resonant_wavelengths(k), '--', 'Color', [0.3 0.3 0.8]);
end
ylim(yl);
grid on;
xlabel('\lambda (nm)');
ylabel('D (ps/(nm km))');
title('Anti-resonant fibre dispersion');

subplot(2,1,2);
semilogy(lambda_nm, max(loss_dB_km, realmin), 'LineWidth', 1.2);
grid on;
xlabel('\lambda (nm)');
ylabel('Loss (dB/km)');
title('Confinement loss');

% D-beta2 consistency check:
% beta2_ps2_km -> SI: s^2/m. D_SI = -2*pi*c/lambda^2 * beta2_SI.
c = 299792458;
lambda_m = lambda_nm * 1e-9;
beta2_SI = beta2_ps2_km * 1e-27;
D_from_beta2 = -(2*pi*c ./ (lambda_m.^2)) .* beta2_SI * 1e6;

figure('Color', 'w');
plot(lambda_nm, D_ps_nm_km, 'k-', 'LineWidth', 1.2);
hold on;
plot(lambda_nm, D_from_beta2, 'r--', 'LineWidth', 1.0);
grid on;
xlabel('\lambda (nm)');
ylabel('D (ps/(nm km))');
legend('Exported D', 'Converted from \beta_2', 'Location', 'best');
title('D(\lambda) and \beta_2 conversion check');
"""
    open(path, "w") do io
        write(io, script)
    end
end

function write_fiber_model_hdf5_mat(path, data, metadata)
    h5open(path, "w") do file
        for (key, value) in data
            file[key] = value
        end

        g = create_group(file, "metadata")
        for (key, value) in metadata
            if value isa Number || value isa AbstractVector{<:Number}
                g[string(key)] = value
            end
        end
    end
end

function export_antiresonant_fiber_model(fiber_obj, filepath; format="mat", metadata=Dict())
    fmt = lowercase(format)
    fmt in ("mat", "all") || error("Only format=\"mat\" or format=\"all\" is supported")

    gas = get(metadata, "gas", :Ar)
    gas = gas isa Symbol ? gas : Symbol(gas)
    pressure = Float64(get(metadata, "pressure_bar", get(metadata, "pressure", 10.0)))
    wallthickness_m = hasproperty(fiber_obj, :wallthickness) ? getfield(fiber_obj, :wallthickness) :
                      Float64(get(metadata, "wall_thickness_m", NaN))
    wallthickness_um = wallthickness_m * 1e6
    core_radius = hasproperty(fiber_obj, :m) ? Capillary.radius(getfield(fiber_obj, :m), 0.0) :
                  Float64(get(metadata, "core_radius_m", NaN))
    λ0_nm = Float64(get(metadata, "lambda0_nm", 1030.0))

    base_step_nm = Float64(get(metadata, "export_step_nm", 1.0))
    resonance_step_nm = Float64(get(metadata, "export_resonance_step_nm", 0.1))
    resonance_window_nm = Float64(get(metadata, "export_resonance_window_nm", 2.0))

    λ_nm, resonances = make_export_wavelength_grid(wallthickness_um;
                                                   base_step_nm=base_step_nm,
                                                   resonance_step_nm=resonance_step_nm,
                                                   resonance_window_nm=resonance_window_nm)
    λ_m = λ_nm .* 1e-9
    ω = 2π .* PhysData.c ./ λ_m

    neffs = [Modes.neff(fiber_obj, ωi) for ωi in ω]
    n_eff_real = real.(neffs)
    n_eff_imag = imag.(neffs)
    β = n_eff_real .* ω ./ PhysData.c

    dβdω = derivative_same_length(β, ω)
    β2 = derivative_same_length(dβdω, ω)
    β3 = derivative_same_length(β2, ω)

    D_ps_nm_km = -(2π * PhysData.c ./ (λ_m .^ 2)) .* β2 .* 1e6
    beta2_ps2_km = β2 .* 1e27
    beta3_ps3_km = β3 .* 1e39
    loss_dB_km = [Modes.dB_per_m(fiber_obj, ωi) * 1000.0 for ωi in ω]

    Aeff = Modes.Aeff(fiber_obj)
    A_eff_um2 = fill(Aeff * 1e12, length(λ_nm))
    n2 = PhysData.n2(gas, pressure)
    gamma_W_km = n2 .* ω ./ (PhysData.c * Aeff) .* 1000.0

    windows, _ = find_anti_resonant_windows(wallthickness_um)
    band = [NaN, NaN]
    for (lo, hi) in windows
        if lo <= λ0_nm <= hi
            band = [lo, hi]
            break
        end
    end

    data = Dict{String, Any}(
        "lambda_nm" => λ_nm,
        "D_ps_nm_km" => D_ps_nm_km,
        "beta2_ps2_km" => beta2_ps2_km,
        "beta3_ps3_km" => beta3_ps3_km,
        "loss_dB_km" => loss_dB_km,
        "n_eff_real" => n_eff_real,
        "n_eff_imag" => n_eff_imag,
        "A_eff_um2" => A_eff_um2,
        "gamma_W_km" => gamma_W_km,
        "resonant_wavelengths" => resonances,
        "anti_resonant_band" => band,
        "core_radius" => [core_radius],
        "tube_count" => [NaN],
        "wall_thickness" => [wallthickness_m]
    )

    n = length(λ_nm)
    for key in ("D_ps_nm_km", "beta2_ps2_km", "beta3_ps3_km", "loss_dB_km",
                "n_eff_real", "n_eff_imag", "A_eff_um2", "gamma_W_km")
        length(data[key]) == n || error("export array length mismatch for $key")
    end
    all(diff(λ_nm) .> 0) || error("lambda_nm is not strictly increasing")

    ratio = finite_ratio(data["D_ps_nm_km"], data["beta2_ps2_km"], data["beta3_ps3_km"],
                         data["loss_dB_km"], data["n_eff_real"], data["n_eff_imag"],
                         data["A_eff_um2"], data["gamma_W_km"])
    if ratio < 0.9
        println("警告: 导出数据有限值比例仅为 $(round(ratio * 100, digits=2))%")
    end

    export_meta = Dict{Any, Any}(metadata)
    export_meta["created_at"] = string(now())
    export_meta["format_note"] = "HDF5-backed .mat; read in MATLAB with h5read."
    export_meta["model"] = "Antiresonant.ZeisbergerMode"
    export_meta["gas"] = string(gas)
    export_meta["pressure_bar"] = pressure
    export_meta["core_radius_m"] = core_radius
    export_meta["wall_thickness_m"] = wallthickness_m
    export_meta["wall_thickness_um"] = wallthickness_um
    export_meta["tube_count"] = NaN
    export_meta["tube_count_source"] = "not_defined_in_ZeisbergerMode"
    export_meta["lambda_min_nm"] = minimum(λ_nm)
    export_meta["lambda_max_nm"] = maximum(λ_nm)
    export_meta["sample_count"] = length(λ_nm)
    export_meta["base_step_nm"] = base_step_nm
    export_meta["resonance_step_nm"] = resonance_step_nm
    export_meta["resonance_window_nm"] = resonance_window_nm
    export_meta["finite_ratio"] = ratio
    export_meta["resonant_wavelengths_nm"] = resonances
    export_meta["anti_resonant_band_nm"] = band

    paths = export_path_set(filepath)
    write_fiber_model_hdf5_mat(paths["mat"], data, export_meta)
    write_fiber_model_csv(paths["csv"], data)
    write_metadata_json(paths["metadata"], export_meta)
    write_matlab_reader_script(paths["matlab"])

    println("反谐振光纤模型已导出:")
    println("  MAT/HDF5: $(paths["mat"])")
    println("  CSV:      $(paths["csv"])")
    println("  Metadata: $(paths["metadata"])")
    println("  MATLAB:   $(paths["matlab"])")
    return paths
end

function parse_command_line_args()
    args = Dict{String, Any}(
        "wall-thickness" => 0.65,
        "diameter" => 30.0,
        "energy" => 1.5,
        "tau" => 15.0,
        "pressure" => 10.0,
        "chirp" => 0.0,
        "output" => "anti_resonant_simulation.h5",
        "export-fiber-model" => false,
        "export-dir" => "fiber_model_exports",
        "export-step-nm" => 1.0,
        "export-resonance-step-nm" => 0.1,
        "export-resonance-window-nm" => 2.0,
        "search-extreme-samples" => false,
        "search-output-dir" => "extreme_sample_search",
        "search-candidates" => 80,
        "search-run-top" => 8,
        "search-run-all-runnable" => false,
        "search-resume" => true,
        "search-repair-corrupt" => true,
        "search-overwrite" => false,
        "search-seed" => 42,
        "search-energy-min" => 1.5,
        "search-energy-max" => 3.0,
        "search-tau-min" => 5.0,
        "search-tau-max" => 12.0,
        "search-pressure-min" => 10.0,
        "search-pressure-max" => 50.0,
        "search-diameter-min" => 10.0,
        "search-diameter-max" => 25.0,
        "search-energy-step" => 0.5,
        "search-tau-step" => 1.0,
        "search-pressure-step" => 5.0,
        "search-diameter-step" => 5.0
    )

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]

        if arg == "--wall-thickness" || arg == "-t"
            i += 1
            i <= length(ARGS) || error("--wall-thickness 需要一个数值参数")
            args["wall-thickness"] = parse(Float64, ARGS[i])
        elseif arg == "--diameter" || arg == "-d"
            i += 1
            i <= length(ARGS) || error("--diameter 需要一个数值参数")
            args["diameter"] = parse(Float64, ARGS[i])
        elseif arg == "--energy" || arg == "-e"
            i += 1
            i <= length(ARGS) || error("--energy 需要一个数值参数")
            args["energy"] = parse(Float64, ARGS[i])
        elseif arg == "--tau" || arg == "-τ"
            i += 1
            i <= length(ARGS) || error("--tau 需要一个数值参数")
            args["tau"] = parse(Float64, ARGS[i])
        elseif arg == "--pressure" || arg == "-p"
            i += 1
            i <= length(ARGS) || error("--pressure 需要一个数值参数")
            args["pressure"] = parse(Float64, ARGS[i])
        elseif arg == "--chirp" || arg == "-c"
            i += 1
            i <= length(ARGS) || error("--chirp 需要一个数值参数")
            args["chirp"] = parse(Float64, ARGS[i])
        elseif arg == "--output" || arg == "-o"
            i += 1
            i <= length(ARGS) || error("--output 需要一个文件路径")
            args["output"] = ARGS[i]
        elseif arg == "--export-fiber-model"
            args["export-fiber-model"] = true
        elseif arg == "--export-dir"
            i += 1
            i <= length(ARGS) || error("--export-dir 需要一个目录路径")
            args["export-dir"] = ARGS[i]
        elseif arg == "--export-step-nm"
            i += 1
            i <= length(ARGS) || error("--export-step-nm 需要一个数值参数")
            args["export-step-nm"] = parse(Float64, ARGS[i])
        elseif arg == "--export-resonance-step-nm"
            i += 1
            i <= length(ARGS) || error("--export-resonance-step-nm 需要一个数值参数")
            args["export-resonance-step-nm"] = parse(Float64, ARGS[i])
        elseif arg == "--export-resonance-window-nm"
            i += 1
            i <= length(ARGS) || error("--export-resonance-window-nm 需要一个数值参数")
            args["export-resonance-window-nm"] = parse(Float64, ARGS[i])
        elseif arg == "--search-extreme-samples"
            args["search-extreme-samples"] = true
        elseif arg == "--search-output-dir"
            i += 1
            i <= length(ARGS) || error("--search-output-dir requires a directory path")
            args["search-output-dir"] = ARGS[i]
        elseif arg == "--search-candidates"
            i += 1
            i <= length(ARGS) || error("--search-candidates requires an integer value")
            args["search-candidates"] = parse(Int, ARGS[i])
        elseif arg == "--search-run-top"
            i += 1
            i <= length(ARGS) || error("--search-run-top requires an integer value")
            args["search-run-top"] = parse(Int, ARGS[i])
        elseif arg == "--search-run-all-runnable"
            args["search-run-all-runnable"] = true
        elseif arg == "--search-resume"
            i += 1
            i <= length(ARGS) || error("--search-resume requires true or false")
            args["search-resume"] = parse_bool_arg(ARGS[i], "--search-resume")
        elseif arg == "--search-repair-corrupt"
            i += 1
            i <= length(ARGS) || error("--search-repair-corrupt requires true or false")
            args["search-repair-corrupt"] = parse_bool_arg(ARGS[i], "--search-repair-corrupt")
        elseif arg == "--search-overwrite"
            i += 1
            i <= length(ARGS) || error("--search-overwrite requires true or false")
            args["search-overwrite"] = parse_bool_arg(ARGS[i], "--search-overwrite")
        elseif arg == "--search-seed"
            i += 1
            i <= length(ARGS) || error("--search-seed requires an integer value")
            args["search-seed"] = parse(Int, ARGS[i])
        elseif arg == "--search-energy-min"
            i += 1
            i <= length(ARGS) || error("--search-energy-min requires a numeric value")
            args["search-energy-min"] = parse(Float64, ARGS[i])
        elseif arg == "--search-energy-max"
            i += 1
            i <= length(ARGS) || error("--search-energy-max requires a numeric value")
            args["search-energy-max"] = parse(Float64, ARGS[i])
        elseif arg == "--search-tau-min"
            i += 1
            i <= length(ARGS) || error("--search-tau-min requires a numeric value")
            args["search-tau-min"] = parse(Float64, ARGS[i])
        elseif arg == "--search-tau-max"
            i += 1
            i <= length(ARGS) || error("--search-tau-max requires a numeric value")
            args["search-tau-max"] = parse(Float64, ARGS[i])
        elseif arg == "--search-pressure-min"
            i += 1
            i <= length(ARGS) || error("--search-pressure-min requires a numeric value")
            args["search-pressure-min"] = parse(Float64, ARGS[i])
        elseif arg == "--search-pressure-max"
            i += 1
            i <= length(ARGS) || error("--search-pressure-max requires a numeric value")
            args["search-pressure-max"] = parse(Float64, ARGS[i])
        elseif arg == "--search-diameter-min"
            i += 1
            i <= length(ARGS) || error("--search-diameter-min requires a numeric value")
            args["search-diameter-min"] = parse(Float64, ARGS[i])
        elseif arg == "--search-diameter-max"
            i += 1
            i <= length(ARGS) || error("--search-diameter-max requires a numeric value")
            args["search-diameter-max"] = parse(Float64, ARGS[i])
        elseif arg == "--search-energy-step"
            i += 1
            i <= length(ARGS) || error("--search-energy-step requires a numeric value")
            args["search-energy-step"] = parse(Float64, ARGS[i])
        elseif arg == "--search-tau-step"
            i += 1
            i <= length(ARGS) || error("--search-tau-step requires a numeric value")
            args["search-tau-step"] = parse(Float64, ARGS[i])
        elseif arg == "--search-pressure-step"
            i += 1
            i <= length(ARGS) || error("--search-pressure-step requires a numeric value")
            args["search-pressure-step"] = parse(Float64, ARGS[i])
        elseif arg == "--search-diameter-step"
            i += 1
            i <= length(ARGS) || error("--search-diameter-step requires a numeric value")
            args["search-diameter-step"] = parse(Float64, ARGS[i])
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
反谐振光纤单次模拟脚本

用法: julia anti_resonant_simulation.jl [选项]

选项:
  -t, --wall-thickness VALUE  管壁厚度 (μm, 默认: 0.65)
  -d, --diameter VALUE        纤芯直径 (μm, 默认: 30)
  -e, --energy VALUE          脉冲能量 (μJ, 默认: 1.5)
  -τ, --tau VALUE             脉冲宽度 (fs, 默认: 15)
  -p, --pressure VALUE        气压 (bar, 默认: 10)
  -c, --chirp VALUE           啁啾参数 (默认: 0)
  -o, --output FILE           输出文件路径 (默认: anti_resonant_simulation.h5)
  -h, --help                  显示此帮助信息

示例:
  julia anti_resonant_simulation.jl -t 0.65 -d 30 -e 1.5 -p 10
  julia anti_resonant_simulation.jl --wall-thickness 0.36 --energy 1.0 --pressure 30
""")
end

# ============================================================================
# 参数验证
# ============================================================================

function print_help()
    println("""
Anti-resonant hollow-core fibre single simulation
Usage: julia anti_resonant_simulation.jl [options]

Options:
  -t, --wall-thickness VALUE          Wall thickness in um (default: 0.65)
  -d, --diameter VALUE                Core diameter in um (default: 30)
  -e, --energy VALUE                  Pulse energy in uJ (default: 1.5)
  --tau VALUE                         Pulse duration in fs (default: 15)
  -p, --pressure VALUE                Gas pressure in bar (default: 10)
  -c, --chirp VALUE                   Chirp parameter (default: 0)
  -o, --output FILE                   Simulation HDF5 output file
  --export-fiber-model                Export fibre model data (.mat/.csv/.json)
  --export-dir DIR                    Export directory (default: fiber_model_exports)
  --export-step-nm VALUE              Base wavelength step in nm (default: 1.0)
  --export-resonance-step-nm VALUE    Dense step near resonances in nm (default: 0.1)
  --export-resonance-window-nm VALUE  Half-width around each resonance in nm (default: 2.0)
  --search-extreme-samples            Search for strong-UV / strong-ionization samples
  --search-output-dir DIR             Extreme sample output directory
  --search-candidates VALUE           Number of candidate parameter sets (default: 80)
  --search-run-top VALUE              Number of top candidates to simulate (default: 8)
  --search-run-all-runnable           Simulate every runnable candidate after precheck
  --search-resume true|false          Skip complete extreme samples (default: true)
  --search-repair-corrupt true|false  Move incomplete/corrupt samples aside and rerun (default: true)
  --search-overwrite true|false       Regenerate existing complete extreme samples (default: false)
  --search-seed VALUE                 Random seed for candidate generation (default: 42)
  --search-energy-min/max VALUE       Search pulse energy range in uJ (default: 1.5-3.0)
  --search-tau-min/max VALUE          Search pulse duration range in fs (default: 5-12)
  --search-pressure-min/max VALUE     Search pressure range in bar (default: 10-50)
  --search-diameter-min/max VALUE     Search core diameter range in um (default: 10-25)
  --search-energy-step VALUE          Energy grid step in uJ (default: 0.5)
  --search-tau-step VALUE             Pulse duration grid step in fs (default: 1)
  --search-pressure-step VALUE        Pressure grid step in bar (default: 5)
  --search-diameter-step VALUE        Core diameter grid step in um (default: 5)
  -h, --help                          Show this help

Examples:
  julia anti_resonant_simulation.jl -t 0.65 -d 30 -e 1.5 -p 10
  julia anti_resonant_simulation.jl --export-fiber-model --export-dir fiber_model_exports
  julia anti_resonant_simulation.jl --search-extreme-samples --search-candidates 20 --search-run-top 3
  julia anti_resonant_simulation.jl --search-extreme-samples --search-candidates 0 --search-run-all-runnable
""")
end

function validate_parameters(args)
    wt = args["wall-thickness"]
    if wt <= 0.0
        error("管壁厚度必须大于0 (当前值: $wt μm)")
    end
    if wt < 0.08 || wt > 0.65
        println("警告: 管壁厚度 $wt μm 超出典型范围 [0.08, 0.65] μm")
    end

    diameter = args["diameter"]
    if diameter <= 0.0
        error("纤芯直径必须大于0 (当前值: $diameter μm)")
    end
    if diameter < 10.0 || diameter > 50.0
        println("警告: 纤芯直径 $diameter μm 超出典型范围 [10, 50] μm")
    end

    energy = args["energy"]
    if energy <= 0.0
        error("脉冲能量必须大于0 (当前值: $energy μJ)")
    end
    if energy < 0.3 || energy > 3.0
        println("警告: 脉冲能量 $energy μJ 超出典型范围 [0.3, 3.0] μJ")
    end

    tau = args["tau"]
    if tau <= 0.0
        error("脉冲宽度必须大于0 (当前值: $tau fs)")
    end
    if tau < 5.0 || tau > 50.0
        println("警告: 脉冲宽度 $tau fs 超出典型范围 [5, 50] fs")
    end

    pressure = args["pressure"]
    if pressure <= 0.0
        error("气压必须大于0 (当前值: $pressure bar)")
    end
    if pressure < 0.5 || pressure > 50.0
        println("警告: 气压 $pressure bar 超出典型范围 [0.5, 50] bar")
    end

    if args["export-step-nm"] <= 0.0
        error("--export-step-nm 必须大于0")
    end
    if args["export-resonance-step-nm"] <= 0.0
        error("--export-resonance-step-nm 必须大于0")
    end
    if args["export-resonance-window-nm"] < 0.0
        error("--export-resonance-window-nm 必须大于等于0")
    end

    if args["search-candidates"] < 0
        error("--search-candidates must be non-negative; use 0 for the full search grid")
    end
    if args["search-run-top"] < 0
        error("--search-run-top must be non-negative")
    end
    for name in ("energy", "tau", "pressure", "diameter")
        lo = args["search-$(name)-min"]
        hi = args["search-$(name)-max"]
        if lo <= 0.0 || hi <= 0.0 || lo >= hi
            error("--search-$(name)-min/max must be positive and min < max")
        end
        step = args["search-$(name)-step"]
        if step <= 0.0
            error("--search-$(name)-step must be positive")
        end
    end

    return true
end

# ============================================================================
# 物理参数计算函数（与 data_generation.jl 保持一致）
# ============================================================================

function calculate_beta2(ω, β)
    n = length(ω)
    dβdω = similar(β, n)
    dβdω[1] = (β[2] - β[1]) / (ω[2] - ω[1])
    dβdω[end] = (β[end] - β[end - 1]) / (ω[end] - ω[end - 1])
    for i in 2:(n - 1)
        dβdω[i] = (β[i + 1] - β[i - 1]) / (ω[i + 1] - ω[i - 1])
    end

    β2 = similar(dβdω, n)
    β2[1] = (dβdω[2] - dβdω[1]) / (ω[2] - ω[1])
    β2[end] = (dβdω[end] - dβdω[end - 1]) / (ω[end] - ω[end - 1])
    for i in 2:(n - 1)
        β2[i] = (dβdω[i + 1] - dβdω[i - 1]) / (ω[i + 1] - ω[i - 1])
    end

    return β2
end

function calculate_physics_features(mode, grid, λ0, τfwhm, energy, pressure, gas)
    ω0 = 2π * PhysData.c / λ0

    neffs = [Modes.neff(mode, ω) for ω in grid.ω]
    βs = grid.ω .* real.(neffs) / PhysData.c

    idx = argmin(abs.(grid.ω .- ω0))

    β2s = calculate_beta2(grid.ω, βs)
    β2 = β2s[idx]

    Aeff = Modes.Aeff(mode)

    n2 = PhysData.n2(gas, pressure)
    k0 = ω0 / PhysData.c
    γ = n2 * k0 / Aeff

    τ0 = τfwhm / (2 * log(1 + sqrt(2)))
    P0 = energy / (τ0 * sqrt(π))

    if abs(β2) < 1e-30
        println("警告: β2 接近零，使用保护值")
        L0 = τ0^2 / 1e-30
    else
        L0 = τ0^2 / abs(β2)
    end

    N = sqrt(γ * P0 * L0)

    I_p = PhysData.ionisation_potential(gas)
    E0 = sqrt(2 * P0 / (π * Aeff))
    γ_K = ω0 * sqrt(2 * PhysData.m_e * I_p) / (PhysData.electron * E0)

    if abs(n2) < 1e-30
        P_cr = Inf
    else
        P_cr = 3.77 * 1e4 * (λ0^2) / (8π * n2 * Aeff)
    end
    P_ratio = P0 / P_cr

    return Dict(
        "beta2" => β2,
        "gamma" => γ,
        "N" => N,
        "L0" => L0,
        "gamma_K" => γ_K,
        "P_ratio" => P_ratio,
        "Aeff" => Aeff,
        "neff" => real(neffs[idx]),
        "P0" => P0,
        "τ0" => τ0,
        "E0" => E0
    )
end

# ============================================================================
# 反谐振光纤专用传播函数
# ============================================================================

function prop_antiresonant(radius, flength, gas, pressure, wallthickness_m;
                          λlims=(200e-9, 2500e-9), trange=500e-15,
                          λ0=1030e-9, τfwhm, energy,
                          ϕ=Float64[], pulseshape=:sech,
                          model=:full, loss=true,
                          raman=false, plasma=true,
                          shotnoise=true,
                          saveN=201, filepath=nothing,
                          status_period=5)
    grid = Grid.RealGrid(flength, λ0, λlims, trange)
    mode = Antiresonant.ZeisbergerMode(radius, gas, pressure;
                                      wallthickness=wallthickness_m,
                                      model=model, loss=loss)
    density = z -> PhysData.density(gas, pressure, PhysData.roomtemp)
    resp = Interface.makeresponse(grid, gas, raman, true, plasma,
                                 true, false, true, true,
                                 Dict{Symbol, Any}(), 0.0, PhysData.roomtemp)
    inputs = Interface.makeinputs(mode, λ0, nothing, τfwhm, nothing, ϕ, nothing, energy,
                                 pulseshape, :linear, nothing)
    inputs, noise_field = Interface.makenoise(grid, mode, inputs, shotnoise,
                                              Random.GLOBAL_RNG)
    linop, Eω, transform, FT = Interface.setup(grid, mode, density, resp, inputs,
                                                false, 1e-3, Val(true);
                                                noise_field)
    stats = Stats.default(grid, Eω, mode, linop, transform; gas=gas)
    output = Interface.makeoutput(grid, saveN, stats, filepath, nothing, nothing, nothing)
    Interface.saveargs(output; radius, flength, gas, pressure, λlims, trange,
                      λ0, τfwhm, ϕ, energy, pulseshape, model, loss,
                      raman, plasma, saveN, filepath)
    Luna.run(Eω, grid, linop, transform, FT, output; status_period)
    return output
end

# ============================================================================
# 主程序
# ============================================================================

# ============================================================================
# Extreme UV / ionization sample search
# ============================================================================

function stepped_values(lo, hi, step)
    values = collect(lo:step:hi)
    if isempty(values) || values[end] < hi - 1e-9
        push!(values, hi)
    end
    return round.(values; digits=9)
end

function generate_extreme_candidate_parameters(n; ranges, steps, seed=42, chirp=0.0)
    rng = MersenneTwister(seed)
    energies = stepped_values(ranges["energy"][1], ranges["energy"][2], steps["energy"])
    taus = stepped_values(ranges["tau"][1], ranges["tau"][2], steps["tau"])
    pressures = stepped_values(ranges["pressure"][1], ranges["pressure"][2], steps["pressure"])
    diameters = stepped_values(ranges["diameter"][1], ranges["diameter"][2], steps["diameter"])

    candidates = Vector{Dict{String, Any}}()
    candidate_id = 1
    for energy_uJ in energies, tau_fs in taus, pressure_bar in pressures, diameter_um in diameters
        push!(candidates, Dict{String, Any}(
            "candidate_id" => candidate_id,
            "energy_uJ" => energy_uJ,
            "tau_fs" => tau_fs,
            "pressure_bar" => pressure_bar,
            "diameter_um" => diameter_um,
            "chirp" => chirp
        ))
        candidate_id += 1
    end

    if n > 0 && length(candidates) > n
        candidates = shuffle(rng, candidates)[1:n]
        for (i, candidate) in enumerate(candidates)
            candidate["candidate_id"] = i
        end
    end
    return candidates
end

function positive_finite(x, fallback=0.0)
    return isfinite(x) && x > 0.0 ? x : fallback
end

function score_extreme_candidate(params; λ0, λlims_sim, trange, gas, wallthickness_m,
                                 flength=0.5)
    energy = params["energy_uJ"] * 1e-6
    τfwhm = params["tau_fs"] * 1e-15
    pressure = params["pressure_bar"]
    diameter = params["diameter_um"]
    radius = diameter / 2 * 1e-6

    row = copy(params)
    try
        mode = Antiresonant.ZeisbergerMode(radius, gas, pressure;
                                           wallthickness=wallthickness_m,
                                           model=:full, loss=true)
        grid = Grid.RealGrid(flength, λ0, λlims_sim, trange)
        features = calculate_physics_features(mode, grid, λ0, τfwhm, energy, pressure, gas)

        E0 = Float64(features["E0"])
        I_p = PhysData.ionisation_potential(gas)
        Emax_ppt = 2 * barrier_suppression_field(I_p, 1)
        E0_ratio = E0 / Emax_ppt
        E0_dataset_ratio = E0 / EXTREME_DATASET_E_LIMIT
        gamma_K = Float64(features["gamma_K"])
        P_ratio = Float64(features["P_ratio"])
        N = Float64(features["N"])
        gamma = Float64(features["gamma"])
        Aeff = Float64(features["Aeff"])
        field_limit_exceeded = E0 >= EXTREME_DATASET_E_LIMIT
        strong_ionization_candidate = (E0_dataset_ratio >= EXTREME_STRONG_E_LIMIT_RATIO) ||
                                      (E0_ratio >= EXTREME_STRONG_PPT_RATIO) ||
                                      (gamma_K <= EXTREME_STRONG_GAMMA_K)

        ionization_score = E0_dataset_ratio + E0_ratio + 1.0 / max(gamma_K, 1e-12) + log1p(positive_finite(P_ratio))
        uv_prior_score = log1p(positive_finite(N)) +
                         log1p(abs(gamma) * 1e6) +
                         params["energy_uJ"] / 3.0 +
                         12.0 / params["tau_fs"] +
                         25.0 / params["diameter_um"]
        combined_score = ionization_score + uv_prior_score

        row["E0_V_m"] = E0
        row["E0_GV_m"] = E0 / 1e9
        row["Emax_ppt_V_m"] = Emax_ppt
        row["E0_over_Emax_ppt"] = E0_ratio
        row["dataset_E_limit_V_m"] = EXTREME_DATASET_E_LIMIT
        row["E0_over_dataset_limit"] = E0_dataset_ratio
        row["gamma_K"] = gamma_K
        row["P_ratio"] = P_ratio
        row["N"] = N
        row["gamma"] = gamma
        row["Aeff"] = Aeff
        row["beta2"] = Float64(features["beta2"])
        row["ionization_score"] = ionization_score
        row["uv_prior_score"] = uv_prior_score
        row["combined_score"] = combined_score
        row["field_limit_exceeded"] = field_limit_exceeded
        row["strong_ionization_candidate"] = strong_ionization_candidate
        row["precheck_status"] = row["field_limit_exceeded"] ? "field_limit_exceeded" : "candidate_ok"
    catch e
        row["E0_V_m"] = NaN
        row["E0_GV_m"] = NaN
        row["Emax_ppt_V_m"] = NaN
        row["E0_over_Emax_ppt"] = NaN
        row["dataset_E_limit_V_m"] = EXTREME_DATASET_E_LIMIT
        row["E0_over_dataset_limit"] = NaN
        row["gamma_K"] = NaN
        row["P_ratio"] = NaN
        row["N"] = NaN
        row["gamma"] = NaN
        row["Aeff"] = NaN
        row["beta2"] = NaN
        row["ionization_score"] = -Inf
        row["uv_prior_score"] = -Inf
        row["combined_score"] = -Inf
        row["field_limit_exceeded"] = false
        row["strong_ionization_candidate"] = false
        row["precheck_status"] = "precheck_failed"
        row["failure_reason"] = sprint(showerror, e)
    end
    row["simulation_status"] = "not_run"
    row["uv_fraction_200_700"] = NaN
    row["uv_peak_200_700"] = NaN
    row["final_peak_wavelength_nm"] = NaN
    row["output_h5"] = ""
    row["propagation_png"] = ""
    row["propagation_zoom_2cm_png"] = ""
    row["spectral_propagation_zoom_2cm_png"] = ""
    row["propagation_zoom_10cm_png"] = ""
    row["spectral_propagation_zoom_10cm_png"] = ""
    row["propagation_highres_png"] = ""
    row["spectral_propagation_highres_png"] = ""
    row["spectrum_png"] = ""
    row["spectrum_db_png"] = ""
    row["time_png"] = ""
    return row
end

function trapz_integral(x, y)
    length(x) == length(y) || error("trapz_integral inputs must have the same length")
    if length(x) < 2
        return 0.0
    end
    total = 0.0
    for i in 1:(length(x) - 1)
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    end
    return total
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
            if maximum(idx) > length(ωo) || minimum(idx) < 1
                return Float64[]
            end
            ω = ωo[idx]
            if length(ω) == freq_count
                return Float64.(2 * pi .* PhysData.c ./ ω .* 1e9)
            end
        end
    end

    return Float64[]
end

function compute_final_uv_metrics(filepath; λrange_total_nm=(200.0, 2500.0),
                                  λrange_uv_nm=(200.0, 700.0))
    metrics = Dict{String, Any}(
        "uv_fraction_200_700" => NaN,
        "uv_peak_200_700" => NaN,
        "final_peak_wavelength_nm" => NaN
    )

    try
        h5open(filepath, "r") do file
            if !haskey(file, "Eω") || !haskey(file, "z")
                return metrics
            end
            Eω = read(file["Eω"])
            z = vec(read(file["z"]))
            if ndims(Eω) != 2 || isempty(z)
                return metrics
            end

            final_spectrum = size(Eω, 2) == length(z) ? vec(Eω[:, end]) : vec(Eω[end, :])
            λ_nm = read_wavelength_axis_nm(file, length(final_spectrum))
            if length(λ_nm) != length(final_spectrum)
                return metrics
            end

            power = abs2.(final_spectrum)
            valid = isfinite.(λ_nm) .& isfinite.(power) .& (λ_nm .> 0)
            λ_nm = λ_nm[valid]
            power = power[valid]
            if length(λ_nm) < 2
                return metrics
            end

            perm = sortperm(λ_nm)
            λ_nm = λ_nm[perm]
            power = power[perm]

            total_mask = (λrange_total_nm[1] .<= λ_nm) .& (λ_nm .<= λrange_total_nm[2])
            uv_mask = (λrange_uv_nm[1] .<= λ_nm) .& (λ_nm .<= λrange_uv_nm[2])
            if count(total_mask) < 2 || count(uv_mask) < 2
                return metrics
            end

            total_power = trapz_integral(λ_nm[total_mask], power[total_mask])
            uv_power = trapz_integral(λ_nm[uv_mask], power[uv_mask])
            if total_power > 0.0
                metrics["uv_fraction_200_700"] = uv_power / total_power
            end
            metrics["uv_peak_200_700"] = maximum(power[uv_mask])
            total_λ = λ_nm[total_mask]
            total_power_axis = power[total_mask]
            metrics["final_peak_wavelength_nm"] = total_λ[argmax(total_power_axis)]
        end
    catch e
        metrics["failure_reason"] = sprint(showerror, e)
    end

    return metrics
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

function plot_final_spectrum_db(h5_filepath, fig_filepath; λrange=(200e-9, 2500e-9),
                                dBmin=-40.0)
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

function plot_propagation_zoom(output, filepath; λrange=(200e-9, 2500e-9),
                               zrange_cm=(0.0, 2.0), dBmin=-40.0)
    Plotting.prop_2D(output, :λ, dBmin=dBmin, λrange=λrange,
                     trange=(-100e-15, 100e-15))
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
    fig.suptitle("Propagation Zoom: z = $(zrange_cm[1])-$(zrange_cm[2]) cm")
    safe_savefig(filepath)
    PyPlot.close()
    return filepath
end

function plot_spectral_propagation_zoom(output, filepath; λrange=(200e-9, 2500e-9),
                                        zrange_cm=(0.0, 2.0), dBmin=-40.0)
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

const EARLY_HIGHRES_FLENGTH = 0.005
const EARLY_HIGHRES_SAVEN = 201

function run_early_highres_propagation(radius, gas, pressure, wallthickness_m;
                                       λ0, τfwhm, energy, ϕ_vec=Float64[],
                                       λlims=(200e-9, 2500e-9),
                                       trange=500e-15,
                                       saveN=EARLY_HIGHRES_SAVEN)
    return prop_antiresonant(radius, EARLY_HIGHRES_FLENGTH, gas, pressure, wallthickness_m;
                             λ0, τfwhm, energy,
                             λlims=λlims, trange=trange,
                             ϕ=ϕ_vec,
                             pulseshape=:sech,
                             model=:full,
                             loss=true,
                             raman=false,
                             plasma=true,
                             saveN=saveN,
                             filepath=nothing,
                             status_period=30)
end

function visualize_early_highres_propagation(output, prefix;
                                             λrange=(200e-9, 2500e-9),
                                             zrange_cm=(0.0, 0.5),
                                             dBmin=-40.0)
    propagation_highres_png = "$(prefix)_propagation_z0_0p5cm_highres.png"
    spectral_highres_png = "$(prefix)_spectral_propagation_z0_0p5cm_highres.png"

    plot_propagation_zoom(output, propagation_highres_png;
                          λrange=λrange, zrange_cm=zrange_cm, dBmin=dBmin)
    plot_spectral_propagation_zoom(output, spectral_highres_png;
                                   λrange=λrange, zrange_cm=zrange_cm, dBmin=dBmin)
    return propagation_highres_png, spectral_highres_png
end

function visualize_extreme_sample(output, prefix, flength)
    propagation_png = "$(prefix)_propagation.png"
    propagation_zoom_2cm_png = "$(prefix)_propagation_z0_2cm.png"
    spectral_propagation_zoom_2cm_png = "$(prefix)_spectral_propagation_z0_2cm.png"
    propagation_zoom_10cm_png = "$(prefix)_propagation_z0_10cm.png"
    spectral_propagation_zoom_10cm_png = "$(prefix)_spectral_propagation_z0_10cm.png"
    spectrum_png = "$(prefix)_spectrum.png"
    spectrum_db_png = "$(prefix)_spectrum_db.png"
    time_png = "$(prefix)_time.png"
    h5_filepath = "$(prefix).h5"

    Plotting.prop_2D(output, :λ, dBmin=-40.0,
                     λrange=(200e-9, 2500e-9),
                     trange=(-100e-15, 100e-15))
    safe_savefig(propagation_png)
    PyPlot.close()

    plot_propagation_zoom(output, propagation_zoom_2cm_png;
                          λrange=(200e-9, 2500e-9),
                          zrange_cm=(0.0, 2.0),
                          dBmin=-40.0)

    plot_spectral_propagation_zoom(output, spectral_propagation_zoom_2cm_png;
                                   λrange=(200e-9, 2500e-9),
                                   zrange_cm=(0.0, 2.0),
                                   dBmin=-40.0)

    plot_propagation_zoom(output, propagation_zoom_10cm_png;
                          λrange=(200e-9, 2500e-9),
                          zrange_cm=(0.0, 10.0),
                          dBmin=-40.0)

    plot_spectral_propagation_zoom(output, spectral_propagation_zoom_10cm_png;
                                   λrange=(200e-9, 2500e-9),
                                   zrange_cm=(0.0, 10.0),
                                   dBmin=-40.0)

    Plotting.spec_1D(output, range(0.0, flength, length=5),
                     λrange=(200e-9, 2500e-9))
    safe_savefig(spectrum_png)
    PyPlot.close()

    plot_final_spectrum_db(h5_filepath, spectrum_db_png;
                           λrange=(200e-9, 2500e-9),
                           dBmin=-40.0)

    Plotting.time_1D(output, range(0.0, flength, length=5))
    safe_savefig(time_png)
    PyPlot.close()

    return propagation_png, propagation_zoom_2cm_png, spectral_propagation_zoom_2cm_png,
           propagation_zoom_10cm_png, spectral_propagation_zoom_10cm_png,
           spectrum_png, spectrum_db_png, time_png
end

function save_extreme_sample_metadata(filepath, params, features, wallthickness_um, wallthickness_m, flength)
    h5open(filepath, "r+") do file
        if !haskey(file, "physics_features")
            g = create_group(file, "physics_features")
            for (key, val) in features
                if val isa Number
                    g[key] = val
                end
            end
            g["wallthickness"] = wallthickness_m
            g["fibre_model"] = "antiresonant"
            g["chirp_enabled"] = (params["chirp"] != 0.0)
        end

        if !haskey(file, "input")
            inp = create_group(file, "input")
            inp["energy"] = params["energy_uJ"]
            inp["tau"] = params["tau_fs"]
            inp["pressure"] = params["pressure_bar"]
            inp["diameter"] = params["diameter_um"]
            inp["wallthickness"] = wallthickness_um
            inp["chirp"] = params["chirp"]
            inp["flength"] = flength
        end
    end
end

function extreme_sample_prefix(index, output_dir)
    return joinpath(output_dir, "extreme_sample_$(lpad(index, 3, '0'))")
end

function extreme_sample_paths(index, output_dir)
    prefix = extreme_sample_prefix(index, output_dir)
    return Dict{String, String}(
        "prefix" => prefix,
        "h5" => "$(prefix).h5",
        "tmp" => "$(prefix).h5.tmp",
        "done" => "$(prefix).h5.done",
        "propagation_png" => "$(prefix)_propagation.png",
        "propagation_zoom_2cm_png" => "$(prefix)_propagation_z0_2cm.png",
        "spectral_propagation_zoom_2cm_png" => "$(prefix)_spectral_propagation_z0_2cm.png",
        "propagation_zoom_10cm_png" => "$(prefix)_propagation_z0_10cm.png",
        "spectral_propagation_zoom_10cm_png" => "$(prefix)_spectral_propagation_z0_10cm.png",
        "propagation_highres_png" => "$(prefix)_propagation_z0_0p5cm_highres.png",
        "spectral_propagation_highres_png" => "$(prefix)_spectral_propagation_z0_0p5cm_highres.png",
        "spectrum_png" => "$(prefix)_spectrum.png",
        "spectrum_db_png" => "$(prefix)_spectrum_db.png",
        "time_png" => "$(prefix)_time.png"
    )
end

function extreme_temp_filepath(index, output_dir)
    return extreme_sample_paths(index, output_dir)["tmp"]
end

function extreme_done_marker_path(filepath)
    return filepath * ".done"
end

function read_key_value_marker(path)
    data = Dict{String, String}()
    if !isfile(path)
        return data
    end
    for line in eachline(path)
        parts = split(line, "=", limit=2)
        if length(parts) == 2
            data[strip(parts[1])] = strip(parts[2])
        end
    end
    return data
end

function has_valid_extreme_done_marker(filepath, expected_saveN)
    marker = extreme_done_marker_path(filepath)
    if !isfile(filepath) || !isfile(marker)
        return false
    end
    data = read_key_value_marker(marker)
    marker_saveN = tryparse(Int, get(data, "expected_saveN", ""))
    marker_size = tryparse(Int, get(data, "file_size", ""))
    if marker_saveN === nothing || marker_saveN != expected_saveN
        return false
    end
    if marker_size !== nothing && marker_size != filesize(filepath)
        return false
    end
    return true
end

function is_complete_extreme_sample(filepath, expected_saveN)
    has_valid_extreme_done_marker(filepath, expected_saveN) && return true
    if !isfile(filepath)
        return false
    end
    try
        h5open(filepath, "r") do file
            for key in ("Eω", "z", "physics_features", "input")
                if !haskey(file, key)
                    return false
                end
            end
            z = vec(read(file["z"]))
            return length(z) >= expected_saveN
        end
    catch
        return false
    end
end

function write_extreme_done_marker(filepath, row, expected_saveN)
    marker = extreme_done_marker_path(filepath)
    open(marker, "w") do io
        println(io, "status=complete")
        println(io, "candidate_id=$(get(row, "candidate_id", ""))")
        println(io, "simulation_rank=$(get(row, "simulation_rank", ""))")
        println(io, "energy_uJ=$(get(row, "energy_uJ", ""))")
        println(io, "tau_fs=$(get(row, "tau_fs", ""))")
        println(io, "pressure_bar=$(get(row, "pressure_bar", ""))")
        println(io, "diameter_um=$(get(row, "diameter_um", ""))")
        println(io, "chirp=$(get(row, "chirp", ""))")
        println(io, "expected_saveN=$expected_saveN")
        println(io, "file_size=$(filesize(filepath))")
        println(io, "completed_at=$(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
    end
    return marker
end

function move_corrupt_extreme_sample(filepath, output_dir, reason)
    corrupt_dir = joinpath(output_dir, "corrupt_extreme_samples")
    mkpath(corrupt_dir)
    timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    dest = joinpath(corrupt_dir, "$(basename(filepath)).$(timestamp).$(reason)")
    if isfile(filepath)
        mv(filepath, dest; force=true)
        println("  Moved corrupt extreme sample to: $dest")
    end
    marker = extreme_done_marker_path(filepath)
    if isfile(marker)
        mv(marker, dest * ".done"; force=true)
    end
    return dest
end

function run_extreme_sample(params, index, output_dir; λ0, λlims_sim, trange,
                            gas, wallthickness_um, wallthickness_m, flength,
                            saveN, resume=true, repair_corrupt=true,
                            overwrite=false)
    paths = extreme_sample_paths(index, output_dir)
    prefix = paths["prefix"]
    filepath = paths["h5"]
    tmp_filepath = paths["tmp"]

    if isfile(tmp_filepath)
        println("  Removing stale temporary extreme sample: $tmp_filepath")
        rm(tmp_filepath; force=true)
    end

    if isfile(filepath)
        if overwrite
            println("  search-overwrite enabled; removing existing extreme sample: $filepath")
            rm(filepath; force=true)
            marker = extreme_done_marker_path(filepath)
            isfile(marker) && rm(marker; force=true)
        elseif is_complete_extreme_sample(filepath, saveN)
            if !has_valid_extreme_done_marker(filepath, saveN)
                write_extreme_done_marker(filepath, params, saveN)
            end
            uv_metrics = compute_final_uv_metrics(filepath)
            return Dict{String, Any}(
                "simulation_status" => "skipped_complete",
                "output_h5" => filepath,
                "propagation_png" => paths["propagation_png"],
                "propagation_zoom_2cm_png" => paths["propagation_zoom_2cm_png"],
                "spectral_propagation_zoom_2cm_png" => paths["spectral_propagation_zoom_2cm_png"],
                "propagation_zoom_10cm_png" => paths["propagation_zoom_10cm_png"],
                "spectral_propagation_zoom_10cm_png" => paths["spectral_propagation_zoom_10cm_png"],
                "propagation_highres_png" => paths["propagation_highres_png"],
                "spectral_propagation_highres_png" => paths["spectral_propagation_highres_png"],
                "spectrum_png" => paths["spectrum_png"],
                "spectrum_db_png" => paths["spectrum_db_png"],
                "time_png" => paths["time_png"],
                "uv_fraction_200_700" => uv_metrics["uv_fraction_200_700"],
                "uv_peak_200_700" => uv_metrics["uv_peak_200_700"],
                "final_peak_wavelength_nm" => uv_metrics["final_peak_wavelength_nm"]
            )
        elseif repair_corrupt
            move_corrupt_extreme_sample(filepath, output_dir, "failed_integrity_check")
        else
            return Dict{String, Any}(
                "simulation_status" => "incomplete_or_corrupt_repair_disabled",
                "output_h5" => filepath,
                "failure_reason" => "existing extreme sample failed integrity check and search-repair-corrupt=false"
            )
        end
    end

    energy = params["energy_uJ"] * 1e-6
    τfwhm = params["tau_fs"] * 1e-15
    pressure = params["pressure_bar"]
    diameter = params["diameter_um"]
    radius = diameter / 2 * 1e-6
    τ0 = τfwhm / (2 * log(1 + sqrt(2)))
    gdd = params["chirp"] * τ0^2
    ϕ_vec = Float64[0.0, 0.0, gdd]

    output = prop_antiresonant(radius, flength, gas, pressure, wallthickness_m;
                              λ0, τfwhm, energy,
                              λlims=λlims_sim, trange,
                              ϕ=ϕ_vec,
                              pulseshape=:sech,
                              model=:full,
                              loss=true,
                              raman=false,
                              plasma=true,
                              saveN=saveN,
                              filepath=tmp_filepath)

    mode = Antiresonant.ZeisbergerMode(radius, gas, pressure;
                                       wallthickness=wallthickness_m,
                                       model=:full, loss=true)
    grid = Grid.RealGrid(flength, λ0, λlims_sim, trange)
    features = calculate_physics_features(mode, grid, λ0, τfwhm, energy, pressure, gas)
    save_extreme_sample_metadata(tmp_filepath, params, features, wallthickness_um, wallthickness_m, flength)

    if !is_complete_extreme_sample(tmp_filepath, saveN)
        error("temporary extreme output failed integrity check: $tmp_filepath")
    end

    mv(tmp_filepath, filepath; force=true)

    propagation_png, propagation_zoom_2cm_png, spectral_propagation_zoom_2cm_png,
        propagation_zoom_10cm_png, spectral_propagation_zoom_10cm_png,
        spectrum_png, spectrum_db_png, time_png =
        visualize_extreme_sample(output, prefix, flength)
    highres_output = run_early_highres_propagation(radius, gas, pressure, wallthickness_m;
                                                  λ0, τfwhm, energy,
                                                  ϕ_vec=ϕ_vec,
                                                  λlims=λlims_sim,
                                                  trange=trange,
                                                  saveN=EARLY_HIGHRES_SAVEN)
    propagation_highres_png, spectral_propagation_highres_png =
        visualize_early_highres_propagation(highres_output, prefix;
                                            λrange=λlims_sim,
                                            zrange_cm=(0.0, 0.5),
                                            dBmin=-40.0)
    uv_metrics = compute_final_uv_metrics(filepath)
    write_extreme_done_marker(filepath, params, saveN)

    return Dict{String, Any}(
        "simulation_status" => "complete",
        "output_h5" => filepath,
        "propagation_png" => propagation_png,
        "propagation_zoom_2cm_png" => propagation_zoom_2cm_png,
        "spectral_propagation_zoom_2cm_png" => spectral_propagation_zoom_2cm_png,
        "propagation_zoom_10cm_png" => propagation_zoom_10cm_png,
        "spectral_propagation_zoom_10cm_png" => spectral_propagation_zoom_10cm_png,
        "propagation_highres_png" => propagation_highres_png,
        "spectral_propagation_highres_png" => spectral_propagation_highres_png,
        "spectrum_png" => spectrum_png,
        "spectrum_db_png" => spectrum_db_png,
        "time_png" => time_png,
        "uv_fraction_200_700" => uv_metrics["uv_fraction_200_700"],
        "uv_peak_200_700" => uv_metrics["uv_peak_200_700"],
        "final_peak_wavelength_nm" => uv_metrics["final_peak_wavelength_nm"]
    )
end

function csv_cell(v)
    if v isa AbstractString
        return "\"" * replace(v, "\"" => "\"\"") * "\""
    elseif v isa Bool
        return v ? "true" : "false"
    elseif v isa Number
        return string(v)
    else
        return "\"" * replace(string(v), "\"" => "\"\"") * "\""
    end
end

function write_extreme_summary_csv(path, rows)
    columns = [
        "candidate_id", "simulation_rank", "energy_uJ", "tau_fs", "pressure_bar",
        "diameter_um", "chirp", "precheck_status", "simulation_status",
        "field_limit_exceeded", "strong_ionization_candidate", "E0_GV_m",
        "dataset_E_limit_V_m", "E0_over_dataset_limit", "E0_over_Emax_ppt",
        "gamma_K", "P_ratio", "N", "gamma", "Aeff", "beta2", "ionization_score",
        "uv_prior_score", "combined_score", "uv_fraction_200_700",
        "uv_peak_200_700", "final_peak_wavelength_nm", "output_h5",
        "propagation_png", "propagation_zoom_2cm_png", "spectral_propagation_zoom_2cm_png",
        "propagation_zoom_10cm_png", "spectral_propagation_zoom_10cm_png",
        "propagation_highres_png", "spectral_propagation_highres_png",
        "spectrum_png", "spectrum_db_png", "time_png", "failure_reason"
    ]
    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            values = [csv_cell(get(row, col, "")) for col in columns]
            println(io, join(values, ","))
        end
    end
end

function sortable_metric(row, key)
    v = get(row, key, -Inf)
    return (v isa Number && isfinite(v)) ? Float64(v) : -Inf
end

function search_extreme_samples(; n_candidates, n_run_top, run_all_runnable=false,
                                resume=true, repair_corrupt=true, overwrite=false,
                                output_dir, ranges, steps,
                                seed=42, wallthickness_um=0.65, λ0=1030e-9,
                                λlims_sim=(200e-9, 2500e-9), trange=500e-15,
                                flength=0.5, gas=:Ar, chirp=0.0)
    mkpath(output_dir)
    wallthickness_m = wallthickness_um * 1e-6
    save_interval = 0.05
    saveN = max(round(Int, flength * 100 / save_interval) + 1, 11)

    println(repeat("=", 70))
    println("Extreme UV / ionization sample search")
    simulation_plan = run_all_runnable ? "all runnable" : string(n_run_top)
    println("Candidate cap: $(n_candidates == 0 ? "full grid" : string(n_candidates)), full simulations: $simulation_plan, output: $output_dir")
    println("Resume: $resume, repair corrupt: $repair_corrupt, overwrite: $overwrite")
    println("Search ranges: $ranges")
    println("Search steps: $steps")
    println(repeat("=", 70))

    candidates = generate_extreme_candidate_parameters(n_candidates; ranges=ranges,
                                                       steps=steps,
                                                       seed=seed, chirp=chirp)
    println("Generated $(length(candidates)) candidate parameter sets")
    rows = Dict{String, Any}[]
    for params in candidates
        row = score_extreme_candidate(params; λ0, λlims_sim, trange,
                                      gas, wallthickness_m, flength)
        push!(rows, row)
    end

    runnable = filter(row -> get(row, "precheck_status", "") == "candidate_ok", rows)
    hard_limited_count = count(row -> get(row, "field_limit_exceeded", false), rows)
    strong_ionization_count = count(row -> get(row, "strong_ionization_candidate", false), rows)
    precheck_failed_count = count(row -> get(row, "precheck_status", "") == "precheck_failed", rows)
    println("Precheck summary: runnable=$(length(runnable)), hard field-limit exceeded=$hard_limited_count, " *
            "strong-ionization candidates=$strong_ionization_count, failed=$precheck_failed_count")
    ranked = sort(runnable; by=row -> sortable_metric(row, "combined_score"), rev=true)
    to_run = run_all_runnable ? ranked : ranked[1:min(n_run_top, length(ranked))]
    println("Selected $(length(to_run)) candidate(s) for full propagation")

    for (rank, row) in enumerate(to_run)
        row["simulation_rank"] = rank
        println("\n[$rank/$(length(to_run))] Running candidate $(row["candidate_id"]): " *
                "E=$(round(row["energy_uJ"], digits=3)) uJ, " *
                "tau=$(round(row["tau_fs"], digits=3)) fs, " *
                "P=$(round(row["pressure_bar"], digits=3)) bar, " *
                "d=$(round(row["diameter_um"], digits=3)) um")
        try
            run_result = run_extreme_sample(row, rank, output_dir;
                                            λ0, λlims_sim, trange, gas,
                                            wallthickness_um, wallthickness_m,
                                            flength, saveN,
                                            resume=resume,
                                            repair_corrupt=repair_corrupt,
                                            overwrite=overwrite)
            for (key, value) in run_result
                row[key] = value
            end
            if !haskey(row, "simulation_status") || row["simulation_status"] == "not_run"
                row["simulation_status"] = "complete"
            end
            if row["simulation_status"] in ("complete", "skipped_complete")
                println("  Status: $(row["simulation_status"])")
                println("  UV fraction 200-700 nm: $(row["uv_fraction_200_700"])")
            else
                println("  Status: $(row["simulation_status"])")
            end
        catch e
            row["simulation_status"] = "simulation_failed"
            row["failure_reason"] = sprint(showerror, e)
            println("  WARNING: candidate failed: $(row["failure_reason"])")
            try
                PyPlot.close("all")
            catch
            end
        end
    end

    for row in rows
        if !haskey(row, "simulation_rank")
            row["simulation_rank"] = ""
        end
        if !haskey(row, "failure_reason")
            row["failure_reason"] = ""
        end
    end

    write_extreme_summary_csv(joinpath(output_dir, "summary.csv"), rows)
    uv_ranked = sort(rows; by=row -> sortable_metric(row, "uv_fraction_200_700"), rev=true)
    ion_ranked = sort(rows; by=row -> sortable_metric(row, "ionization_score"), rev=true)
    write_extreme_summary_csv(joinpath(output_dir, "summary_ranked_by_uv.csv"), uv_ranked)
    write_extreme_summary_csv(joinpath(output_dir, "summary_ranked_by_ionization.csv"), ion_ranked)

    println("\nExtreme sample search complete.")
    println("  Summary: $(joinpath(output_dir, "summary.csv"))")
    println("  Ranked by UV: $(joinpath(output_dir, "summary_ranked_by_uv.csv"))")
    println("  Ranked by ionization: $(joinpath(output_dir, "summary_ranked_by_ionization.csv"))")
    return rows
end

function main()
    args = parse_command_line_args()
    validate_parameters(args)

    WALL_THICKNESS_UM = args["wall-thickness"]
    diameter = args["diameter"]
    energy_μJ = args["energy"]
    τfwhm_fs = args["tau"]
    pressure = args["pressure"]
    C = args["chirp"]
    output_file = args["output"]
    export_fiber_model = args["export-fiber-model"]

    FIXED_FLENGTH = 0.5
    gas = :Ar
    λ0 = 1030e-9
    λlims_sim = (200e-9, 2500e-9)
    trange = 500e-15

    energy = energy_μJ * 1e-6
    τfwhm = τfwhm_fs * 1e-15
    radius = diameter / 2 * 1e-6
    flength = FIXED_FLENGTH
    wallthickness_m = WALL_THICKNESS_UM * 1e-6

    if args["search-extreme-samples"]
        search_ranges = Dict{String, Tuple{Float64, Float64}}(
            "energy" => (args["search-energy-min"], args["search-energy-max"]),
            "tau" => (args["search-tau-min"], args["search-tau-max"]),
            "pressure" => (args["search-pressure-min"], args["search-pressure-max"]),
            "diameter" => (args["search-diameter-min"], args["search-diameter-max"])
        )
        search_steps = Dict{String, Float64}(
            "energy" => args["search-energy-step"],
            "tau" => args["search-tau-step"],
            "pressure" => args["search-pressure-step"],
            "diameter" => args["search-diameter-step"]
        )
        search_extreme_samples(
            n_candidates=args["search-candidates"],
            n_run_top=args["search-run-top"],
            run_all_runnable=args["search-run-all-runnable"],
            resume=args["search-resume"],
            repair_corrupt=args["search-repair-corrupt"],
            overwrite=args["search-overwrite"],
            output_dir=args["search-output-dir"],
            ranges=search_ranges,
            steps=search_steps,
            seed=args["search-seed"],
            wallthickness_um=WALL_THICKNESS_UM,
            λ0=λ0,
            λlims_sim=λlims_sim,
            trange=trange,
            flength=flength,
            gas=gas,
            chirp=C
        )
        return
    end

    println(repeat("=", 70))
    println("反谐振光纤单次模拟")
    println(repeat("=", 70))
    println("管壁厚度: $(WALL_THICKNESS_UM) μm")
    println("纤芯直径: $(diameter) μm")
    println("脉冲能量: $(energy_μJ) μJ")
    println("脉冲宽度: $(τfwhm_fs) fs")
    println("气压: $(pressure) bar")
    println("啁啾参数: $(C)")
    println("光纤长度: $(flength*100) cm (固定)")
    println("输出文件: $(output_file)")
    if export_fiber_model
        println("光纤模型导出目录: $(args["export-dir"])")
    end
    println(repeat("=", 70))

    println("\n初始化反谐振光纤模式...")
    mode = Antiresonant.ZeisbergerMode(radius, gas, pressure;
                                       wallthickness=wallthickness_m,
                                       model=:full, loss=true)
    println("模式初始化完成")

    if export_fiber_model
        export_metadata = Dict{String, Any}(
            "gas" => string(gas),
            "pressure_bar" => pressure,
            "core_radius_m" => radius,
            "core_diameter_um" => diameter,
            "wall_thickness_m" => wallthickness_m,
            "wall_thickness_um" => WALL_THICKNESS_UM,
            "lambda0_nm" => λ0 * 1e9,
            "pulse_energy_uJ" => energy_μJ,
            "pulse_tau_fs" => τfwhm_fs,
            "chirp" => C,
            "fiber_length_m" => flength,
            "simulation_lambda_min_nm" => λlims_sim[1] * 1e9,
            "simulation_lambda_max_nm" => λlims_sim[2] * 1e9,
            "export_step_nm" => args["export-step-nm"],
            "export_resonance_step_nm" => args["export-resonance-step-nm"],
            "export_resonance_window_nm" => args["export-resonance-window-nm"]
        )
        mkpath(args["export-dir"])
        export_antiresonant_fiber_model(mode, args["export-dir"];
                                        format="mat",
                                        metadata=export_metadata)
    end

    println("\n生成色散曲线...")
    plot_dispersion_curve(mode, gas, pressure, WALL_THICKNESS_UM, "anti_resonant")

    save_interval = 0.05
    flength_cm = flength * 100
    saveN = max(round(Int, flength_cm / save_interval) + 1, 11)
    println("保存点数量: $(saveN)")

    τ0 = τfwhm / (2 * log(1 + sqrt(2)))
    gdd = C * τ0^2
    ϕ_vec = Float64[0.0, 0.0, gdd]

    println("\n生成输入脉冲图像...")
    plot_input_pulse(Grid.RealGrid(flength, λ0, λlims_sim, trange),
                     τfwhm, energy, λ0, ϕ_vec, :sech, "anti_resonant")

    aeff = Modes.Aeff(mode)
    τ0 = τfwhm / (2 * log(1 + sqrt(2)))
    P0 = energy / (τ0 * sqrt(π))
    E0 = sqrt(2 * P0 / (π * aeff))
    I_p = PhysData.ionisation_potential(gas)
    Emax_ppt = 2 * barrier_suppression_field(I_p, 1)
    println("\n峰值电场强度: E₀ = $(round(E0/1e9, digits=2)) GV/m")
    println("PPT 电离率最大场强: E_max = $(round(Emax_ppt/1e9, digits=2)) GV/m")
    if E0 > Emax_ppt
        println("="^70)
        println("ERROR: 峰值电场强度 E₀ 超过 PPT 电离率模型的最大场强限制！")
        println("  当前 E₀ = $(round(E0/1e9, digits=2)) GV/m")
        println("  PPT 限制 E_max = $(round(Emax_ppt/1e9, digits=2)) GV/m")
        println("  比值 E₀/E_max = $(round(E0/Emax_ppt, digits=2))")
        println("")
        println("可能的原因：")
        println("  1. 脉冲能量过高（当前: $(energy_μJ) μJ）")
        println("  2. 纤芯直径过小（当前: $(diameter) μm）")
        println("  3. 脉冲宽度过窄（当前: $(τfwhm_fs) fs）")
        println("")
        println("建议的解决方案：")
        println("  a) 降低脉冲能量（例如: -e 0.3）")
        println("  b) 增大纤芯直径（例如: -d 30 或更大）")
        println("  c) 增大脉冲宽度（例如: -τ 30）")
        println("  d) 关闭等离子体电离模型（修改 plasma=false）")
        println("="^70)
        error("场强超限: E₀ > E_max(PPT)，请调整输入参数")
    end

    safe_rm(output_file)

    println("\n开始模拟（使用ZeisbergerMode反谐振光纤模型）...")
    output = prop_antiresonant(radius, flength, gas, pressure, wallthickness_m;
                              λ0, τfwhm, energy,
                              λlims=λlims_sim, trange,
                              ϕ=ϕ_vec,
                              pulseshape=:sech,
                              model=:full,
                              loss=true,
                              raman=false,
                              plasma=true,
                              saveN=saveN,
                              filepath=output_file)

    println("模拟完成！结果已保存到: $(output_file)")

    println("\n计算物理特征...")
    grid = Grid.RealGrid(flength, λ0, λlims_sim, trange)
    features = calculate_physics_features(mode, grid, λ0, τfwhm, energy, pressure, gas)

    println("保存物理特征...")
    h5open(output_file, "r+") do file
        if !haskey(file, "physics_features")
            g = create_group(file, "physics_features")
            for (key, val) in features
                if val isa Number
                    g[key] = val
                end
            end
            g["wallthickness"] = wallthickness_m
            g["fibre_model"] = "antiresonant"
            g["chirp_enabled"] = (C != 0.0)
        end

        if !haskey(file, "input")
            inp = create_group(file, "input")
            inp["energy"] = energy_μJ
            inp["tau"] = τfwhm_fs
            inp["pressure"] = pressure
            inp["diameter"] = diameter
            inp["wallthickness"] = WALL_THICKNESS_UM
            inp["chirp"] = C
            inp["flength"] = flength
        end
    end

    println("\n物理特征:")
    println("  β2: $(features["beta2"]) s²/m")
    println("  γ: $(features["gamma"]) W⁻¹m⁻¹")
    println("  N: $(features["N"])")
    println("  L0: $(features["L0"]) m")
    println("  γ_K: $(features["gamma_K"])")
    println("  P_ratio: $(features["P_ratio"])")
    println("  Aeff: $(features["Aeff"]) m²")
    println("  neff: $(features["neff"])")
    println("  P0: $(features["P0"]*1e6) μW")

    println("\n生成可视化结果...")
    try
        Plotting.prop_2D(output, :λ, dBmin=-40.0,
                         λrange=(50e-9, 3000e-9),
                         trange=(-100e-15, 100e-15))
        safe_savefig("anti_resonant_propagation.png")
        println("已保存传播图: anti_resonant_propagation.png")

        plot_propagation_zoom(output, "anti_resonant_propagation_z0_2cm.png";
                              λrange=(200e-9, 2500e-9),
                              zrange_cm=(0.0, 2.0),
                              dBmin=-40.0)
        println("已保存传播局部放大图: anti_resonant_propagation_z0_2cm.png")

        plot_spectral_propagation_zoom(output, "anti_resonant_spectral_propagation_z0_2cm.png";
                                       λrange=(200e-9, 2500e-9),
                                       zrange_cm=(0.0, 2.0),
                                       dBmin=-40.0)
        println("已保存频域传播局部放大图: anti_resonant_spectral_propagation_z0_2cm.png")

        plot_propagation_zoom(output, "anti_resonant_propagation_z0_10cm.png";
                              λrange=(200e-9, 2500e-9),
                              zrange_cm=(0.0, 10.0),
                              dBmin=-40.0)
        println("已保存传播局部放大图: anti_resonant_propagation_z0_10cm.png")

        plot_spectral_propagation_zoom(output, "anti_resonant_spectral_propagation_z0_10cm.png";
                                       λrange=(200e-9, 2500e-9),
                                       zrange_cm=(0.0, 10.0),
                                       dBmin=-40.0)
        println("已保存频域传播局部放大图: anti_resonant_spectral_propagation_z0_10cm.png")

        highres_output = run_early_highres_propagation(radius, gas, pressure, wallthickness_m;
                                                       λ0, τfwhm, energy,
                                                       ϕ_vec=ϕ_vec,
                                                       λlims=λlims_sim,
                                                       trange=trange,
                                                       saveN=EARLY_HIGHRES_SAVEN)
        visualize_early_highres_propagation(highres_output, "anti_resonant";
                                            λrange=λlims_sim,
                                            zrange_cm=(0.0, 0.5),
                                            dBmin=-40.0)
        println("已保存高密度早期传播图: anti_resonant_propagation_z0_0p5cm_highres.png")
        println("已保存高密度早期频域传播图: anti_resonant_spectral_propagation_z0_0p5cm_highres.png")

        Plotting.spec_1D(output, range(0.0, flength, length=5),
                         λrange=(50e-9, 3000e-9))
        safe_savefig("anti_resonant_spectrum.png")
        println("已保存光谱图: anti_resonant_spectrum.png")

        plot_final_spectrum_db(output_file, "anti_resonant_spectrum_db.png";
                               λrange=(200e-9, 2500e-9),
                               dBmin=-40.0)
        println("已保存最终光谱 dB 图: anti_resonant_spectrum_db.png")

        Plotting.time_1D(output, range(0.0, flength, length=5))
        safe_savefig("anti_resonant_time.png")
        println("已保存时间域图: anti_resonant_time.png")

        println("\n所有可视化结果已保存！")
    catch e
        println("警告: 可视化失败: $(e)")
        println("这可能由于缺少PyPlot依赖项。")
    end

    println("\n" * repeat("=", 70))
    println("模拟完成！")
    println("输出文件: $(output_file)")
    println(repeat("=", 70))
end

main()
