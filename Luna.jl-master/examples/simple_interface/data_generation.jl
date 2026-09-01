using Luna
using Luna.PhysData
using Luna.Modes
using Luna.LinearOps
using Luna.Capillary
using Luna.Grid
using Statistics
using Random
using Dates
using HDF5
import FFTW
using PyPlot
import Luna.Interface

function manual_diff(y, x)
    n = length(y)
    d = similar(y, n - 1)
    for i in 1:(n - 1)
        d[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    end
    return d
end

function find_resonance_wavelengths(wallthickness_um, ncl=1.45, nco=1.0;
                                     λ_min_nm=PLOT_λLIMS_NM[1],
                                     λ_max_nm=PLOT_λLIMS_NM[2],
                                     max_order=20)
    wt_m = wallthickness_um * 1e-6
    resonance_orders = collect(1:max_order)
    λ_res_nm = 2.0 .* wt_m .* sqrt(ncl^2 - nco^2) ./ resonance_orders .* 1e9
    λ_res_nm = λ_res_nm[λ_min_nm .< λ_res_nm .< λ_max_nm]
    return sort(λ_res_nm, rev=true)
end

function find_anti_resonant_windows(wallthickness_um, ncl=1.45, nco=1.0;
                                     λ_min_nm=PLOT_λLIMS_NM[1],
                                     λ_max_nm=PLOT_λLIMS_NM[2],
                                     margin_nm=3.0)
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
        log_message("已删除旧文件: $(filepath)")
    end
    PyPlot.savefig(filepath, dpi=300, bbox_inches="tight")
end

function safe_rm(filepath)
    if isfile(filepath)
        rm(filepath; force=true)
        log_message("已删除旧文件: $(filepath)")
    end
end

function compute_window_dispersion(mode, λ_min_nm, λ_max_nm; min_points=30, debug=false)
    if λ_max_nm - λ_min_nm < 10.0
        return Float64[], Float64[], Float64[], Float64[]
    end

    n_points = max(min_points, round(Int, (λ_max_nm - λ_min_nm) / 0.3))
    n_points = min(n_points, 5000)

    λs = range(λ_min_nm * 1e-9, λ_max_nm * 1e-9, length=n_points)
    ωs = 2π * PhysData.c ./ λs
    neffs = [Modes.neff(mode, ω) for ω in ωs]

    real_neff = real.(neffs)
    nan_mask = isnan.(real_neff) .| isinf.(real_neff)
    n_nan = count(nan_mask)
    if debug
        log_message("    窗口 [$(round(λ_min_nm,digits=1)), $(round(λ_max_nm,digits=1))] nm: total=$n_points, invalid=$n_nan")
    end
    if any(nan_mask)
        good_idx = findall(.!nan_mask)
        if length(good_idx) > 10
            real_neff = interpolate_nan(ωs, real_neff, nan_mask, good_idx)
        end
    end

    valid = .!isnan.(real_neff) .& .!isinf.(real_neff)
    valid_idx = findall(valid)
    if length(valid_idx) < (min_points ÷ 2)
        return Float64[], Float64[], Float64[], Float64[]
    end

    ωs_clean = ωs[valid_idx]
    real_neff_clean = real_neff[valid_idx]
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

        D_poly = -2π * c ./ (eval_ω .^ 2) .* β2_poly .* 1e9
        D_use = D_poly
        β2_use = β2_poly .* 1e27
        λ_use = eval_λ
        ω_loss = eval_ω
    else
        D_use = D_fd
        β2_use = β2_fd .* 1e27
        λ_use = λ_final_fd
        ω_loss = ω_final_fd
    end

    loss_dbm = zeros(Float64, length(ω_loss))
    for i in eachindex(ω_loss)
        loss_dbm[i] = Modes.dB_per_m(mode, ω_loss[i])
    end

    finite = isfinite.(D_use) .& isfinite.(β2_use) .& isfinite.(loss_dbm)
    return λ_use[finite], D_use[finite], β2_use[finite], loss_dbm[finite]
end

# ============================================================================
# 可视化辅助函数（用于首样本的色散与脉冲展示）
# ============================================================================

function plot_dispersion_curve(mode, gas, pressure, wallthickness_um, output_prefix="dispersion")
    windows, λ_res_nm = find_anti_resonant_windows(wallthickness_um)

    all_λ = Float64[]
    all_D = Float64[]
    all_β2 = Float64[]
    all_loss = Float64[]

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
    else
        error("No finite dispersion data available for anti-resonant fibre plotting.")
    end

    try
        n_effective_windows = length(λ_res_nm) + 1
        λ0_nm = 1030.0
        gas_str = gas isa Symbol ? string(gas) : string(gas)

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
            PyPlot.ylim(-y_range_D, y_range_D)
            label_y = y_range_D * 0.7
            text_y = -y_range_D * 0.7
        else
            label_y = 0.5
            text_y = -0.5
        end
        PyPlot.text(λ0_nm + 5, label_y, "λ₀=1030 nm", fontsize=9, color="gray")

        for λr in λ_res_nm
            if PLOT_λLIMS_NM[1] < λr < PLOT_λLIMS_NM[2]
                PyPlot.axvline(x=λr, color="blue", linestyle="--",
                               linewidth=0.6, alpha=0.55)
                wt_m = wallthickness_um * 1e-6
                m_order = round(Int, 2 * wt_m * sqrt(1.45^2 - 1.0^2) * 1e9 / λr)
                PyPlot.text(λr + 2, text_y,
                           "m=$m_order",
                           fontsize=7.5, color="blue", alpha=0.6, rotation=90)
            end
        end

        PyPlot.xlim(PLOT_λLIMS_NM...)
        PyPlot.tight_layout()
        d_png_path = "$(output_prefix)_dispersion_D.png"
        safe_savefig(d_png_path)
        PyPlot.close()
        log_message("色散系数 D 图像已保存: $(d_png_path)")

        PyPlot.figure(figsize=(16, 8))
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
        PyPlot.yticks([-10, -5, 0, 5, 10], ["-10", "-5", "0", "5", "10"])
        PyPlot.minorticks_on()
        PyPlot.text(λ0_nm + 5, 7.0, "λ₀=1030 nm", fontsize=9, color="gray")

        for λr in λ_res_nm
            if PLOT_λLIMS_NM[1] < λr < PLOT_λLIMS_NM[2]
                PyPlot.axvline(x=λr, color="blue", linestyle="--",
                               linewidth=0.6, alpha=0.55)
                wt_m = wallthickness_um * 1e-6
                m_order = round(Int, 2 * wt_m * sqrt(1.45^2 - 1.0^2) * 1e9 / λr)
                PyPlot.text(λr + 2, -8.5,
                           "m=$m_order",
                           fontsize=7.5, color="blue", alpha=0.6, rotation=90)
            end
        end

        PyPlot.xlim(PLOT_λLIMS_NM...)
        PyPlot.tight_layout()
        b2_png_path = "$(output_prefix)_dispersion_beta2.png"
        safe_savefig(b2_png_path)
        PyPlot.close()
        log_message("群速度色散 β₂ 图像已保存: $(b2_png_path)")

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
            PyPlot.text(λ0_nm + 5, loss_mid, "λ₀=1030 nm", fontsize=9, color="gray")

            for λr in λ_res_nm
                if PLOT_λLIMS_NM[1] < λr < PLOT_λLIMS_NM[2]
                    PyPlot.axvline(x=λr, color="blue", linestyle="--",
                                   linewidth=0.6, alpha=0.55)
                    wt_m = wallthickness_um * 1e-6
                    m_order = round(Int, 2 * wt_m * sqrt(1.45^2 - 1.0^2) * 1e9 / λr)
                    PyPlot.text(λr + 2, minimum(loss_plot) * 1.5,
                               "m=$m_order",
                               fontsize=7.5, color="blue", alpha=0.6, rotation=90)
                end
            end

            PyPlot.xlim(PLOT_λLIMS_NM...)
            PyPlot.tight_layout()
            loss_png_path = "$(output_prefix)_loss.png"
            safe_savefig(loss_png_path)
            PyPlot.close()
            log_message("损耗曲线已保存: $(loss_png_path)")
        else
            log_message("警告: 无有效损耗数据，跳过损耗曲线")
        end

        log_message("反谐振窗口: $n_effective_windows 个, 有效数据点: $(length(all_λ))")
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
        pf = pulseshape == :sech ? Fields.SechField(; λ0, energy, τfwhm, ϕ=ϕ_vec) :
             Fields.GaussField(; λ0, energy, τfwhm, ϕ=ϕ_vec)
        Et = Fields.make_Et(pf, grid)
        t = grid.t
        ω0 = PhysData.wlfreq(λ0)
        It = abs2.(Et)
        It = It ./ maximum(It)
        dt = t[2] - t[1]
        n_orig = length(Et)
        Eω_fft = abs2.(FFTW.rfft(Et))
        freqs = FFTW.rfftfreq(n_orig, 1/dt)
        freq0_THz = ω0 / (2π) * 1e-12
        freqs_THz = freqs * 1e-12
        main_idx = freqs_THz .<= freq0_THz * 1.1
        fig, (ax1, ax2) = PyPlot.subplots(1, 2, figsize=(14, 5))
        ax1.plot(t * 1e15, It, color="#1f77b4", linewidth=1.2)
        ax1.set_xlabel("Time (fs)", fontsize=12)
        ax1.set_ylabel("Normalized Intensity", fontsize=12)
        ax1.set_title("Time Domain", fontsize=12)
        ax1.grid(true, alpha=0.3)
        ax2.plot((freqs_THz .- freq0_THz)[main_idx], (Eω_fft ./ maximum(Eω_fft))[main_idx],
                 color="#ff7f0e", linewidth=1.2)
        ax2.set_xlabel("Frequency Offset (THz)", fontsize=12)
        ax2.set_ylabel("Normalized Spectral Intensity", fontsize=12)
        ax2.set_title("Frequency Domain", fontsize=12)
        ax2.grid(true, alpha=0.3)
        PyPlot.suptitle("Input Pulse at λ₀ = $(round(λ0*1e9, digits=1)) nm, " *
                        "τ = $(round(τfwhm*1e15, digits=1)) fs, " *
                        "E = $(round(energy*1e6, digits=3)) μJ", fontsize=13)
        PyPlot.tight_layout()
        PyPlot.savefig("$(output_prefix)_pulse.png", dpi=300, bbox_inches="tight")
        PyPlot.savefig("$(output_prefix)_pulse.pdf", bbox_inches="tight")
        PyPlot.close()
        log_message("脉冲图像已保存: $(output_prefix)_pulse.png /.pdf")
    catch e
        log_message("警告: 脉冲图像绘制失败: $(e)")
    end
end

# ============================================================================
# 命令行参数解析
# ============================================================================

function parse_command_line_args()
    args = Dict{String, Any}()
    
    # 默认值
    args["small-batch"] = false
    args["fibre-model"] = "antiresonant"
    args["wall-thickness"] = 0.65
    args["chirp-enabled"] = false
    args["base-dir"] = "training_data"
    args["resume"] = true
    args["repair-corrupt"] = true
    args["overwrite"] = false
    args["startup-integrity-scan"] = false
    args["early-dense-mode"] = "none"
    args["early-dense-zmax-cm"] = 0.5
    args["early-dense-saveN"] = 201
    args["early-dense-output-suffix"] = "_earlydense"
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        
        if arg == "--small-batch" || arg == "-s"
            args["small-batch"] = true
        elseif arg == "--fibre-model" || arg == "-f"
            if i + 1 <= length(ARGS)
                args["fibre-model"] = ARGS[i+1]
                i += 1
            else
                error("--fibre-model 需要一个参数值: capillary 或 antiresonant")
            end
        elseif arg == "--wall-thickness" || arg == "-t"
            if i + 1 <= length(ARGS)
                try
                    wt = parse(Float64, ARGS[i+1])
                    args["wall-thickness"] = wt
                    i += 1
                catch
                    error("--wall-thickness 需要一个有效的数值参数")
                end
            else
                error("--wall-thickness 需要一个数值参数 (单位: μm)")
            end
        elseif arg == "--chirp" || arg == "-c"
            args["chirp-enabled"] = true
        elseif arg == "--base-dir" || arg == "-b"
            if i + 1 <= length(ARGS)
                args["base-dir"] = ARGS[i+1]
                i += 1
            else
                error("--base-dir 需要一个目录名称参数")
            end
        elseif arg == "--resume"
            if i + 1 <= length(ARGS)
                args["resume"] = parse_bool_arg(ARGS[i+1], "--resume")
                i += 1
            else
                error("--resume 需要 true 或 false")
            end
        elseif arg == "--repair-corrupt"
            if i + 1 <= length(ARGS)
                args["repair-corrupt"] = parse_bool_arg(ARGS[i+1], "--repair-corrupt")
                i += 1
            else
                error("--repair-corrupt 需要 true 或 false")
            end
        elseif arg == "--overwrite"
            if i + 1 <= length(ARGS)
                args["overwrite"] = parse_bool_arg(ARGS[i+1], "--overwrite")
                i += 1
            else
                error("--overwrite 需要 true 或 false")
            end
        elseif arg == "--startup-integrity-scan"
            if i + 1 <= length(ARGS)
                args["startup-integrity-scan"] = parse_bool_arg(ARGS[i+1], "--startup-integrity-scan")
                i += 1
            else
                error("--startup-integrity-scan 需要 true 或 false")
            end
        elseif arg == "--early-dense-mode"
            if i + 1 <= length(ARGS)
                mode = lowercase(strip(ARGS[i+1]))
                if !(mode in ("none", "direct", "restitch"))
                    error("--early-dense-mode must be one of: none, direct, restitch")
                end
                args["early-dense-mode"] = mode
                i += 1
            else
                error("--early-dense-mode requires a value: none, direct, or restitch")
            end
        elseif arg == "--early-dense-zmax-cm"
            if i + 1 <= length(ARGS)
                args["early-dense-zmax-cm"] = parse(Float64, ARGS[i+1])
                i += 1
            else
                error("--early-dense-zmax-cm requires a numeric value")
            end
        elseif arg == "--early-dense-saveN"
            if i + 1 <= length(ARGS)
                args["early-dense-saveN"] = parse(Int, ARGS[i+1])
                i += 1
            else
                error("--early-dense-saveN requires an integer value")
            end
        elseif arg == "--early-dense-output-suffix"
            if i + 1 <= length(ARGS)
                args["early-dense-output-suffix"] = ARGS[i+1]
                i += 1
            else
                error("--early-dense-output-suffix requires a string value")
            end
        elseif arg == "--help" || arg == "-h"
            print_help()
            exit(0)
        else
            println("警告: 未知参数 '$arg'，已忽略")
            flush(stdout)
        end
        
        i += 1
    end
    
    return args
end

function parse_bool_arg(value::String, flag_name::String)
    value_lc = lowercase(strip(value))
    if value_lc in ("true", "1", "yes", "y", "on")
        return true
    elseif value_lc in ("false", "0", "no", "n", "off")
        return false
    else
        error("$flag_name 需要 true 或 false，当前值: $value")
    end
end

function print_help()
    println("""
数据集生成工具 v3.2 - 命令行参数说明

用法: julia data_generation.jl [选项]

选项:
  -s, --small-batch          启用小批量测试模式 (100样本，目录添加_small后缀)
  -f, --fibre-model MODEL    选择光纤模型: capillary 或 antiresonant (默认: antiresonant)
  -t, --wall-thickness VALUE 设置反谐振光纤管壁厚度 (单位: μm, 默认: 0.5)
  -c, --chirp                启用啁啾参数采样 (固定为0)
  -b, --base-dir NAME        设置输出目录基础名 (默认: training_data)
      --resume true/false    断点续生成，跳过完整样本 (默认: true)
      --repair-corrupt true/false
                              移走损坏/不完整样本并重算 (默认: true)
      --overwrite true/false 重新生成已有完整样本 (默认: false)
      --startup-integrity-scan true/false
                              启动时深度扫描全部 HDF5 (默认: false)
  -h, --help                 显示此帮助信息

示例:
  julia data_generation.jl --small-batch
  julia data_generation.jl -f antiresonant -t 0.65 --no-chirp
  julia data_generation.jl -f capillary -s -b my_data

注意事项:
  - 管壁厚度参数仅在 --fibre-model=antiresonant 时生效
  - 管壁厚度有效范围: 0.1 - 5.0 μm
  - 小批量模式生成100个样本，完整模式生成10000个样本
""")
    flush(stdout)
end

# ============================================================================
# 参数验证函数
# ============================================================================

function validate_wall_thickness(wt::Float64)
    if wt <= 0.0
        error("管壁厚度必须大于0 (当前值: $wt μm)")
    end
    if wt < 0.1
        println("警告: 管壁厚度 $wt μm 小于0.1 μm，可能不符合物理实际")
        flush(stdout)
    end
    if wt > 5.0
        println("警告: 管壁厚度 $wt μm 大于5.0 μm，可能不符合物理实际")
        flush(stdout)
    end
    return true
end

function validate_fibre_model(model::String)
    if model != "capillary" && model != "antiresonant"
        error("光纤模型必须是 'capillary' 或 'antiresonant' (当前值: '$model')")
    end
    return Symbol(model)
end

# ============================================================================
# 配置参数
# ============================================================================

# 解析命令行参数
cli_args = parse_command_line_args()

# 1. 小批量测试功能开关
const SMALL_BATCH_MODE = cli_args["small-batch"]

# 2. 光纤模型类型
const FIBRE_MODEL = validate_fibre_model(cli_args["fibre-model"])

# 3. 反谐振光纤管壁厚度 (μm)
const WALL_THICKNESS_UM = cli_args["wall-thickness"]
validate_wall_thickness(WALL_THICKNESS_UM)

# 4. 啁啾参数启用/禁用开关
const CHIRP_ENABLED = cli_args["chirp-enabled"]
const CHIRP_RANGE = (-3.0, 3.0)

# 5. 数据集基础目录名
const BASE_OUTPUT_DIR = cli_args["base-dir"]

const RESUME_ENABLED = cli_args["resume"]
const REPAIR_CORRUPT = cli_args["repair-corrupt"]
const OVERWRITE_EXISTING = cli_args["overwrite"]
const STARTUP_INTEGRITY_SCAN = cli_args["startup-integrity-scan"]
const EARLY_DENSE_MODE = Symbol(cli_args["early-dense-mode"])
const EARLY_DENSE_ZMAX_CM = cli_args["early-dense-zmax-cm"]
const EARLY_DENSE_SAVEN = cli_args["early-dense-saveN"]
const EARLY_DENSE_OUTPUT_SUFFIX = cli_args["early-dense-output-suffix"]

# 6. 光纤长度固定值 (根据模式自动设置)
const FIXED_FLENGTH = if FIBRE_MODEL == :capillary
    1.0  # 毛细管模式: 1.0 m
else
    0.5  # 反谐振模式: 0.5 m
end

# ============================================================================
# 动态目录命名与参数设置
# ============================================================================

function generate_output_dirname()
    parts = [BASE_OUTPUT_DIR]
    
    # 添加模型标识
    if FIBRE_MODEL == :antiresonant
        push!(parts, "ar")
        # 添加管壁厚度信息（保留1位小数）
        wt_str = replace(string(round(WALL_THICKNESS_UM, digits=1)), "." => "p")
        push!(parts, "t$(wt_str)")
    else
        push!(parts, "cap")
    end
    
    # 添加啁啾标识
    if !CHIRP_ENABLED
        push!(parts, "nochirp")
    end
    
    # 小批量模式后缀
    if SMALL_BATCH_MODE
        push!(parts, "small")
    end
    
    return join(parts, "_")
end

output_dir = generate_output_dirname()
if !isdir(output_dir)
    mkdir(output_dir)
end

log_file = joinpath(output_dir, "data_generation.log")

# 根据模式设置样本数量
const TOTAL_SIMULATIONS = SMALL_BATCH_MODE ? 100 : 10000
const BATCH_SIZE = SMALL_BATCH_MODE ? 20 : 100
const START_INDEX = 1
const MAX_RETRIES = 3

const success_log = joinpath(output_dir, "success.log")
const fail_log = joinpath(output_dir, "fail.log")
const corrupt_dir = joinpath(output_dir, "corrupt_samples")

# ============================================================================
# 日志与工具函数
# ============================================================================

function log_message(message)
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    log_entry = "[$timestamp] $message"
    println(log_entry)
    flush(stdout)
    open(log_file, "a") do io
        write(io, log_entry * "\n")
        flush(io)
    end
end

function log_success(index, params_str)
    open(success_log, "a") do io
        write(io, "$index|$params_str\n")
    end
end

function log_failure(index, reason)
    open(fail_log, "a") do io
        write(io, "$index|$reason\n")
    end
end

function sample_filename(index)
    return "sample_$(lpad(index, 6, '0')).h5"
end

function sample_filepath(index)
    return joinpath(output_dir, sample_filename(index))
end

function temp_sample_filepath(index)
    return sample_filepath(index) * ".tmp"
end

function early_dense_filepath(index)
    base = splitext(sample_filename(index))[1]
    return joinpath(output_dir, "$(base)$(EARLY_DENSE_OUTPUT_SUFFIX).h5")
end

function temp_early_dense_filepath(index)
    return early_dense_filepath(index) * ".tmp"
end

function done_marker_path(filepath)
    return filepath * ".done"
end

function has_done_marker(filepath, expected_saveN)
    marker = done_marker_path(filepath)
    if !isfile(filepath) || !isfile(marker)
        return false
    end
    try
        file_size = filesize(filepath)
        content = read(marker, String)
        return occursin("status=complete", content) &&
               occursin("expected_saveN=$(expected_saveN)", content) &&
               occursin("file_size=$(file_size)", content)
    catch
        return false
    end
end

function write_done_marker(filepath, index, expected_saveN)
    marker = done_marker_path(filepath)
    file_size = filesize(filepath)
    completed_at = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    open(marker, "w") do io
        write(io, "status=complete\n")
        write(io, "sample_index=$(index)\n")
        write(io, "expected_saveN=$(expected_saveN)\n")
        write(io, "file_size=$(file_size)\n")
        write(io, "completed_at=$(completed_at)\n")
    end
end

function eω_layout(Eω, z)
    z_len = length(z)
    if ndims(Eω) >= 2 && size(Eω, 2) == z_len
        return :freq_z
    elseif ndims(Eω) >= 2 && size(Eω, 1) == z_len
        return :z_freq
    else
        error("Cannot determine Eω layout for size $(size(Eω)) and z length $z_len")
    end
end

function replace_h5_dataset!(file, name, value)
    if haskey(file, name)
        HDF5.delete_object(file, name)
    end
    file[name] = value
end

function write_early_dense_metadata!(file, original_filepath, early_saveN, early_zmax_cm)
    if !haskey(file, "metadata")
        create_group(file, "metadata")
    end
    meta = file["metadata"]
    for key in ("sampling_mode", "early_dense_source", "early_dense_zmax_cm", "early_dense_saveN")
        if haskey(meta, key)
            HDF5.delete_object(meta, key)
        end
    end
    meta["sampling_mode"] = "early_dense"
    meta["early_dense_source"] = original_filepath
    meta["early_dense_zmax_cm"] = early_zmax_cm
    meta["early_dense_saveN"] = early_saveN
end

function is_complete_sample(filepath, expected_saveN)
    if !isfile(filepath)
        return false
    end

    if has_done_marker(filepath, expected_saveN)
        return true
    end

    try
        h5open(filepath, "r") do f
            required = ("Eω", "z", "physics_features")
            if !all(key -> haskey(f, key), required)
                return false
            end

            z = read(f["z"])
            if length(z) < expected_saveN
                return false
            end

            if haskey(f, "metadata") && haskey(f["metadata"], "status")
                status = read(f["metadata"]["status"])
                return String(status) == "complete"
            end

            return true
        end
    catch e
        return false
    end
end

function move_corrupt_sample(filepath, index, reason)
    if !isfile(filepath)
        return
    end
    if !isdir(corrupt_dir)
        mkpath(corrupt_dir)
    end
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    target = joinpath(corrupt_dir, "$(basename(filepath)).$(stamp).bad")
    mv(filepath, target; force=true)
    log_failure(index, "incomplete_or_corrupt:$reason:moved_to=$(target)")
    log_message("Sample $index: moved incomplete/corrupt file to $target")
end

function mark_sample_complete(filepath, index, expected_saveN)
    h5open(filepath, "r+") do file
        if !haskey(file, "metadata")
            create_group(file, "metadata")
        end
        meta = file["metadata"]
        meta["status"] = "complete"
        meta["sample_index"] = index
        meta["expected_saveN"] = expected_saveN
        meta["completed_at"] = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    end
end

function clean_fftw_cache()
    try
        cache_dir = Luna.Utils.cachedir()
        if isdir(cache_dir)
            files = readdir(cache_dir)
            for file in files
                if startswith(file, "FFTWcache_")
                    filepath = joinpath(cache_dir, file)
                    if isfile(filepath)
                        rm(filepath)
                        log_message("已删除FFTW缓存文件: $filepath")
                    end
                end
            end
        else
            log_message("缓存目录不存在: $cache_dir")
        end
    catch e
        log_message("清理FFTW缓存时出错: $e")
    end
end

function latin_hypercube_sampling(ranges, n_samples)
    n_params = length(ranges)
    samples = zeros(n_samples, n_params)
    
    for i in 1:n_params
        sample_points = LinRange(ranges[i][1], ranges[i][2], n_samples)
        samples[:, i] = shuffle(sample_points)
    end
    
    return samples
end

# ============================================================================
# 参数范围定义
# ============================================================================

function build_param_ranges()
    if FIBRE_MODEL == :capillary
        # 毛细管模型参数范围
        ranges = [
            (log10(1.0), log10(100.0)),  # Energy: 1 - 100 μJ
            (log10(5), log10(50)),       # tau: 5 - 50 fs
        ]
        
        # 啁啾参数
        if CHIRP_ENABLED
            push!(ranges, CHIRP_RANGE)
        else
            push!(ranges, (0.0, 0.0))
        end
        
        # 气压和直径 (长度已固定，不再采样)
        push!(ranges, (0.0, 20.0))      # pressure: 0 - 20 bar
        push!(ranges, (100.0, 300.0))   # diameter: 100 - 300 μm
    else
        # 反谐振模型参数范围 (保持原有配置)
        ranges = [
            (log10(0.3), log10(3.0)),    # Energy: 0.3 - 3 μJ
            (log10(5), log10(50)),       # tau: 5 - 50 fs
        ]
        
        # 啁啾参数
        if CHIRP_ENABLED
            push!(ranges, CHIRP_RANGE)
        else
            push!(ranges, (0.0, 0.0))
        end
        
        # 气压和直径 (长度已固定，不再采样)
        push!(ranges, (0.5, 50.0))      # pressure: 0.5 - 50 bar
        push!(ranges, (10.0, 50.0))   # diameter: 10 - 50 μm
    end
    
    return ranges
end

const λ0 = 1030e-9
const PHYS_λLIMS = (200e-9, 2500e-9)
const PLOT_λLIMS_NM = (50.0, 3000.0)
const PLOT_λRANGE = (50e-9, 3000e-9)
const trange = 500e-15
const save_interval = 0.10
const gas = :Ar
const max_E_threshold = 8.625348297553e10

param_ranges = build_param_ranges()

log_message(repeat("=", 70))
log_message("数据集生成 v3.3 - 固定长度版本")
log_message(repeat("=", 70))
log_message("小批量模式: $(SMALL_BATCH_MODE ? "启用 (100样本)" : "禁用 (10000样本)")")
log_message("光纤模型: $(FIBRE_MODEL == :antiresonant ? "反谐振光纤" : "毛细管")")
if FIBRE_MODEL == :antiresonant
    log_message("管壁厚度: $(WALL_THICKNESS_UM) μm (固定值)")
end
log_message("光纤长度: $(FIXED_FLENGTH) m (根据模式固定)")
log_message("啁啾参数: $(CHIRP_ENABLED ? "启用采样 (范围: $(CHIRP_RANGE))" : "禁用 (固定为0)")")
log_message("纤芯直径范围: $(FIBRE_MODEL == :capillary ? "[100, 300]" : "[10, 50]") μm")
log_message("物理传播波长网格: $(PHYS_λLIMS[1] * 1e9)-$(PHYS_λLIMS[2] * 1e9) nm")
log_message("绘图显示波长范围: $(PLOT_λLIMS_NM[1])-$(PLOT_λLIMS_NM[2]) nm")
log_message("输出目录: $output_dir")
log_message(repeat("=", 70))

log_message("生成拉丁超立方体采样...")
samples = latin_hypercube_sampling(param_ranges, TOTAL_SIMULATIONS)
log_message("采样生成完成，共 $TOTAL_SIMULATIONS 个采样点，$(length(param_ranges))个参数维度")

function calculate_saveN(flength)
    flength_cm = flength * 100
    saveN = round(Int, flength_cm / save_interval) + 1
    return max(saveN, 11)
end

function create_mode(radius, gas, pressure; model=:full, loss=true)
    if FIBRE_MODEL == :antiresonant
        wallthickness_m = WALL_THICKNESS_UM * 1e-6
        return Antiresonant.ZeisbergerMode(radius, gas, pressure; 
                                           wallthickness=wallthickness_m, 
                                           model=model, loss=loss)
    else
        return Capillary.MarcatiliMode(radius, gas, pressure; 
                                       model=model, loss=loss)
    end
end

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
        log_message("β2接近零 ($β2)，使用保护值")
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

function prop_antiresonant(radius, flength, gas, pressure, wallthickness_m;
                          λlims=PHYS_λLIMS, trange=500e-15,
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

function process_sample(index, sample)
    for attempt in 1:MAX_RETRIES
        try
            log_E = sample[1]
            log_τ = sample[2]
            C = sample[3]
            pressure = sample[4]
            diameter = sample[5]
            
            energy = 10^log_E * 1e-6
            τfwhm = 10^log_τ * 1e-15
            radius = diameter / 2 * 1e-6
            flength = FIXED_FLENGTH  # 使用固定长度
            
            if pressure <= 0.0
                log_message("采样点 $index: 压力值无效 ($pressure bar)，跳过")
                log_failure(index, "invalid_pressure=$pressure")
                return false
            end
            
            if diameter <= 0.0
                log_message("采样点 $index: 纤芯直径无效 ($diameter μm)，跳过")
                log_failure(index, "invalid_diameter=$diameter")
                return false
            end
            
            filepath = sample_filepath(index)
            tmp_filepath = temp_sample_filepath(index)
            saveN = calculate_saveN(flength)

            if isfile(tmp_filepath)
                log_message("Sample $index: removing stale temporary file $tmp_filepath")
                rm(tmp_filepath; force=true)
            end

            if isfile(filepath)
                if OVERWRITE_EXISTING
                    log_message("Sample $index: overwrite enabled, removing existing file")
                    rm(filepath; force=true)
                elseif is_complete_sample(filepath, saveN)
                    if !has_done_marker(filepath, saveN)
                        write_done_marker(filepath, index, saveN)
                    end
                    if RESUME_ENABLED
                        log_message("Sample $index: complete file exists, skipping")
                    else
                        log_message("Sample $index: complete file exists; overwrite is false, skipping to avoid data loss")
                    end
                    return :skipped
                elseif REPAIR_CORRUPT
                    move_corrupt_sample(filepath, index, "failed_integrity_check")
                else
                    log_message("Sample $index: existing file is incomplete/corrupt and repair is disabled, skipping")
                    log_failure(index, "incomplete_or_corrupt_repair_disabled")
                    return false
                end
            end
            
            model_str = FIBRE_MODEL == :antiresonant ? "反谐振" : "毛细管"
            chirp_str = CHIRP_ENABLED ? "C=$C" : "C=0(固定)"
            log_message("开始模拟 $index/$TOTAL_SIMULATIONS (尝试 $attempt/$MAX_RETRIES) [模型: $model_str]")
            log_message("参数: E=$(energy*1e6)μJ, τ=$(τfwhm*1e15)fs, $chirp_str, P=$pressure bar, L=$flength m(固定), d=$diameter μm")
            
            if FIBRE_MODEL == :antiresonant
                log_message("  管壁厚度: $(WALL_THICKNESS_UM) μm (固定)")
            end
            
            mode = create_mode(radius, gas, pressure; model=:full, loss=true)

            Aeff = Modes.Aeff(mode)
            τ0 = τfwhm / (2 * log(1 + sqrt(2)))
            P0 = energy / (τ0 * sqrt(π))
            E0 = sqrt(2 * P0 / (π * Aeff))
            
            if E0 > max_E_threshold
                log_message("电场强度过高: $E0 V/m，跳过此模拟")
                log_failure(index, "E0_exceeded=$(E0)")
                return false
            end
            
            log_message("光纤长度: $(flength*100) cm, 保存点数量: $saveN")
            
            gdd = C * τ0^2
            ϕ_vec = Float64[0.0, 0.0, gdd]

            if index == 1
                plot_input_pulse(Grid.RealGrid(flength, λ0, PHYS_λLIMS, trange),
                                 τfwhm, energy, λ0, ϕ_vec, :sech,
                                 "training_data_pulse")
            end

            if FIBRE_MODEL == :antiresonant
                wallthickness_m = WALL_THICKNESS_UM * 1e-6
                output = prop_antiresonant(radius, flength, gas, pressure, wallthickness_m;
                                          λ0, τfwhm, energy,
                                          λlims=PHYS_λLIMS, trange=trange,
                                          ϕ=ϕ_vec,
                                          pulseshape=:sech,
                                          model=:full,
                                          loss=true,
                                          raman=false,
                                          plasma=true,
                                          saveN=saveN,
                                          filepath=tmp_filepath)
            else
                output = prop_capillary(radius, flength, gas, pressure;
                                       λ0, τfwhm, energy,
                                       λlims=PHYS_λLIMS, trange=trange,
                                       ϕ=ϕ_vec,
                                       pulseshape=:sech,
                                       model=:full,
                                       loss=true,
                                       raman=false,
                                       plasma=true,
                                       saveN=saveN,
                                       filepath=tmp_filepath)
            end
            
            grid = Grid.RealGrid(flength, λ0, PHYS_λLIMS, trange)
            mode = create_mode(radius, gas, pressure; model=:full, loss=true)
            features = calculate_physics_features(mode, grid, λ0, τfwhm, energy, pressure, gas)
            
            h5open(tmp_filepath, "r+") do file
                if !haskey(file, "physics_features")
                    g = create_group(file, "physics_features")
                    for key in ("beta2", "gamma", "N", "L0", "gamma_K", "P_ratio", "Aeff", "neff")
                        g[key] = features[key]
                    end
                    g["wallthickness"] = WALL_THICKNESS_UM * 1e-6
                    g["fibre_model"] = string(FIBRE_MODEL)
                    g["chirp_enabled"] = CHIRP_ENABLED
                end
            end

            mark_sample_complete(tmp_filepath, index, saveN)

            if !is_complete_sample(tmp_filepath, saveN)
                log_failure(index, "tmp_failed_integrity_check")
                error("temporary output failed integrity check: $tmp_filepath")
            end

            mv(tmp_filepath, filepath; force=true)
            write_done_marker(filepath, index, saveN)
            if EARLY_DENSE_MODE == :direct
                restitch_early_dense_sample(index, sample)
            end
            
            params_str = "E=$(energy*1e6)μJ,τ=$(τfwhm*1e15)fs,C=$C,P=$pressure,L=$(FIXED_FLENGTH)(fixed),d=$diameter"
            if FIBRE_MODEL == :antiresonant
                params_str *= ",t=$(WALL_THICKNESS_UM)"
            end
            log_success(index, params_str)
            log_message("模拟完成，结果保存到: $filepath")
            return :success
        catch e
            error_msg = string(e)
            bt = catch_backtrace()
            if occursin("failed to import wisdom", error_msg) && attempt < MAX_RETRIES
                log_message("FFTW wisdom导入失败，清理缓存并重试...")
                clean_fftw_cache()
                sleep(2)
            elseif occursin("Field strength", error_msg) && occursin("exceeds maximum", error_msg)
                log_message("采样点 $index: 电场强度超限，永久跳过")
                log_failure(index, "field_strength_exceeded")
                return false
            else
                log_message("采样点 $index 模拟失败: $e")
                log_message("堆栈: $(sprint(showerror, e, bt))")
                if attempt == MAX_RETRIES
                    log_failure(index, "error: $error_msg")
                    return false
                end
                sleep(1)
            end
        end
    end
    return false
end

function restitch_early_dense_sample(index, sample)
    original_filepath = sample_filepath(index)
    output_filepath = early_dense_filepath(index)
    tmp_output = temp_early_dense_filepath(index)
    highres_tmp = output_filepath * ".highres.tmp"

    if !is_complete_sample(original_filepath, calculate_saveN(FIXED_FLENGTH))
        log_message("Early-dense sample $index: original sample is incomplete, skipping")
        return false
    end
    if isfile(output_filepath) && !OVERWRITE_EXISTING
        existing_complete = false
        try
            h5open(output_filepath, "r") do f
                if haskey(f, "metadata") && haskey(f["metadata"], "sampling_mode")
                    if read(f["metadata"]["sampling_mode"]) == "early_dense"
                        existing_complete = true
                    end
                end
            end
            if existing_complete
                log_message("Early-dense sample $index: output exists, skipping")
                return :skipped
            end
        catch
            if REPAIR_CORRUPT
                rm(output_filepath; force=true)
            else
                return false
            end
        end
    end

    rm(tmp_output; force=true)
    rm(highres_tmp; force=true)

    log_E, log_τ, C, pressure, diameter = sample
    energy = 10^log_E * 1e-6
    τfwhm = 10^log_τ * 1e-15
    radius = diameter / 2 * 1e-6
    τ0 = τfwhm / (2 * log(1 + sqrt(2)))
    ϕ_vec = Float64[0.0, 0.0, C * τ0^2]
    early_flength = EARLY_DENSE_ZMAX_CM / 100.0

    if FIBRE_MODEL == :antiresonant
        wallthickness_m = WALL_THICKNESS_UM * 1e-6
        prop_antiresonant(radius, early_flength, gas, pressure, wallthickness_m;
                         λ0, τfwhm, energy,
                         λlims=PHYS_λLIMS, trange=trange,
                         ϕ=ϕ_vec,
                         pulseshape=:sech,
                         model=:full,
                         loss=true,
                         raman=false,
                         plasma=true,
                         saveN=EARLY_DENSE_SAVEN,
                         filepath=highres_tmp)
    else
        prop_capillary(radius, early_flength, gas, pressure;
                      λ0, τfwhm, energy,
                      λlims=PHYS_λLIMS, trange=trange,
                      ϕ=ϕ_vec,
                      pulseshape=:sech,
                      model=:full,
                      loss=true,
                      raman=false,
                      plasma=true,
                      saveN=EARLY_DENSE_SAVEN,
                      filepath=highres_tmp)
    end

    cp(original_filepath, tmp_output; force=true)

    z_new = Float64[]
    Eω_new = nothing
    h5open(highres_tmp, "r") do fh
        h5open(original_filepath, "r") do fo
            zh = read(fh["z"])
            zo = read(fo["z"])
            Eh = read(fh["Eω"])
            Eo = read(fo["Eω"])
            layout = eω_layout(Eo, zo)
            keep_late = zo .> early_flength
            z_new = vcat(zh, zo[keep_late])
            if layout == :freq_z
                Eω_new = cat(Eh, Eo[:, keep_late]; dims=2)
            else
                Eω_new = cat(Eh, Eo[keep_late, :]; dims=1)
            end
        end
    end

    if any(diff(z_new) .<= 0)
        error("Early-dense z axis is not strictly increasing for sample $index")
    end

    h5open(tmp_output, "r+") do f
        replace_h5_dataset!(f, "z", z_new)
        replace_h5_dataset!(f, "Eω", Eω_new)
        write_early_dense_metadata!(f, original_filepath, EARLY_DENSE_SAVEN, EARLY_DENSE_ZMAX_CM)
    end

    mv(tmp_output, output_filepath; force=true)
    write_done_marker(output_filepath, index, length(z_new))
    rm(highres_tmp; force=true)
    log_message("Early-dense sample $index saved to $output_filepath with $(length(z_new)) z points")
    return :success
end

function restitch_all_early_dense_samples(samples)
    log_message("Starting early-dense restitch mode: z<= $(EARLY_DENSE_ZMAX_CM) cm, saveN=$(EARLY_DENSE_SAVEN)")
    success_count = 0
    skip_count = 0
    fail_count = 0
    for i in START_INDEX:TOTAL_SIMULATIONS
        result = try
            restitch_early_dense_sample(i, samples[i, :])
        catch e
            log_message("Early-dense sample $i failed: $e")
            false
        end
        if result === :success
            success_count += 1
        elseif result === :skipped
            skip_count += 1
        else
            fail_count += 1
        end
    end
    log_message("Early-dense restitch complete: success=$success_count skipped=$skip_count failed=$fail_count")
end

function count_existing_files()
    complete_count = 0
    h5_count = 0
    tmp_count = 0
    expected_saveN = calculate_saveN(FIXED_FLENGTH)
    if isdir(output_dir)
        for f in readdir(output_dir)
            if startswith(f, "sample_") && endswith(f, ".h5")
                h5_count += 1
                filepath = joinpath(output_dir, f)
                if has_done_marker(filepath, expected_saveN)
                    complete_count += 1
                elseif STARTUP_INTEGRITY_SCAN && is_complete_sample(filepath, expected_saveN)
                    complete_count += 1
                end
            elseif startswith(f, "sample_") && endswith(f, ".h5.tmp")
                tmp_count += 1
            end
        end
    end
    return h5_count, complete_count, tmp_count
end

function main()
    log_message(repeat("=", 70))
    log_message("数据集生成 v3.3 - 固定长度版本")
    log_message(repeat("=", 70))
    log_message("运行模式: $(SMALL_BATCH_MODE ? "小批量测试" : "完整生成")")
    log_message("光纤模型: $FIBRE_MODEL")
    if FIBRE_MODEL == :antiresonant
        log_message("管壁厚度: $(WALL_THICKNESS_UM) μm (固定值)")
    end
    log_message("光纤长度: $(FIXED_FLENGTH) m (根据模式固定)")
    log_message("啁啾参数: $(CHIRP_ENABLED ? "启用采样" : "禁用 (固定为0)")")
    if FIBRE_MODEL == :capillary
        log_message("纤芯直径范围: [100, 300] μm")
    else
        log_message("纤芯直径范围: [10, 50] μm")
    end
    log_message("目标模拟数量: $TOTAL_SIMULATIONS")
    log_message("批处理大小: $BATCH_SIZE")
    log_message("断点续生成: $RESUME_ENABLED, 修复损坏文件: $REPAIR_CORRUPT, 覆盖已有样本: $OVERWRITE_EXISTING, 启动深度扫描: $STARTUP_INTEGRITY_SCAN")
    log_message("Early-dense mode: $EARLY_DENSE_MODE, zmax=$(EARLY_DENSE_ZMAX_CM) cm, saveN=$EARLY_DENSE_SAVEN")
    if EARLY_DENSE_MODE == :restitch
        restitch_all_early_dense_samples(samples)
        return
    end
    existing_files, existing_complete_count, existing_tmp_count = count_existing_files()
    log_message("已有样本文件数: $existing_files")
    log_message("已有完成标记/完整样本数: $existing_complete_count")
    log_message("已有临时文件数: $existing_tmp_count")
    log_message("输出目录: $output_dir")
    log_message(repeat("=", 70))
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    regenerated_corrupt_count = 0
    new_success_count = 0
    expected_saveN = calculate_saveN(FIXED_FLENGTH)
    
    start_idx = START_INDEX
    end_idx = min(START_INDEX + BATCH_SIZE - 1, TOTAL_SIMULATIONS)
    
    while start_idx <= TOTAL_SIMULATIONS
        log_message("处理批次: $start_idx-$end_idx")
        
        for i in start_idx:end_idx
            sample_path_i = sample_filepath(i)
            existed_without_done_before = isfile(sample_path_i) && !has_done_marker(sample_path_i, expected_saveN)
            result = process_sample(i, samples[i, :])
            if result === :skipped
                skip_count += 1
            elseif result === :success
                success_count += 1
                new_success_count += 1
                if existed_without_done_before
                    regenerated_corrupt_count += 1
                end
            else
                fail_count += 1
            end
        end
        
        start_idx = end_idx + 1
        end_idx = min(start_idx + BATCH_SIZE - 1, TOTAL_SIMULATIONS)
        
        if start_idx <= TOTAL_SIMULATIONS
            log_message("批次完成 - 新成功: $new_success_count, 跳过: $skip_count, 失败: $fail_count, 重生成: $regenerated_corrupt_count, 休息5秒...")
            sleep(5)
        end
    end
    
    log_message(repeat("=", 70))
    log_message("数据生成任务完成！")
    log_message("新成功: $new_success_count, 跳过完整样本: $skip_count, 失败: $fail_count, 重生成损坏/不完整样本: $regenerated_corrupt_count")
    log_message("完成率: $((new_success_count + skip_count)/TOTAL_SIMULATIONS*100)%")
    log_message(repeat("=", 70))
end

main()
