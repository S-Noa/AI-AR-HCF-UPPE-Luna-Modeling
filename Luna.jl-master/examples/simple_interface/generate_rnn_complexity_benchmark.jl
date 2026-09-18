#!/usr/bin/env julia

"""Generate controlled early-propagation AR-HCF trajectories for RNN tests.

The output is intentionally separate from the production training data.  Each
trajectory covers 0--5 cm with 200 saved planes so it matches the sequence
length of the reference SC RNN benchmark while isolating simple and complex
spectral regimes.
"""

using Luna
using Luna.PhysData
using Luna.Modes
using Luna.Grid
using Luna.Interface
using Luna.Stats
using HDF5
using Random
using Printf

const C_LIGHT = PhysData.c

function parse_args()
    values = Dict{String, Any}(
        "class" => "simple",
        "output-dir" => "rnn_complexity_candidates",
        "count" => 1600,
        "start-index" => 1,
        "seed" => 20260915,
        "max-attempts" => 32,
        "overwrite" => false,
    )
    i = 1
    while i <= length(ARGS)
        key = ARGS[i]
        key == "--help" && return nothing
        startswith(key, "--") || error("Unknown argument: $key")
        i += 1
        i <= length(ARGS) || error("$key requires a value")
        value = ARGS[i]
        if key == "--class" || key == "--output-dir"
            values[key[3:end]] = value
        elseif key == "--count" || key == "--start-index" || key == "--seed" || key == "--max-attempts"
            values[key[3:end]] = parse(Int, value)
        elseif key == "--overwrite"
            values["overwrite"] = lowercase(value) in ("1", "true", "yes")
        else
            error("Unknown argument: $key")
        end
        i += 1
    end
    values["class"] in ("simple", "moderate", "complex") || error("--class must be simple, moderate, or complex")
    return values
end

function print_help()
    println("""
Generate controlled AR-HCF RNN benchmark candidates.

Usage:
  julia --project=../.. generate_rnn_complexity_benchmark.jl \\
    --class simple|moderate|complex --output-dir DIR [--count 1600]

Every trajectory uses Ar, t=0.65 um, L=5 cm, 200 saved z planes,
plasma=true, raman=false, loss=true, and shotnoise=false.
""")
end

function ranges_for(label)
    if label == "simple"
        return (energy=(0.3, 1.0), tau=(25.0, 50.0), pressure=(0.5, 15.0), diameter=(30.0, 50.0))
    end
    if label == "moderate"
        # Deliberately retain smooth, weak-to-moderate nonlinear broadening.
        # This bridges the nearly stationary Simple set and the strongly
        # restructuring Complex set without targeting ionisation-dominated maps.
        return (energy=(0.7, 1.6), tau=(15.0, 35.0), pressure=(5.0, 25.0), diameter=(22.0, 40.0))
    end
    # A numerically feasible, still strongly nonlinear subset.  The original
    # extreme proposal (2--3 uJ, 5--12 fs, 20--50 bar, 10--20 um) routinely
    # self-compresses beyond Luna's finite PPT lookup table within 5 cm.
    return (energy=(2.0, 2.5), tau=(10.0, 16.0), pressure=(10.0, 25.0), diameter=(18.0, 25.0))
end

sample_uniform(rng, interval) = interval[1] + rand(rng) * (interval[2] - interval[1])

# Match the initial-field calculation in data_generation.jl.  The 0.5 factor
# leaves headroom for nonlinear self-compression before the PPT lookup reaches
# Luna's tabulated maximum field.
const PPT_FIELD_LIMIT = 8.625348297553e10
const INITIAL_FIELD_SAFETY_RATIO = 0.5

function initial_field_strength(energy_uj, tau_fs, pressure_bar, diameter_um)
    radius = diameter_um * 0.5e-6
    mode = Luna.Antiresonant.ZeisbergerMode(radius, :Ar, pressure_bar;
        wallthickness=0.65e-6, model=:full, loss=true)
    aeff = Modes.Aeff(mode)
    tau0 = tau_fs * 1e-15 / (2 * log(1 + sqrt(2)))
    peak_power = energy_uj * 1e-6 / (tau0 * sqrt(pi))
    return sqrt(2 * peak_power / (pi * aeff))
end

function run_sample(filepath, energy_uj, tau_fs, pressure_bar, diameter_um)
    length_m = 0.05
    wall_m = 0.65e-6
    lambda0 = 1030e-9
    lambda_limits = (200e-9, 2500e-9)
    trange = 500e-15
    save_n = 200
    gas = :Ar
    radius = diameter_um * 0.5e-6
    energy = energy_uj * 1e-6
    tau = tau_fs * 1e-15

    grid = Grid.RealGrid(length_m, lambda0, lambda_limits, trange)
    mode = Luna.Antiresonant.ZeisbergerMode(radius, gas, pressure_bar;
        wallthickness=wall_m, model=:full, loss=true)
    density = z -> PhysData.density(gas, pressure_bar, PhysData.roomtemp)
    response = Interface.makeresponse(grid, gas, false, true, true,
        true, false, true, true, Dict{Symbol, Any}(), 0.0, PhysData.roomtemp)
    inputs = Interface.makeinputs(mode, lambda0, nothing, tau, nothing, Float64[], nothing,
        energy, :sech, :linear, nothing)
    inputs, noise_field = Interface.makenoise(grid, mode, inputs, false, Random.GLOBAL_RNG)
    linop, eomega, transform, ft = Interface.setup(grid, mode, density, response, inputs,
        false, 1e-3, Val(true); noise_field)
    stats = Stats.default(grid, eomega, mode, linop, transform; gas=gas)
    output = Interface.makeoutput(grid, save_n, stats, filepath, nothing, nothing, nothing)
    Interface.saveargs(output; radius, flength=length_m, gas, pressure=pressure_bar,
        lambda_limits, trange, lambda0, tau, phi=Float64[], energy, pulseshape=:sech,
        model=:full, loss=true, raman=false, plasma=true, saveN=save_n, filepath)
    Luna.run(eomega, grid, linop, transform, ft, output; status_period=1000)
end

function write_metadata(filepath, label, index, seed, energy, tau, pressure, diameter)
    h5open(filepath, "r+") do file
        group = haskey(file, "benchmark_params") ? file["benchmark_params"] : create_group(file, "benchmark_params")
        group["class"] = label
        group["candidate_index"] = index
        group["seed"] = seed
        group["energy_uj"] = energy
        group["tau_fs"] = tau
        group["pressure_bar"] = pressure
        group["diameter_um"] = diameter
        group["wallthickness_um"] = 0.65
        group["length_cm"] = 5.0
        group["save_n"] = 200
        group["shotnoise"] = false
    end
end

function main()
    args = parse_args()
    if args === nothing
        print_help()
        return
    end
    output_dir = abspath(args["output-dir"])
    mkpath(output_dir)
    label = args["class"]
    count = args["count"]
    start_index = args["start-index"]
    seed = args["seed"]
    ranges = ranges_for(label)
    println("Generating $label candidates in $output_dir")
    max_attempts = args["max-attempts"]
    println("count=$count start=$start_index seed=$seed L=5 cm saveN=200 max_attempts=$max_attempts")
    for index in start_index:(start_index + count - 1)
        filename = @sprintf("candidate_%04d.h5", index)
        filepath = joinpath(output_dir, filename)
        marker = filepath * ".done"
        if isfile(filepath) && isfile(marker) && !args["overwrite"]
            println("[$index] already complete, skipping")
            continue
        end
        temporary = filepath * ".tmp"
        completed = false
        for attempt in 1:max_attempts
            rng = MersenneTwister(seed + 10_000 * index + attempt)
            energy = sample_uniform(rng, ranges.energy)
            tau = sample_uniform(rng, ranges.tau)
            pressure = sample_uniform(rng, ranges.pressure)
            diameter = sample_uniform(rng, ranges.diameter)
            initial_field = initial_field_strength(energy, tau, pressure, diameter)
            if label != "simple" && initial_field > INITIAL_FIELD_SAFETY_RATIO * PPT_FIELD_LIMIT
                println(@sprintf("[%d] attempt %d rejected by initial field %.4e V/m", index, attempt, initial_field))
                continue
            end
            rm(temporary; force=true)
            try
                println(@sprintf("[%d] attempt %d E=%.4f uJ tau=%.3f fs p=%.3f bar d=%.3f um", index, attempt, energy, tau, pressure, diameter))
                run_sample(temporary, energy, tau, pressure, diameter)
                write_metadata(temporary, label, index, seed, energy, tau, pressure, diameter)
                mv(temporary, filepath; force=true)
                open(marker, "w") do io
                    println(io, "complete=true")
                    println(io, "save_n=200")
                    println(io, "attempt=$attempt")
                    println(io, "initial_field_v_per_m=$initial_field")
                end
                completed = true
                break
            catch err
                rm(temporary; force=true)
                @warn "Candidate attempt failed; resampling" index attempt exception=(err, catch_backtrace())
            end
        end
        if !completed
            @error "Candidate exhausted all attempts" index max_attempts
        end
    end
end

main()
