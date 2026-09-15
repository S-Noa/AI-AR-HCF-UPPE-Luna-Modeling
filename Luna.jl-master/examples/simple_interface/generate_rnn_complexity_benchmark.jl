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
        elseif key == "--count" || key == "--start-index" || key == "--seed"
            values[key[3:end]] = parse(Int, value)
        elseif key == "--overwrite"
            values["overwrite"] = lowercase(value) in ("1", "true", "yes")
        else
            error("Unknown argument: $key")
        end
        i += 1
    end
    values["class"] in ("simple", "complex") || error("--class must be simple or complex")
    return values
end

function print_help()
    println("""
Generate controlled AR-HCF RNN benchmark candidates.

Usage:
  julia --project=../.. generate_rnn_complexity_benchmark.jl \\
    --class simple|complex --output-dir DIR [--count 1600]

Every trajectory uses Ar, t=0.65 um, L=5 cm, 200 saved z planes,
plasma=true, raman=false, loss=true, and shotnoise=false.
""")
end

function ranges_for(label)
    if label == "simple"
        return (energy=(0.3, 1.0), tau=(25.0, 50.0), pressure=(0.5, 15.0), diameter=(30.0, 50.0))
    end
    return (energy=(2.0, 3.0), tau=(5.0, 12.0), pressure=(20.0, 50.0), diameter=(10.0, 20.0))
end

sample_uniform(rng, interval) = interval[1] + rand(rng) * (interval[2] - interval[1])

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
    ranges = ranges_for(args["class"])
    println("Generating $(args[\"class\"]) candidates in $output_dir")
    println("count=$(args[\"count\"]) start=$(args[\"start-index\"]) seed=$(args[\"seed\"]) L=5 cm saveN=200")
    for index in args["start-index"]:(args["start-index"] + args["count"] - 1)
        filename = @sprintf("candidate_%04d.h5", index)
        filepath = joinpath(output_dir, filename)
        marker = filepath * ".done"
        if isfile(filepath) && isfile(marker) && !args["overwrite"]
            println("[$index] already complete, skipping")
            continue
        end
        rng = MersenneTwister(args["seed"] + index)
        energy = sample_uniform(rng, ranges.energy)
        tau = sample_uniform(rng, ranges.tau)
        pressure = sample_uniform(rng, ranges.pressure)
        diameter = sample_uniform(rng, ranges.diameter)
        temporary = filepath * ".tmp"
        rm(temporary; force=true)
        try
            println(@sprintf("[%d] E=%.4f uJ tau=%.3f fs p=%.3f bar d=%.3f um", index, energy, tau, pressure, diameter))
            run_sample(temporary, energy, tau, pressure, diameter)
            write_metadata(temporary, args["class"], index, args["seed"], energy, tau, pressure, diameter)
            mv(temporary, filepath; force=true)
            open(marker, "w") do io
                println(io, "complete=true")
                println(io, "save_n=200")
            end
        catch err
            rm(temporary; force=true)
            @error "Candidate failed" index exception=(err, catch_backtrace())
        end
    end
end

main()
