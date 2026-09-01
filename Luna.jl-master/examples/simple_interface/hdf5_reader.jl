using Luna
using Luna.Output
using Luna.Processing
using Luna.Plotting
using HDF5

"""
HDF5 file reader for Luna simulation output

This script provides functions to read and analyze HDF5 output files from Luna simulations,
particularly for anti-resonant fiber simulations. It includes data validation and error handling.
"""

"""
Open an HDF5 output file and return an HDF5Output object

# Arguments
- `file_path::String`: Path to the HDF5 file

# Returns
- `HDF5Output`: An HDF5Output object for the file

# Throws
- `ErrorException`: If the file doesn't exist or is not a valid HDF5 file
"""
function open_hdf5_output(file_path::String)
    if !isfile(file_path)
        error("File not found: $file_path")
    end
    
    try
        return HDF5Output(file_path)
    catch e
        error("Failed to open HDF5 file: $e")
    end
end

"""
Extract simulation parameters from the HDF5 output

# Arguments
- `output::HDF5Output`: The HDF5Output object

# Returns
- `Dict{String, Any}`: A dictionary containing simulation parameters

# Throws
- `ErrorException`: If required parameters are missing
"""
function extract_parameters(output::HDF5Output)
    params = Dict{String, Any}()
    
    try
        # Basic simulation info
        if haskey(output, "meta")
            params["meta"] = output["meta"]
        end
        
        # Simulation type
        if haskey(output, "simulation_type")
            params["simulation_type"] = output["simulation_type"]
        end
        
        # Propagation parameters
        if haskey(output, "prop_capillary_args")
            params["prop_capillary_args"] = output["prop_capillary_args"]
        end
        
        # Grid information
        if haskey(output, "grid")
            params["grid"] = output["grid"]
        end
        
        # Modes (if present)
        if haskey(output, "modes")
            params["modes"] = output["modes"]
        end
        
        # z positions
        if haskey(output, "z")
            params["z"] = output["z"]
        else
            error("Missing z positions in output")
        end
        
        # Check for essential data
        if !haskey(output, "Eω")
            error("Missing Eω data in output")
        end
        
        return params
    catch e
        error("Failed to extract parameters: $e")
    end
end

"""
Extract statistics from the HDF5 output

# Arguments
- `output::HDF5Output`: The HDF5Output object

# Returns
- `Dict{String, Any}`: A dictionary containing statistics

# Throws
- `ErrorException`: If statistics are missing
"""
function extract_statistics(output::HDF5Output)
    try
        if haskey(output, "stats")
            return output["stats"]
        else
            error("No statistics found in output")
        end
    catch e
        error("Failed to extract statistics: $e")
    end
end

"""
Extract spectral data at specified z positions

# Arguments
- `output::HDF5Output`: The HDF5Output object
- `z_slice`: z position(s) to extract data from (can be a single value or array)
- `specaxis::Symbol`: Spectral axis to use (:λ, :f, or :ω)
- `λrange::Tuple{Float64, Float64}`: Wavelength range to limit to

# Returns
- `Tuple{AbstractVector, AbstractArray, AbstractVector}`: (spectral axis, spectral data, actual z positions)
"""
function extract_spectral_data(output::HDF5Output, z_slice=maximum(output["z"]);
                              specaxis::Symbol=:λ, λrange=(150e-9, 2000e-9))
    try
        return getIω(output, specaxis, z_slice; λrange=λrange)
    catch e
        error("Failed to extract spectral data: $e")
    end
end

"""
Extract time-domain data at specified z positions

# Arguments
- `output::HDF5Output`: The HDF5Output object
- `z_slice`: z position(s) to extract data from (can be a single value or array)
- `trange::Tuple{Float64, Float64}`: Time range to limit to
- `oversampling::Int`: Oversampling factor for time domain

# Returns
- `Tuple{AbstractVector, AbstractArray, AbstractVector}`: (time axis, time-domain data, actual z positions)
"""
function extract_time_data(output::HDF5Output, z_slice=maximum(output["z"]);
                          trange=(-50e-15, 50e-15), oversampling=4)
    try
        return getEt(output, z_slice; trange=trange, oversampling=oversampling)
    catch e
        error("Failed to extract time-domain data: $e")
    end
end

"""
Print summary of simulation parameters

# Arguments
- `params::Dict{String, Any}`: Simulation parameters
"""
function print_parameters_summary(params::Dict{String, Any})
    println("=== Simulation Parameters ===")
    
    if haskey(params, "prop_capillary_args")
        cap_args = params["prop_capillary_args"]
        println("\nFiber Parameters:")
        println("  Length: $(get(cap_args, "flength", "N/A")) m")
        println("  Radius: $(get(cap_args, "radius", "N/A")) m")
        println("  Gas: $(get(cap_args, "gas", "N/A"))")
        println("  Pressure: $(get(cap_args, "pressure", "N/A")) bar")
        
        println("\nPulse Parameters:")
        println("  Wavelength: $(get(cap_args, "λ0", "N/A")) m")
        println("  Pulse width: $(get(cap_args, "τfwhm", "N/A")) s")
        println("  Energy: $(get(cap_args, "energy", "N/A")) J")
    end
    
    if haskey(params, "z")
        z = params["z"]
        println("\nPropagation:")
        println("  Start: $(z[1]) m")
        println("  End: $(z[end]) m")
        println("  Points: $(length(z))")
    end
end

"""
Print summary of statistics

# Arguments
- `stats::Dict{String, Any}`: Statistics data
"""
function print_statistics_summary(stats::Dict{String, Any})
    println("\n=== Statistics ===")
    
    if haskey(stats, "energy")
        energy = stats["energy"]
        println("Energy:")
        println("  Initial: $(energy[1] * 1e6) μJ")
        println("  Final: $(energy[end] * 1e6) μJ")
        println("  Loss: $((1 - energy[end]/energy[1]) * 100)%")
    end
    
    if haskey(stats, "peakpower")
        peakpower = stats["peakpower"]
        println("\nPeak Power:")
        println("  Maximum: $(maximum(peakpower) * 1e-3) kW")
        println("  Average: $(mean(peakpower) * 1e-3) kW")
    end
    
    if haskey(stats, "fwhm_t_min")
        fwhm_t = stats["fwhm_t_min"]
        println("\nTemporal FWHM:")
        println("  Minimum: $(minimum(fwhm_t) * 1e15) fs")
        println("  Maximum: $(maximum(fwhm_t) * 1e15) fs")
    end
end

"""
Main function to analyze an HDF5 output file

# Arguments
- `file_path::String`: Path to the HDF5 file
- `visualize::Bool`: Whether to generate visualizations
"""
function analyze_hdf5_output(file_path::String, visualize::Bool=false)
    println("Analyzing HDF5 output file: $file_path")
    
    # Open the file
    output = open_hdf5_output(file_path)
    
    # Extract parameters
    params = extract_parameters(output)
    print_parameters_summary(params)
    
    # Extract statistics
    stats = extract_statistics(output)
    print_statistics_summary(stats)
    
    # Extract and print spectral data info
    specx, Iω, zactual = extract_spectral_data(output)
    println("\nSpectral Data:")
    println("  Spectral axis: $(typeof(specx))")
    println("  Spectral points: $(length(specx))")
    println("  Data shape: $(size(Iω))")
    println("  Extracted at z: $(zactual[1] * 100) cm")
    
    # Extract and print time-domain data info
    t, Et, zactual = extract_time_data(output)
    println("\nTime-domain Data:")
    println("  Time axis points: $(length(t))")
    println("  Data shape: $(size(Et))")
    println("  Extracted at z: $(zactual[1] * 100) cm")
    
    # Generate visualizations if requested
    if visualize
        println("\nGenerating visualizations...")
        try
            # Plot statistics
            stats_figs = stats(output)
            
            # Plot 2D propagation
            prop_fig = prop_2D(output, :λ)
            
            # Plot time-domain at end
            time_fig = time_1D(output)
            
            # Plot spectrum at end
            spec_fig = spec_1D(output)
            
            println("Visualizations generated successfully!")
        catch e
            println("Warning: Failed to generate visualizations: $e")
            println("This may be due to missing PyPlot dependencies.")
        end
    end
    
    println("\nAnalysis complete!")
    return params, stats
end

# Command-line interface
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia hdf5_reader.jl <hdf5_file_path> [--visualize]")
        exit(1)
    end
    
    file_path = ARGS[1]
    visualize = length(ARGS) > 1 && ARGS[2] == "--visualize"
    
    try
        analyze_hdf5_output(file_path, visualize)
    catch e
        println("Error: $e")
        exit(1)
    end
end
