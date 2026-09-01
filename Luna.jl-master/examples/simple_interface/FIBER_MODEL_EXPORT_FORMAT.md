# Anti-resonant Fibre Model Export Format

This export is produced by `anti_resonant_simulation.jl` when
`--export-fiber-model` is enabled.

## Files

- `fiber_model_YYYYmmdd_HHMMSS.mat`
  HDF5-backed MATLAB container. Read it in MATLAB with `h5read`.
- `fiber_model_YYYYmmdd_HHMMSS.csv`
  Plain-text table containing the main wavelength-dependent arrays.
- `fiber_model_YYYYmmdd_HHMMSS_metadata.json`
  Metadata and integrity information.
- `read_fiber_model_matlab.m`
  MATLAB example script for reading and plotting the export.

## Main Datasets

All wavelength-dependent arrays have the same length and are sampled on
`lambda_nm`, which is monotonic increasing.

| Dataset | Meaning | Unit |
| --- | --- | --- |
| `lambda_nm` | Vacuum wavelength | nm |
| `D_ps_nm_km` | Dispersion parameter | ps/(nm km) |
| `beta2_ps2_km` | Group-velocity dispersion | ps^2/km |
| `beta3_ps3_km` | Third-order dispersion | ps^3/km |
| `loss_dB_km` | Power loss | dB/km |
| `n_eff_real` | Real part of effective index | dimensionless |
| `n_eff_imag` | Imaginary part of effective index | dimensionless |
| `A_eff_um2` | Effective mode area | um^2 |
| `gamma_W_km` | Nonlinear coefficient | 1/(W km) |

## Fibre-specific Datasets

| Dataset | Meaning | Unit |
| --- | --- | --- |
| `resonant_wavelengths` | Tube-wall resonance wavelengths | nm |
| `anti_resonant_band` | Working anti-resonant band around the pump wavelength | nm |
| `core_radius` | Core radius | m |
| `tube_count` | Tube count, `NaN` for `ZeisbergerMode` | count |
| `wall_thickness` | Tube-wall thickness | m |

## Notes

- The `.mat` file is HDF5-backed and intended for MATLAB `h5read`.
- `tube_count` is exported as `NaN` because `Antiresonant.ZeisbergerMode`
  does not define an explicit tube count.
- The export wavelength range is independent from the propagation grid used
  by the pulse simulation.
- `D_ps_nm_km` is computed from `beta2` with
  `D = -(2*pi*c/lambda^2) * beta2`.
