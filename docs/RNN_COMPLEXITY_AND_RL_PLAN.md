# RNN Complexity Benchmark and RL Inverse Design

The controlled RNN benchmark is separate from early-dense production data. It compares simple and complex AR-HCF trajectories with Ar, t0p6, a 0--5 cm propagation length, 200 saved z planes, 251 wavelength bins, and a 10-spectrum recurrent window. Each class retains 1,250 train plus 50 test trajectories after complexity ranking.

The complex class uses a numerically feasible strongly nonlinear subdomain: 2.0--2.5 uJ, 10--16 fs, 10--25 bar, and 18--25 um. The initially proposed extreme domain self-compressed above Luna's finite PPT ionisation-rate table for nearly every smoke candidate. The generator therefore calculates the same initial field used by the primary data generator, applies a 0.5-PPT-limit guard, and retries failed candidates. Complexity remains defined by the saved spectral evolution and ranking, rather than by admitting numerically invalid runs.

Two target representations are exported: `raw_power.mat` for the original Keras `load_data(..., 'dBm')` path and `per_sample_minmax.mat` for no-additional-normalization runs. Original Keras and PyTorch Keras-compatible models use the same RMSprop/MSE 50+30 epoch schedule.

`train_luna_rnn.py` now supports fixed 1/2/3/4-step recursive evaluation from all valid true local histories, z=0 only, or both. This separates immediate local feedback sensitivity from long full-rollout drift.

The first RL task optimizes only energy, tau, pressure, and diameter within the t0p6 support. It requires a raw4 global-log forward model and uses a linear-power UV-fraction reward. Derived features are never directly optimized. SAC, random search, differential evolution, and constrained raw4 gradient optimization share parameter bounds and require Luna validation.
