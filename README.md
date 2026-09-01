# AI-AR-HCF-UPPE-Luna-Modeling

This repository collects simulation, preprocessing, model training, evaluation, and manuscript assets for AI-assisted ultrafast pulse propagation prediction in gas-filled antiresonant hollow-core fibers.

The project combines Luna/UPPE simulations with machine-learning surrogate models. The current research focus is forward prediction from pulse, gas, and fiber parameters to final spectra and full spectral-evolution maps, with follow-up work on early-dense data, absolute-intensity modeling, RNN baselines, and inverse design.

## Repository layout

- `Luna.jl-master/`: nested Julia package root for Luna plus local simple-interface experiments.
- `Luna.jl-master/examples/simple_interface/`: AR-HCF data generation, preprocessing, Transformer/CNN/MLP training, evaluation, and visualization scripts.
- `rnnnonlinear-master/`: migrated and adapted RNN baseline code.
- `figures/`: selected paper or presentation figures.
- `docs/`: project state, experiment notes, reusable commands, and development workflow.

Large generated outputs such as `.h5`, `.mat`, `.npy`, model checkpoints, PPT/PDF/DOCX files, and generated figures are ignored by Git by default. Use external storage or Git LFS if any of those artifacts need versioning.

## Main entry points

- `Luna.jl-master/examples/simple_interface/data_generation.jl`: generate Luna/UPPE AR-HCF samples.
- `Luna.jl-master/examples/simple_interface/data_preprocessing.py`: convert HDF5 simulations into model-ready arrays.
- `Luna.jl-master/examples/simple_interface/train_mlp.py`: train and evaluate MLP, temporal CNN, and temporal Transformer models.
- `Luna.jl-master/examples/simple_interface/visualize_hdf5.jl`: visualize HDF5 simulation samples and z-range zooms.
- `rnnnonlinear-master/rnnnonlinear-master/prepare_luna_data.py`: prepare Luna temporal data for the RNN baseline.
- `rnnnonlinear-master/rnnnonlinear-master/train_luna_rnn.py`: train Luna RNN baseline variants.
- `rnnnonlinear-master/rnnnonlinear-master/evaluate_luna_rnn.py`: evaluate and visualize RNN stepwise and autoregressive predictions.

## Current modeling line

- Transformer/CNN forward surrogate: direct prediction of full `S(z, lambda)` and final spectrum.
- Luna RNN baseline: compares teacher-forced stepwise prediction with autoregressive rollout.
- Early-dense data: intended to improve early propagation details.
- Global-log/absolute-intensity data: separate task from per-sample normalized spectrum-shape prediction.
- Inverse design: planned as a layer on top of trained forward surrogates, with Luna/UPPE verification.

## Working with Codex

For a new session, read these files first:

1. `AGENTS.md`
2. `docs/PROJECT_STATE.md`
3. `docs/EXPERIMENTS.md`
4. `docs/COMMANDS.md`

Then inspect the relevant code before editing. Prefer small, verifiable changes with explicit test commands and a short summary of changed files, verification results, and remaining risks.
