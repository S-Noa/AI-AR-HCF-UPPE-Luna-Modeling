# Experiment notes

Use this file as the lightweight memory for model and data experiments. Keep each entry short enough that a new Codex session can scan it quickly.

## Transformer temporal v3

- Goal: predict normalized log-power final spectra and full `S(z, lambda)` maps from physical input parameters.
- Data: T650 processed data, 200--2500 nm target grid, per-sample normalized log-power representation.
- Reported metrics: `Final R2 = 0.9635`, `Temporal R2 = 0.9650`, `Temporal UV R2 = 0.8927`.
- Observed limitation: good global spectral evolution, but short-wavelength narrow features and fine propagation textures are smoothed.
- Follow-up: use temporal-evolution validation metrics, early-dense data, and moderate wavelength-gradient/UV weighting.

## Temporal CNN baseline

- Goal: direct propagation-map prediction.
- Status: treated as a strong engineering baseline and potentially stronger than Transformer v3 in some visual comparisons.
- Follow-up: compare fairly against Transformer using the same data split, target normalization, and temporal metrics.

## Luna RNN baseline

- Goal: migrate/open-source RNN-style one-step propagation learning to Luna data.
- Stepwise mode: teacher-forced one-step prediction using true history at each step.
- Autoregressive mode: rollout from initial history by feeding predictions back into the model.
- Earlier result: stepwise high, autoregressive failed, indicating error accumulation and missing conditioning.
- Follow-up modes: `open_source_legacy`, `conditional_legacy`, and `rollout_robust`.

## t0p6 early-dense RNN baseline

- Goal: test whether denser early-z sampling improves Luna RNN one-step and autoregressive propagation baselines.
- Source data: `7792` early-dense HDF5 samples in `/mnt/Luna.jl-master/training_data_ar_t0p6_nochirp`.
- Z-grid audit: `7780` files are the main `z_len=1401` / first-10-cm dense grid; `12` files are older `z_len=696` / first-0.5-cm dense grid. Do not mix both in one temporal dataset.
- Filter input: symlink directory `/mnt/Luna.jl-master/training_data_ar_t0p6_earlydense_z1401_links` with only the `7780` consistent samples.
- Preprocessing target: `/mnt/Luna.jl-master/processed_t0p6_earlydense_z1401_v1`, using `sample-filter=earlydense`, `target-points=1000`, and `per_sample_minmax`.
- RNN export target: `/mnt/Luna.jl-master/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat`.
- Completed preprocessing: `Train=5446`, `Val=1167`, `Test=1167`, `y_temporal=(N, 1401, 1000)`.
- Completed smoke tests: legacy smoke reached high stepwise R2 but poor autoregressive behavior; scheduled-sampling smoke completed without OOM.
- Completed comparison: `open_source_legacy` with no conditioning reached high stepwise R2 but remained negative in autoregressive rollout and was stopped at epoch 20.
- Stopped run: `conditional_legacy` one-step was stopped after epoch 14 because it did not improve autoregressive stability enough to justify a third concurrent full job.
- Stopped run: first from-scratch `features_z + scheduled_sampling` was stopped after epoch 14 because autoregressive R2 peaked early (`0.2954` at epoch 2) and then declined while stepwise R2 kept improving.
- Warm-start diagnostic: initializing scheduled sampling from `results_features_z_z1401_v1/best_stepwise_model.pth` preserved high stepwise R2 but kept full z1401 autoregressive R2 negative in the smoke test.
- Diagnostic path: front-10-cm / 201-z-point dataset exported successfully. Scheduled-sampling training selected the best autoregressive checkpoint at epoch 1 and kept final autoregressive R2 positive, unlike full z1401 rollout.
- New planned data route: `rnn_paper_db`, matching the Salmela RNN normalization style with global max, dB clipping, and `[0,1]` targets.
- New planned training route: residual prediction plus curriculum rollout and mixed zero/random rollout starts.
- Completed `rnn_paper_db` full preprocessing: `/mnt/Luna.jl-master/processed_t0p6_earlydense_z1401_rnnpaperdb_v1`.
- `rnn_paper_db` diagnostics: the original `-55 dB` global-max scaling clips too much of the Luna target to zero, with roughly 66% zero values in temporal maps and roughly 88% zero values in final spectra. This makes the autoregressive state space very sparse.
- A 300-sample `rnn_paper_db` diagnostic with `-80 dB` floor reduced zero clipping to about 49%, but a residual scheduled-sampling smoke test still produced negative autoregressive R2.
- Added `--rnn-db-floor`, `--rnn-db-reference-mode`, and `--rnn-db-reference-percentile` to test less aggressive dB normalization without changing default behavior.
- Added `--no-detach-feedback` to `train_luna_rnn.py` so recursive feedback can be trained with differentiable truncated BPTT. A small `z10cm_201` smoke improved slowly but remained negative after 5 epochs, so this is not yet the default route.
- Completed candidate baseline: `/mnt/Luna.jl-master/rnn_earlydense/results_direct_z10cm_101_perminmax_stable_v1`, using per-sample min-max targets, direct sigmoid prediction, 101 front-10-cm z points, low scheduled-sampling feedback, and autoregressive checkpoint selection. It selected epoch 5 and reached full-test `autoregressive_r2=0.4174`, with positive but modest `autoregressive_final_r2=0.1584`.
- Completed harder `z10cm_201` comparison: `/mnt/Luna.jl-master/rnn_earlydense/results_direct_z10cm_201_perminmax_stable_v1` selected epoch 15 and reached full-test `autoregressive_r2=0.2867`, `autoregressive_final_r2=0.0748`, `stepwise_r2=0.9671`.
- Completed warm-start comparison: `/mnt/Luna.jl-master/rnn_earlydense/results_warm_scheduled_z10cm_101_perminmax_v1` selected epoch 1 and reached full-test `autoregressive_r2=0.2830`, `autoregressive_final_r2=0.1686`, `stepwise_r2=0.9831`; warm-start did not improve rollout stability.
- Completed coarse `z10cm_51` comparison: `/mnt/Luna.jl-master/rnn_earlydense/results_direct_z10cm_51_perminmax_stable_v1` selected epoch 2 and reached full-test `autoregressive_r2=0.4296`, `autoregressive_final_r2=0.1774`, `stepwise_r2=0.9126`. This is the current best short RNN autoregressive baseline.
- New attribution plan: compare original RNNnonlinear code/data against our PyTorch training code and low-dimensional Luna tasks. This should separate code mismatch from Luna data difficulty.
- Key metrics: `stepwise_r2`, `autoregressive_r2`, final autoregressive spectrum quality, and temporal evolution plots.

## RNN attribution matrix

- Goal: answer whether Luna autoregressive failure is primarily a code-port issue or a data/task-difficulty issue.
- Original-code/original-data check: run the unmodified `salmelal/rnnnonlinear` Keras workflow on Zenodo data, preferably `SC_spec_251` or `norm_NLSE_spec_128`.
- PyTorch-code/original-data check: convert the original `.mat` data to HDF5 with `/data=(N,n_grid,n_steps)` using `convert_rnnnonlinear_mat.py`, then train/evaluate with `train_luna_rnn.py` using `conditioning=none`, `window_size=10`, direct sigmoid prediction, and the same train/test evolution counts.
- Low-dimensional Luna check: export `z10cm_51_lambda251`, `z10cm_101_lambda251`, and optionally `z50cm_200_lambda251` from the existing processed early-dense data using `prepare_luna_data.py --lambda-target-points 251`.
- Interpretation:
  - If PyTorch code succeeds on original data, code migration is mostly sound.
  - If low-dimensional Luna succeeds but full Luna fails, the bottleneck is rollout length/output dimensionality.
  - If low-dimensional Luna still fails, the AR-HCF UPPE trajectory distribution is harder than the original NLSE/GNLSE tasks.

## Early-dense data

- Goal: improve early propagation details in the first centimeters.
- Status: t0p6 cloud source data is available. The main usable subset for current RNN baselines is the `z_len=1401` set; the mixed raw early-dense directory should not be used directly for temporal preprocessing.
- Follow-up: use the processed early-dense data for Luna RNN baseline first, then retrain or fine-tune Transformer/CNN.

## Global-log / absolute-intensity modeling

- Goal: predict physically meaningful log-power scale rather than only per-sample normalized spectral shape.
- Important constraint: do not use sigmoid output activation for unbounded global-log targets; use identity output activation.
- Follow-up: start with a small model smoke test before full temporal-map training.

## Inverse design / reinforcement learning

- Goal: use a trained forward surrogate as a fast environment for parameter search.
- Recommended first step: differentiable or black-box optimization through the forward surrogate, followed by Luna verification.
- RL role: future inverse-design policy layer, not a replacement for the forward surrogate.
