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

### Per-sample-minmax no-detach horizon sweep

- Purpose: hold the target space and conditioning fixed while increasing the number of autoregressive steps in the first 10 cm of propagation.
- Shared configuration: direct sigmoid prediction, `features_z`, scheduled-sampling feedback `0.0 -> 0.03`, mixed rollout starts with `zero_start_prob=0.9`, differentiable feedback (`--no-detach-feedback`), learning rate `3e-5`, and gradient clipping `1.0`.
- Completed 51-point task:
  - `lambda251`: full-test `autoregressive_r2=0.4685`, final `R2=0.2819`.
  - `lambda1000`: full-test `autoregressive_r2=0.4873`, final `R2=0.3301`; selected autoregressive checkpoint from epoch 1.
- Queued tasks: `z10cm_101_lambda1000` and `z10cm_201_lambda1000`. Their purpose is diagnostic: determine whether the 51-point gain survives 91 and 191 recurrent prediction steps.

## Original SC versus Luna data-dynamics comparison

- Goal: visually and quantitatively compare the recurrent target spaces without retraining either model.
- Inputs: original `SC_spec_251` dBm data and Luna raw-power `z10cm_51_lambda251` data mapped through the same original global `-55 dB` pipeline.
- Outputs: a 2x3 PPT overview, four quantile-selected ground-truth target pairs, target sparsity, z-direction dynamics, spectral occupancy, local 10-step window ambiguity, JSON metrics, and a Markdown report.
- Interpretation boundary: local-window ambiguity is a descriptive predictability statistic. It supports a more or less stable recurrent mapping under a given representation, but does not independently prove a physical causal mechanism.

## RNN attribution matrix

- Goal: answer whether Luna autoregressive failure is primarily a code-port issue or a data/task-difficulty issue.
- Original-code/original-data check: run the unmodified `salmelal/rnnnonlinear` Keras workflow on Zenodo data, preferably `SC_spec_251` or `norm_NLSE_spec_128`.
- PyTorch-code/original-data check: convert the original `.mat` data to HDF5 with `/data=(N,n_grid,n_steps)` using `convert_rnnnonlinear_mat.py`, then train/evaluate with `train_luna_rnn.py` using `conditioning=none`, `window_size=10`, direct sigmoid prediction, and the same train/test evolution counts.
- Low-dimensional Luna check: export `z10cm_51_lambda251`, `z10cm_101_lambda251`, and optionally `z50cm_200_lambda251` from the existing processed early-dense data using `prepare_luna_data.py --lambda-target-points 251`.
- Interpretation:
  - If PyTorch code succeeds on original data, code migration is mostly sound.
  - If low-dimensional Luna succeeds but full Luna fails, the bottleneck is rollout length/output dimensionality.
  - If low-dimensional Luna still fails, the AR-HCF UPPE trajectory distribution is harder than the original NLSE/GNLSE tasks.
- Cloud execution started:
  - `/mnt/Luna.jl-master/rnn_earlydense/run_luna_lambda251_attribution.sh` exports and trains Luna `lambda251` tasks.
  - `/mnt/Luna.jl-master/rnn_original_data/run_original_data_attribution.sh` downloads Zenodo `RNNnonlinear_v2.zip`, then converts original `.mat` data for PyTorch training.
  - `/mnt/Luna.jl-master/rnn_original_code_env/setup_tf1_env.sh` builds an isolated Python 3.7 / TensorFlow 1.x / Keras 2.x environment for unmodified original-code experiments.
  - `/mnt/Luna.jl-master/rnn_original_code_env/run_original_code_luna_lambda251.sh` runs original `load_data.py`, `make_RNN_model.py`, and `pred_evo.py` on Luna `z10cm_51_lambda251` mat5 data after the environment is ready.
  - `/mnt/Luna.jl-master/rnn_original_code_env/run_original_code_original_data.sh` runs original code on extracted Zenodo data after both the environment and data archive are ready.

### Original Keras code on Luna low-dimensional data

- Completed date: 2026-09-07 status check.
- Data: Luna early-dense `z10cm_51_lambda251`, `data=(7780, 251, 51)`, train/test evolutions `7002/778`, `window_size=10`.
- Code path: original Keras-style `load_data.py`, `make_RNN_model.py`, and `pred_evo.py` inside `/mnt/Luna.jl-master/rnn_original_code_env/epoch_sweep_luna_lambda251`.
- Result summary:
  - `per_sample_minmax`, 1 epoch: `stepwise_r2=0.9600`, `autoregressive_r2=0.3082`.
  - `per_sample_minmax`, 3 epochs: `stepwise_r2=0.9743`, `autoregressive_r2=0.3161`.
  - `per_sample_minmax`, 5 epochs: `stepwise_r2=0.9764`, `autoregressive_r2=0.2809`.
  - `rnn_paper_db`, 1 epoch: `stepwise_r2=0.9656`, `autoregressive_r2=-0.1442`.
  - `rnn_paper_db`, 3 epochs: `stepwise_r2=0.9780`, `autoregressive_r2=0.0947`.
  - `rnn_paper_db`, 5 epochs: `stepwise_r2=0.9837`, `autoregressive_r2=-0.0537`.
- Interpretation: original Keras code also shows the same pattern on Luna data: teacher-forced one-step prediction is high, but autoregressive rollout remains weak and often worsens with more one-step training. This points more strongly to task/data dynamics than to only a PyTorch implementation bug.

### Raw-power plus original dBm preprocessing check

- Started: 2026-09-07.
- Purpose: test whether previous Luna-to-RNN preprocessing still differs from the original Salmela pipeline.
- Method: export Luna HDF5 fields as raw positive linear power using `export_luna_raw_power_mat.py`, then run original `load_data(..., normalization='dBm')` so the original code performs global-max normalization, dB compression, clipping, and `[0,1]` scaling.
- Cloud script: `/mnt/Luna.jl-master/rnn_original_code_env/run_raw_power_dBm_luna_lambda251.sh`.
- Output data: `/mnt/Luna.jl-master/rnn_original_code_env/luna_t0p6_earlydense_z10cm_51_lambda251_rawpower_originalcode.mat`.
- Planned epochs: 1, 3, and 5.
- Interpretation rule: if this run remains around the previous `0.3` autoregressive R2 level, preprocessing mismatch is not the main cause; if it improves sharply, the Luna RNN route should use raw-power export plus original dBm scaling.

### Luna-specific raw-power preprocessing sweep

- Started and completed: 2026-09-07.
- Goal: find a preprocessing that keeps Luna raw-power spectra RNN-friendly without copying the original global `-55 dB` scaling too literally.
- Code change: `export_luna_raw_power_mat.py` now supports `--export-normalization` values `raw_linear_power`, `global_db`, `per_sample_db`, `bandwise_db`, and `log1p`. Non-raw modes export `[0,1]` targets directly, so downstream RNN training should use no additional normalization.
- Fixed task: t0p6 early-dense, front `10 cm`, `51` z points, `251` wavelength points over `200--2500 nm`.
- Smoke export on 300 samples:
  - `per_sample_db_p99p9_floor80`: `frac<=0=0.3716`, `frac<=0.1=0.4120`.
  - `per_sample_db_max_floor80`: `frac<=0=0.3779`, `frac<=0.1=0.4189`.
  - `per_sample_db_p99p5_floor80`: `frac<=0=0.3601`, `frac<=0.1=0.3984`.
  - `bandwise_db_p99p9_floor80`: `frac<=0=0.2244`, `frac<=0.1=0.2545`.
  - `log1p_p99p9`: `frac<=0=0.0120`, but `frac<=0.1=0.9385`, so most target values are compressed near zero.
- Cloud runner: `/mnt/Luna.jl-master/rnn_preprocess_experiments/run_relative_db_rnn_experiments.sh`.
- Runner PID when launched: `44195`.
- Priority order: `bandwise_db_p99p9_floor80`, then `per_sample_db_p99p5_floor80`, `per_sample_db_p99p9_floor80`, `per_sample_db_max_floor80`, and finally `log1p_p99p9`.
- Training recipe: PyTorch RNN, `conditioning=none`, direct prediction, sigmoid output, low scheduled sampling `0.0 -> 0.05`, mixed rollout start with `zero-start-prob=0.8`, gradient clipping `1.0`, and autoregressive early stopping.
- Full-test results:
  - `bandwise_db_p99p9_floor80`: `stepwise_r2=0.9587`, `autoregressive_r2=0.3497`, `autoregressive_final_r2=0.1166`.
  - `per_sample_db_p99p5_floor80`: `stepwise_r2=0.9051`, `autoregressive_r2=0.2527`, `autoregressive_final_r2=-0.0144`.
  - `per_sample_db_p99p9_floor80`: `stepwise_r2=0.9601`, `autoregressive_r2=0.3559`, `autoregressive_final_r2=0.1899`.
  - `per_sample_db_max_floor80`: `stepwise_r2=0.9466`, `autoregressive_r2=0.3748`, `autoregressive_final_r2=0.2192`.
  - `log1p_p99p9`: `stepwise_r2=0.5462`, `autoregressive_r2=0.4307`, `autoregressive_final_r2=0.0466`.
- Interpretation: less sparse relative-dB targets did not clearly improve over the best `per_sample_minmax` short baseline. `log1p` reaches similar autoregressive R2 but with very low stepwise R2, suggesting that it compresses the task toward coarse structure rather than producing a genuinely better recurrent model.

### Per-sample-minmax features-z no-detach short-horizon sweep

- Started: 2026-09-08.
- Goal: execute the current best judgment after the relative-dB sweep: keep the more stable `per_sample_minmax` target space, add physical and z conditioning, and allow gradients through recursive feedback.
- Cloud runner: `/mnt/Luna.jl-master/rnn_earlydense/run_perminmax_features_z_nodetach.sh`.
- Active PID when launched: `46166`.
- Experiments run serially:
  - `results_features_z_nodetach_z10cm_51_lambda251_perminmax_v1`
  - `results_features_z_nodetach_z10cm_51_lambda1000_perminmax_v1`
- Training recipe: `scheduled_sampling`, `conditioning=features_z`, direct prediction, sigmoid output, `--no-detach-feedback`, rollout curriculum `10 -> 41`, feedback `0.0 -> 0.03`, mixed rollout start with `zero-start-prob=0.9`, `learning-rate=3e-5`, and `grad-clip=1.0`.
- Target comparison: beat or match the current short RNN baseline around `autoregressive_r2=0.43` while improving final autoregressive R2.

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
