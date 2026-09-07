# Project state

Last updated: 2026-09-07

## Current objective

Prepare the project for reliable continued development while preserving the current research workflow around Luna/UPPE simulation, Transformer/CNN forward surrogates, Luna RNN baselines, and future inverse design.

## Current research lines

- Transformer forward surrogate for full spectral-evolution prediction in gas-filled AR-HCFs.
- Temporal CNN as a strong engineering baseline for propagation-map prediction.
- Luna RNN adaptation to understand stepwise prediction versus autoregressive rollout.
- Early-dense data generation to improve early propagation detail; cloud t0p6 early-dense samples are available, but only the consistent `z_len=1401` subset should be used for temporal preprocessing.
- Global-log or absolute-intensity modeling as a separate task from normalized spectrum-shape prediction.
- Inverse design and reinforcement learning as future layers built on top of verified forward surrogates.

## Known metrics and results

- Transformer temporal v3 reported approximately `Final R2 = 0.9635`.
- Transformer temporal v3 reported approximately `Temporal R2 = 0.9650`.
- Transformer temporal v3 reported approximately `Temporal UV R2 = 0.8927`.
- Luna RNN stepwise evaluation was high, around `R2 = 0.9944`, but autoregressive rollout failed in the earlier run, around `R2 = -0.4099`.
- Cloud t0p6 early-dense source data: `7792` `*_earlydense.h5` samples under `/mnt/Luna.jl-master/training_data_ar_t0p6_nochirp`.
- Early-dense z-grid audit: `7780` files have `z_len=1401`, `early_dense_saveN=1001`, and `early_dense_zmax_cm=10.0`; `12` files have the older `z_len=696`, `early_dense_saveN=201`, and `early_dense_zmax_cm=0.5`.
- RNN early-dense preprocessing should use the symlink filter directory `/mnt/Luna.jl-master/training_data_ar_t0p6_earlydense_z1401_links`, not the mixed source directory.
- t0p6 early-dense z1401 preprocessing completed: `/mnt/Luna.jl-master/processed_t0p6_earlydense_z1401_v1`, with `Train=5446`, `Val=1167`, `Test=1167`, and `y_temporal=(N, 1401, 1000)`.
- RNN z1401 export completed: `/mnt/Luna.jl-master/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat`.
- `open_source_legacy` z1401 training was stopped at epoch 20 after confirming high stepwise R2 but negative autoregressive R2.
- The `conditional_legacy` one-step run was stopped after showing high stepwise R2 but unstable autoregressive R2.
- The first `features_z + scheduled_sampling` z1401 run was stopped after epoch 14 because full autoregressive R2 peaked early and then declined while stepwise R2 kept improving.
- RNN scripts now support warm-start initialization and z-axis crop/downsample export for controlled long-horizon diagnostics.
- Warm-start scheduled-sampling smoke test on full z1401 completed but did not improve full autoregressive rollout; it preserved high stepwise R2 while autoregressive R2 remained negative.
- Early-10-cm downsampled RNN data was exported as `/mnt/Luna.jl-master/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_201_conditional.mat` with `data=(7780, 1000, 201)` and `z_norm=(201,)`.
- Early-10-cm scheduled-sampling training completed; it selected the best autoregressive checkpoint from epoch 1 and reached positive final autoregressive R2, suggesting that the full z1401 failure is strongly affected by long rollout length.
- RNN preprocessing now has a planned Salmela-style `rnn_paper_db` route: global training-set max, dB compression, `-55 dB` floor, and `[0,1]` targets.
- RNN training now has a planned autoregressive-aware route using residual prediction, curriculum rollout horizon, mixed zero/random rollout starts, and early stopping by autoregressive R2.
- The full `rnn_paper_db` early-dense preprocessing completed, but diagnostics showed that `-55 dB` global-max scaling is too sparse for Luna AR-HCF spectra: about two thirds of temporal target values and nearly 88% of final-spectrum values are clipped to zero.
- A 300-sample `rnn_paper_db` diagnostic with `-80 dB` floor reduced zero clipping, but the residual smoke test still had negative autoregressive R2. Treat `rnn_paper_db` as a diagnostic route, not the current best RNN baseline.
- The current most promising usable RNN baseline route is `per_sample_minmax`, direct `S(z+1)` prediction with sigmoid output, front-10-cm z downsampling, autoregressive checkpoint selection, low scheduled-sampling feedback, and optional gradient clipping.
- `train_luna_rnn.py` now supports `--no-detach-feedback` for differentiable truncated rollout feedback. Early smoke tests did not immediately beat the previous detached-feedback `z10cm_201` result, so use it experimentally rather than as the default.
- `results_direct_z10cm_101_perminmax_stable_v1` completed with positive but modest autoregressive performance: full-test `autoregressive_r2=0.4174`, `autoregressive_final_r2=0.1584`, selected epoch 5, and early stopping at epoch 13.
- `results_direct_z10cm_201_perminmax_stable_v1` completed with weaker autoregressive performance than the 101-point task: full-test `autoregressive_r2=0.2867`, `autoregressive_final_r2=0.0748`, selected epoch 15.
- `results_warm_scheduled_z10cm_101_perminmax_v1` completed but did not improve rollout stability: full-test `autoregressive_r2=0.2830`, `autoregressive_final_r2=0.1686`.
- `results_direct_z10cm_51_perminmax_stable_v1` is the current best short RNN autoregressive baseline: full-test `autoregressive_r2=0.4296`, `autoregressive_final_r2=0.1774`, selected epoch 2.
- The next RNN question is now attribution: verify whether autoregressive failure comes mainly from our PyTorch adaptation or from Luna data difficulty by cross-running original RNNnonlinear data/code and low-dimensional Luna tasks.
- RNN attribution runs started on cloud:
  - PyTorch Luna RNN on low-dimensional Luna `z10cm_51_lambda251` completed/started first; `z10cm_101_lambda251` is the next queued comparison.
  - Original RNNnonlinear archive download from Zenodo `4304771` is running under `/mnt/Luna.jl-master/rnn_original_data`.
  - Isolated original-code environment setup is running under `/mnt/Luna.jl-master/rnn_original_code_env` using Miniconda + Python 3.7 + TensorFlow 1.x/Keras 2.x.
  - Original-code watchers are queued for two tasks: original code on original data, and original code on Luna `z10cm_51_lambda251` mat5 data.
- Original Keras RNN code on low-dimensional Luna data completed. On `z10cm_51_lambda251`, it still shows high stepwise R2 but weak autoregressive R2:
  - `per_sample_minmax`: best among 1/3/5 epoch checks is `autoregressive_r2=0.3161` at 3 epochs.
  - `rnn_paper_db`: best among 1/3/5 epoch checks is `autoregressive_r2=0.0947` at 3 epochs.
  This reduces the probability that the Luna autoregressive failure is caused only by the PyTorch port.
- A stricter preprocessing attribution run is now active: `export_luna_raw_power_mat.py` exports Luna raw linear power maps, then original `load_data.py` applies its own `normalization='dBm'`. Cloud run script: `/mnt/Luna.jl-master/rnn_original_code_env/run_raw_power_dBm_luna_lambda251.sh`.
- A follow-up Luna-specific raw-power preprocessing sweep is running under `/mnt/Luna.jl-master/rnn_preprocess_experiments`. It exports the same front-10-cm / 51-z / 251-wavelength Luna raw-power task with less aggressive RNN targets: `bandwise_db_p99p9_floor80`, `per_sample_db_p99p5_floor80`, `per_sample_db_p99p9_floor80`, `per_sample_db_max_floor80`, and `log1p_p99p9`. A 300-sample smoke export showed that `bandwise_db_p99p9_floor80` reduces the fraction of exact-zero targets to about `22%`, compared with about `61%` for original global `-55 dB` dBm scaling.
- Original Zenodo data download is being retried with resumable `curl -C -` because the previous `wget` run stopped at a partial `123M` zip. Cloud run script: `/mnt/Luna.jl-master/rnn_original_data/run_original_data_attribution_resumable.sh`.

## Active manuscript and presentation files

- `transformer_conference_paper_draft.tex`
- `script.md`
- `oral_presentation_visual_prompts.md`
- `summer_plan_review_and_inverse_design.md`
- `transformer_rnn_evaluation_summary.md`
- `model_finetuning_commands.md`

Binary presentation and submission files are intentionally not tracked in normal Git unless Git LFS is configured.

## Next candidates

1. Run the RNN attribution matrix: original code on original data, PyTorch Luna RNN code on original data, and PyTorch Luna RNN code on low-dimensional Luna `lambda251` tasks.
2. Check whether the raw-power plus original `load_data(..., 'dBm')` run improves Luna autoregressive R2. If it does not, treat Luna data complexity as the main cause.
3. Use `z10cm_51` or `z10cm_101` only as short-horizon RNN baselines unless the attribution runs show a fixable code mismatch.
4. Keep `rnn_paper_db` for normalization diagnostics unless a less sparse floor/reference setting clearly improves autoregressive rollout.
5. Compare the relative-db preprocessing sweep against the current best `per_sample_minmax` short RNN baseline.
6. Compare temporal CNN and temporal Transformer under the same early-dense and temporal-selection metrics.
7. Start forward-surrogate-based inverse design before training a standalone inverse model.
