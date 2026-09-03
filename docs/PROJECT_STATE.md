# Project state

Last updated: 2026-09-03

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
- `open_source_legacy` z1401 training is running as the pure one-step baseline; the `conditional_legacy` one-step run was stopped after showing high stepwise R2 but unstable autoregressive R2.
- `features_z + scheduled_sampling` z1401 training is running to target autoregressive drift.

## Active manuscript and presentation files

- `transformer_conference_paper_draft.tex`
- `script.md`
- `oral_presentation_visual_prompts.md`
- `summer_plan_review_and_inverse_design.md`
- `transformer_rnn_evaluation_summary.md`
- `model_finetuning_commands.md`

Binary presentation and submission files are intentionally not tracked in normal Git unless Git LFS is configured.

## Next candidates

1. Monitor `open_source_legacy` and `features_z_scheduled_z1401_v1` training logs.
2. Compare `stepwise_r2`, `autoregressive_r2`, and `autoregressive_final_r2` after enough epochs.
3. If scheduled sampling still fails, test rollout loss or a downsampled early-10-cm diagnostic task.
4. Compare temporal CNN and temporal Transformer under the same early-dense and temporal-selection metrics.
5. Start forward-surrogate-based inverse design before training a standalone inverse model.
