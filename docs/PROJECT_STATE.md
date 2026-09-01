# Project state

Last updated: 2026-09-01

## Current objective

Prepare the project for reliable continued development while preserving the current research workflow around Luna/UPPE simulation, Transformer/CNN forward surrogates, Luna RNN baselines, and future inverse design.

## Current research lines

- Transformer forward surrogate for full spectral-evolution prediction in gas-filled AR-HCFs.
- Temporal CNN as a strong engineering baseline for propagation-map prediction.
- Luna RNN adaptation to understand stepwise prediction versus autoregressive rollout.
- Early-dense data generation to improve early propagation detail.
- Global-log or absolute-intensity modeling as a separate task from normalized spectrum-shape prediction.
- Inverse design and reinforcement learning as future layers built on top of verified forward surrogates.

## Known metrics and results

- Transformer temporal v3 reported approximately `Final R2 = 0.9635`.
- Transformer temporal v3 reported approximately `Temporal R2 = 0.9650`.
- Transformer temporal v3 reported approximately `Temporal UV R2 = 0.8927`.
- Luna RNN stepwise evaluation was high, around `R2 = 0.9944`, but autoregressive rollout failed in the earlier run, around `R2 = -0.4099`.

## Active manuscript and presentation files

- `transformer_conference_paper_draft.tex`
- `script.md`
- `oral_presentation_visual_prompts.md`
- `summer_plan_review_and_inverse_design.md`
- `transformer_rnn_evaluation_summary.md`
- `model_finetuning_commands.md`

Binary presentation and submission files are intentionally not tracked in normal Git unless Git LFS is configured.

## Next candidates

1. Confirm clean Git baseline and remote push.
2. Continue Luna RNN legacy/conditional/scheduled-sampling comparison.
3. Reprocess early-dense data when generation completes.
4. Compare temporal CNN and temporal Transformer under the same early-dense and temporal-selection metrics.
5. Start forward-surrogate-based inverse design before training a standalone inverse model.
