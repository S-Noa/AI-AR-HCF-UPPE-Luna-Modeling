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

## Early-dense data

- Goal: improve early propagation details in the first centimeters.
- Status: generation was in progress during the latest planning discussion.
- Follow-up: reprocess and retrain Transformer/CNN/RNN baselines when available.

## Global-log / absolute-intensity modeling

- Goal: predict physically meaningful log-power scale rather than only per-sample normalized spectral shape.
- Important constraint: do not use sigmoid output activation for unbounded global-log targets; use identity output activation.
- Follow-up: start with a small model smoke test before full temporal-map training.

## Inverse design / reinforcement learning

- Goal: use a trained forward surrogate as a fast environment for parameter search.
- Recommended first step: differentiable or black-box optimization through the forward surrogate, followed by Luna verification.
- RL role: future inverse-design policy layer, not a replacement for the forward surrogate.
