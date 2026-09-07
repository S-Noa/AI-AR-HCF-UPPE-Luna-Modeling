#!/usr/bin/env python3
"""Diagnose why original dBm preprocessing does not rescue Luna RNN rollout.

This script compares the original SC_spec_251 dBm task with a Luna AR-HCF task
exported as raw linear power and normalized by the original RNNnonlinear dBm
pipeline.  It writes quantitative diagnostics, comparison plots, and a concise
Markdown report suitable for presentation notes.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create diagnostics for Luna raw-power + original dBm RNN failure"
    )
    parser.add_argument(
        "--original-data-h5",
        default="/mnt/Luna.jl-master/rnn_original_data/converted/SC_spec_251_dBm.h5",
        help="Original SC data after dBm conversion, HDF5 with /data=(N,n_lambda,n_z)",
    )
    parser.add_argument(
        "--luna-raw-mat",
        default="/mnt/Luna.jl-master/rnn_original_code_env/"
        "luna_t0p6_earlydense_z10cm_51_lambda251_rawpower_originalcode.mat",
        help="Luna raw linear-power .mat, data=(N,n_lambda,n_z)",
    )
    parser.add_argument(
        "--original-stepwise-mat",
        default="/mnt/Luna.jl-master/rnn_original_data/results_pytorch_SC_spec_251_dBm_v1/"
        "stepwise_predictions.mat",
    )
    parser.add_argument(
        "--original-autoreg-mat",
        default="/mnt/Luna.jl-master/rnn_original_data/results_pytorch_SC_spec_251_dBm_v1/"
        "autoregressive_predictions.mat",
    )
    parser.add_argument(
        "--luna-stepwise-mat",
        default="/mnt/Luna.jl-master/rnn_original_code_env/raw_power_dBm_luna_lambda251/"
        "dBm_3e/results/stepwise.mat",
    )
    parser.add_argument(
        "--luna-autoreg-mat",
        default="/mnt/Luna.jl-master/rnn_original_code_env/raw_power_dBm_luna_lambda251/"
        "dBm_3e/results/autoregressive.mat",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/Luna.jl-master/rnn_visual_diagnostics",
    )
    parser.add_argument("--original-test-evo", type=int, default=50)
    parser.add_argument("--original-steps", type=int, default=200)
    parser.add_argument("--luna-test-evo", type=int, default=778)
    parser.add_argument("--luna-steps", type=int, default=51)
    parser.add_argument("--db-floor", type=float, default=-55.0)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    return parser.parse_args()


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def load_h5_data(path):
    with h5py.File(path, "r") as f:
        return f["data"][:].astype(np.float32)


def load_mat_data(path):
    mat = sio.loadmat(path)
    return mat["data"].astype(np.float32)


def original_dbm_scale(raw_power, db_floor=-55.0):
    raw_power = np.asarray(raw_power, dtype=np.float64)
    reference = float(np.max(np.abs(raw_power)))
    if reference <= 0:
        raise ValueError("raw_power reference max is non-positive")
    normalized = np.maximum(raw_power / reference, np.finfo(np.float64).tiny)
    db = 10.0 * np.log10(normalized)
    db = np.clip(db, db_floor, 0.0)
    return (db / abs(db_floor) + 1.0).astype(np.float32), reference


def quantiles(values):
    return {
        str(q): float(v)
        for q, v in zip(
            [0, 1, 5, 25, 50, 75, 95, 99, 100],
            np.percentile(np.asarray(values).reshape(-1), [0, 1, 5, 25, 50, 75, 95, 99, 100]),
        )
    }


def summarize_target(name, data):
    values = data.reshape(-1)
    dz = np.abs(np.diff(data, axis=2))
    dlambda = np.abs(np.diff(data, axis=1))
    step_change = dz.mean(axis=(0, 1))
    final = data[:, :, -1]
    early = data[:, :, : min(10, data.shape[2])]
    summary = {
        "name": name,
        "shape": list(data.shape),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "quantiles": quantiles(values),
        "final_mean": float(np.mean(final)),
        "final_std": float(np.std(final)),
        "final_quantiles": quantiles(final),
        "early_first10_mean": float(np.mean(early)),
        "early_first10_std": float(np.std(early)),
        "frac_le_0": float(np.mean(values <= 0.0)),
        "frac_le_0p001": float(np.mean(values <= 0.001)),
        "frac_le_0p01": float(np.mean(values <= 0.01)),
        "frac_le_0p05": float(np.mean(values <= 0.05)),
        "frac_le_0p1": float(np.mean(values <= 0.1)),
        "frac_le_0p2": float(np.mean(values <= 0.2)),
        "frac_ge_0p999": float(np.mean(values >= 0.999)),
        "mean_abs_dz": float(np.mean(dz)),
        "std_abs_dz": float(np.std(dz)),
        "max_mean_abs_dz": float(np.max(step_change)),
        "argmax_mean_abs_dz": int(np.argmax(step_change)),
        "mean_abs_dz_first10": [float(x) for x in step_change[:10]],
        "mean_abs_dz_last10": [float(x) for x in step_change[-10:]],
        "mean_abs_dlambda": float(np.mean(dlambda)),
        "std_abs_dlambda": float(np.std(dlambda)),
        "final_between_sample_std_mean": float(final.std(axis=0).mean()),
    }
    return summary, step_change


def load_prediction_pair(stepwise_mat, autoreg_mat, test_evo, steps):
    step = sio.loadmat(stepwise_mat)
    auto = sio.loadmat(autoreg_mat)
    step_true = step["Y_test"].astype(np.float32)
    step_pred = step["Y_submit"].astype(np.float32)
    auto_true = auto["Y_test"].astype(np.float32)
    auto_pred = auto["Y_submit"].astype(np.float32)
    expected = test_evo * (steps - 1)
    if auto_true.shape[0] != expected:
        raise ValueError(f"{autoreg_mat} rows={auto_true.shape[0]}, expected {expected}")
    if auto_true.shape != auto_pred.shape:
        raise ValueError(f"Shape mismatch: {auto_true.shape} vs {auto_pred.shape}")
    if step_true.shape != step_pred.shape:
        raise ValueError(f"Shape mismatch: {step_true.shape} vs {step_pred.shape}")
    n_lambda = auto_true.shape[1]
    return {
        "step_true": step_true,
        "step_pred": step_pred,
        "auto_true": auto_true.reshape(test_evo, steps - 1, n_lambda),
        "auto_pred": auto_pred.reshape(test_evo, steps - 1, n_lambda),
        "auto_true_flat": auto_true,
        "auto_pred_flat": auto_pred,
    }


def summarize_predictions(name, pred):
    y = pred["auto_true"]
    p = pred["auto_pred"]
    per_step_r2 = [r2_score(y[:, i, :], p[:, i, :]) for i in range(y.shape[1])]
    per_step_rmse = [
        float(np.sqrt(np.mean((y[:, i, :] - p[:, i, :]) ** 2))) for i in range(y.shape[1])
    ]
    sample_r2 = [r2_score(y[i], p[i]) for i in range(y.shape[0])]
    summary = {
        "name": name,
        "stepwise_r2": r2_score(pred["step_true"], pred["step_pred"]),
        "autoregressive_r2": r2_score(pred["auto_true_flat"], pred["auto_pred_flat"]),
        "final_autoregressive_r2": r2_score(y[:, -1, :], p[:, -1, :]),
        "per_step_r2_first10": [float(x) for x in per_step_r2[:10]],
        "per_step_r2_mid_last": [
            float(per_step_r2[y.shape[1] // 4]),
            float(per_step_r2[y.shape[1] // 2]),
            float(per_step_r2[(3 * y.shape[1]) // 4]),
            float(per_step_r2[-1]),
        ],
        "per_step_rmse_first10": [float(x) for x in per_step_rmse[:10]],
        "per_step_rmse_mid_last": [
            float(per_step_rmse[y.shape[1] // 4]),
            float(per_step_rmse[y.shape[1] // 2]),
            float(per_step_rmse[(3 * y.shape[1]) // 4]),
            float(per_step_rmse[-1]),
        ],
        "sample_autoreg_r2_quantiles": {
            str(q): float(v)
            for q, v in zip([0, 5, 25, 50, 75, 95, 100], np.percentile(sample_r2, [0, 5, 25, 50, 75, 95, 100]))
        },
    }
    return summary, np.asarray(per_step_r2), np.asarray(per_step_rmse), np.asarray(sample_r2)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_hist(original, luna, out_path):
    plt.figure(figsize=(8, 4.8))
    bins = np.linspace(0, 1, 80)
    plt.hist(original.reshape(-1), bins=bins, density=True, alpha=0.6, label="Original SC_spec_251, dBm")
    plt.hist(luna.reshape(-1), bins=bins, density=True, alpha=0.6, label="Luna raw power, original dBm")
    plt.xlabel("Target value in [0, 1]")
    plt.ylabel("Density")
    plt.title("Target-value distribution after dBm scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_profile(original_profile, luna_profile, out_path):
    plt.figure(figsize=(8, 4.8))
    plt.plot(np.linspace(0, 1, original_profile.size), original_profile, label="Original SC_spec_251, dBm")
    plt.plot(np.linspace(0, 1, luna_profile.size), luna_profile, label="Luna raw power, original dBm")
    plt.xlabel("Normalized propagation step")
    plt.ylabel("Mean |Delta target| between adjacent z steps")
    plt.title("Temporal roughness along propagation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_step_r2(original_r2, luna_r2, out_path):
    plt.figure(figsize=(8, 4.8))
    plt.plot(np.linspace(0, 1, original_r2.size), original_r2, label="Original SC_spec_251, dBm")
    plt.plot(np.linspace(0, 1, luna_r2.size), luna_r2, label="Luna raw power, original dBm")
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Normalized rollout step")
    plt.ylabel("Per-step autoregressive R2")
    plt.title("Autoregressive rollout stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_sample_r2(original_sample_r2, luna_sample_r2, out_path):
    plt.figure(figsize=(8, 4.8))
    plt.boxplot([original_sample_r2, luna_sample_r2], labels=["Original SC", "Luna raw+dBm"], showfliers=False)
    plt.ylabel("Sample-level autoregressive R2")
    plt.title("Rollout quality across test trajectories")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_evolution_examples(name, pred, out_dir, sample_indices):
    y = pred["auto_true"]
    p = pred["auto_pred"]
    for idx in sample_indices:
        if idx < 0 or idx >= y.shape[0]:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        images = [y[idx].T, p[idx].T, np.abs(p[idx].T - y[idx].T)]
        titles = ["Ground truth", "Autoregressive prediction", "|Error|"]
        cmaps = ["viridis", "viridis", "magma"]
        vmax = float(max(np.max(images[0]), np.max(images[1]), 1e-6))
        for ax, image, title, cmap in zip(axes, images, titles, cmaps):
            im = ax.imshow(image, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=None if title == "|Error|" else vmax)
            ax.set_title(title)
            ax.set_xlabel("Rollout step")
            ax.set_ylabel("Wavelength index")
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle(f"{name}: sample {idx}")
        safe = name.lower().replace(" ", "_").replace("+", "plus")
        fig.savefig(out_dir / f"{safe}_sample_{idx:03d}_autoregressive_evolution.png", dpi=200)
        plt.close(fig)


def write_summary_csv(path, target_summaries, prediction_summaries):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "dataset", "metric", "value"])
        for summary in target_summaries:
            for key in [
                "frac_le_0",
                "frac_le_0p01",
                "frac_le_0p1",
                "mean_abs_dz",
                "max_mean_abs_dz",
                "argmax_mean_abs_dz",
                "mean_abs_dlambda",
                "final_between_sample_std_mean",
            ]:
                writer.writerow(["target", summary["name"], key, summary[key]])
        for summary in prediction_summaries:
            for key in ["stepwise_r2", "autoregressive_r2", "final_autoregressive_r2"]:
                writer.writerow(["prediction", summary["name"], key, summary[key]])


def write_report(path, results, plot_paths):
    original_t = results["targets"]["original_sc_dbm"]
    luna_t = results["targets"]["luna_raw_original_dbm"]
    original_p = results["predictions"]["original_sc_pytorch"]
    luna_p = results["predictions"]["luna_raw_original_dbm_keras"]
    md = f"""# Raw-Power + Original dBm Failure Diagnostics

## Main Finding

The failure is not simply due to missing the original dBm preprocessing. The
PyTorch implementation reaches `autoregressive_r2={original_p['autoregressive_r2']:.4f}`
on the original `SC_spec_251` dBm data, while the original Keras RNN reaches only
`autoregressive_r2={luna_p['autoregressive_r2']:.4f}` on Luna raw-power data after
the original `load_data(..., normalization='dBm')` pipeline.

## Quantitative Evidence

| Metric | Original SC_spec_251, dBm | Luna raw power, original dBm |
|---|---:|---:|
| Target fraction <= 0 | {original_t['frac_le_0']:.4f} | {luna_t['frac_le_0']:.4f} |
| Target fraction <= 0.1 | {original_t['frac_le_0p1']:.4f} | {luna_t['frac_le_0p1']:.4f} |
| Mean adjacent-z change | {original_t['mean_abs_dz']:.5f} | {luna_t['mean_abs_dz']:.5f} |
| Max mean adjacent-z change | {original_t['max_mean_abs_dz']:.5f} | {luna_t['max_mean_abs_dz']:.5f} |
| Argmax z-change step | {original_t['argmax_mean_abs_dz']} | {luna_t['argmax_mean_abs_dz']} |
| Stepwise R2 | {original_p['stepwise_r2']:.4f} | {luna_p['stepwise_r2']:.4f} |
| Autoregressive R2 | {original_p['autoregressive_r2']:.4f} | {luna_p['autoregressive_r2']:.4f} |
| Final autoregressive R2 | {original_p['final_autoregressive_r2']:.4f} | {luna_p['final_autoregressive_r2']:.4f} |
| Median sample-level autoregressive R2 | {original_p['sample_autoreg_r2_quantiles']['50']:.4f} | {luna_p['sample_autoreg_r2_quantiles']['50']:.4f} |

## Interpretation

Compared with the original SC dataset, the Luna AR-HCF data become much sparser
after the original `global max + -55 dB` scaling: about `{luna_t['frac_le_0']:.1%}`
of Luna values are clipped to zero, compared with `{original_t['frac_le_0']:.1%}`
for the original SC data. The Luna early propagation is also more abrupt: its
largest average z-step change occurs immediately at the input side
(`argmax={luna_t['argmax_mean_abs_dz']}`), while the original SC data peaks later
(`argmax={original_t['argmax_mean_abs_dz']}`).

These two effects make the local-window autoregressive assumption unstable. The
RNN can fit one-step targets under teacher forcing, but once its own predictions
are fed back, weak features and early-z deviations are amplified through the
rollout.

## Figures

- Target distribution: `{plot_paths['hist']}`
- Temporal roughness: `{plot_paths['dz_profile']}`
- Per-step autoregressive R2: `{plot_paths['per_step_r2']}`
- Sample-level R2 distribution: `{plot_paths['sample_r2']}`

## PPT Wording

The original RNN workflow and our PyTorch implementation are valid on the
original SC task. However, Luna UPPE-based AR-HCF propagation produces a sparser
and more abrupt recurrent state space after dBm scaling. Therefore the weak
autoregressive performance on Luna is mainly a task/data issue, not simply a
code migration or normalization mismatch.
"""
    path.write_text(md, encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_data = load_h5_data(args.original_data_h5)
    luna_raw = load_mat_data(args.luna_raw_mat)
    luna_scaled, luna_reference = original_dbm_scale(luna_raw, args.db_floor)

    original_target_summary, original_dz = summarize_target(
        "Original SC_spec_251, dBm", original_data
    )
    luna_target_summary, luna_dz = summarize_target(
        "Luna raw power, original dBm", luna_scaled
    )
    luna_target_summary["raw_power_reference_max"] = luna_reference

    original_pred = load_prediction_pair(
        args.original_stepwise_mat,
        args.original_autoreg_mat,
        args.original_test_evo,
        args.original_steps,
    )
    luna_pred = load_prediction_pair(
        args.luna_stepwise_mat,
        args.luna_autoreg_mat,
        args.luna_test_evo,
        args.luna_steps,
    )
    original_pred_summary, original_per_step_r2, original_rmse, original_sample_r2 = summarize_predictions(
        "Original SC PyTorch RNN", original_pred
    )
    luna_pred_summary, luna_per_step_r2, luna_rmse, luna_sample_r2 = summarize_predictions(
        "Luna raw power original dBm Keras RNN", luna_pred
    )

    plot_paths = {
        "hist": str(out_dir / "target_value_hist_original_vs_luna.png"),
        "dz_profile": str(out_dir / "mean_abs_dz_profile_original_vs_luna.png"),
        "per_step_r2": str(out_dir / "per_step_autoreg_r2_original_vs_luna.png"),
        "sample_r2": str(out_dir / "sample_r2_distribution_original_vs_luna.png"),
    }
    plot_hist(original_data, luna_scaled, plot_paths["hist"])
    plot_profile(original_dz, luna_dz, plot_paths["dz_profile"])
    plot_per_step_r2(original_per_step_r2, luna_per_step_r2, plot_paths["per_step_r2"])
    plot_sample_r2(original_sample_r2, luna_sample_r2, plot_paths["sample_r2"])
    plot_evolution_examples("Original SC PyTorch RNN", original_pred, out_dir, args.sample_indices)
    plot_evolution_examples("Luna raw dBm Keras RNN", luna_pred, out_dir, args.sample_indices)

    results = {
        "targets": {
            "original_sc_dbm": original_target_summary,
            "luna_raw_original_dbm": luna_target_summary,
        },
        "predictions": {
            "original_sc_pytorch": original_pred_summary,
            "luna_raw_original_dbm_keras": luna_pred_summary,
        },
        "plot_paths": plot_paths,
    }
    save_json(out_dir / "rawpower_dBm_failure_diagnostics.json", results)
    write_summary_csv(
        out_dir / "rawpower_dBm_failure_summary.csv",
        [original_target_summary, luna_target_summary],
        [original_pred_summary, luna_pred_summary],
    )
    write_report(out_dir / "rawpower_dBm_failure_diagnostics.md", results, plot_paths)
    print(json.dumps(results["predictions"], indent=2))
    print(f"Saved diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
