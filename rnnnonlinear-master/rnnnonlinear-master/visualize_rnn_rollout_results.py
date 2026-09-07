#!/usr/bin/env python3
"""Visualize RNN stepwise and autoregressive rollout results.

The original RNNnonlinear Keras scripts and the Luna PyTorch RNN scripts both
save MATLAB files with Y_submit and Y_test arrays. This script reshapes those
flat window predictions back to evolution maps, computes common R2 metrics, and
creates heatmaps for visual comparison.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RNN rollout predictions")
    parser.add_argument("--stepwise-mat", required=True)
    parser.add_argument("--autoregressive-mat", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--test-evo", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--wavelength-points", type=int, default=None)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--normalization-label", default="target space")
    parser.add_argument("--summary-csv", default=None, help="Optional CSV file to append one summary row")
    return parser.parse_args()


def safe_name(name):
    keep = []
    for ch in name.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "-", "_", "+", "/"):
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "experiment"


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def load_pair(path):
    mat = sio.loadmat(path)
    if "Y_submit" not in mat or "Y_test" not in mat:
        raise KeyError(f"{path} must contain Y_submit and Y_test")
    pred = np.asarray(mat["Y_submit"], dtype=np.float32)
    true = np.asarray(mat["Y_test"], dtype=np.float32)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch in {path}: {pred.shape} vs {true.shape}")
    return true, pred


def reshape_evolution(flat, test_evo, steps):
    expected = test_evo * (steps - 1)
    if flat.shape[0] != expected:
        raise ValueError(f"Rows={flat.shape[0]}, expected test_evo*(steps-1)={expected}")
    return flat.reshape(test_evo, steps - 1, flat.shape[1])


def compute_metrics(step_true, step_pred, auto_true, auto_pred):
    per_step_r2 = np.asarray(
        [r2_score(auto_true[:, i, :], auto_pred[:, i, :]) for i in range(auto_true.shape[1])],
        dtype=np.float64,
    )
    per_step_rmse = np.sqrt(np.mean((auto_true - auto_pred) ** 2, axis=(0, 2)))
    sample_r2 = np.asarray(
        [r2_score(auto_true[i], auto_pred[i]) for i in range(auto_true.shape[0])],
        dtype=np.float64,
    )
    sample_final_r2 = np.asarray(
        [r2_score(auto_true[i, -1, :], auto_pred[i, -1, :]) for i in range(auto_true.shape[0])],
        dtype=np.float64,
    )
    metrics = {
        "stepwise_r2": r2_score(step_true, step_pred),
        "autoregressive_r2": r2_score(auto_true, auto_pred),
        "final_autoregressive_r2": r2_score(auto_true[:, -1, :], auto_pred[:, -1, :]),
        "per_step_r2_first10": [float(x) for x in per_step_r2[:10]],
        "per_step_r2_mid_last": [
            float(per_step_r2[len(per_step_r2) // 4]),
            float(per_step_r2[len(per_step_r2) // 2]),
            float(per_step_r2[(3 * len(per_step_r2)) // 4]),
            float(per_step_r2[-1]),
        ],
        "per_step_rmse_first10": [float(x) for x in per_step_rmse[:10]],
        "per_step_rmse_mid_last": [
            float(per_step_rmse[len(per_step_rmse) // 4]),
            float(per_step_rmse[len(per_step_rmse) // 2]),
            float(per_step_rmse[(3 * len(per_step_rmse)) // 4]),
            float(per_step_rmse[-1]),
        ],
        "sample_autoreg_r2_quantiles": {
            str(q): float(v)
            for q, v in zip([0, 5, 25, 50, 75, 95, 100], np.percentile(sample_r2, [0, 5, 25, 50, 75, 95, 100]))
        },
        "sample_final_r2_quantiles": {
            str(q): float(v)
            for q, v in zip(
                [0, 5, 25, 50, 75, 95, 100],
                np.percentile(sample_final_r2, [0, 5, 25, 50, 75, 95, 100]),
            )
        },
    }
    return metrics, per_step_r2, per_step_rmse, sample_r2


def plot_map_triplet(out_path, title, truth, pred, normalization_label):
    err = np.abs(pred - truth)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    maps = [truth.T, pred.T, err.T]
    titles = ["Ground truth", "Prediction", "|Error|"]
    cmaps = ["viridis", "viridis", "magma"]
    vmax = max(float(np.nanmax(truth)), float(np.nanmax(pred)), 1e-6)
    for ax, image, sub_title, cmap in zip(axes, maps, titles, cmaps):
        if sub_title == "|Error|":
            im = ax.imshow(image, origin="lower", aspect="auto", cmap=cmap)
        else:
            im = ax.imshow(image, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(sub_title)
        ax.set_xlabel("Propagation step")
        ax.set_ylabel("Wavelength index")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"{title} ({normalization_label})")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_final_spectrum(out_path, title, truth, step_pred, auto_pred):
    x = np.arange(truth.size)
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(x, truth, label="Ground truth", linewidth=1.8)
    plt.plot(x, step_pred, label="Stepwise", linewidth=1.4)
    plt.plot(x, auto_pred, label="Autoregressive", linewidth=1.4)
    plt.xlabel("Wavelength index")
    plt.ylabel("Target value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_curves(out_dir, prefix, per_step_r2, per_step_rmse, sample_r2):
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(np.arange(1, per_step_r2.size + 1), per_step_r2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Autoregressive rollout step")
    plt.ylabel("Per-step R2")
    plt.title("Per-step autoregressive R2")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_per_step_autoregressive_r2.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.5, 4.2))
    plt.plot(np.arange(1, per_step_rmse.size + 1), per_step_rmse)
    plt.xlabel("Autoregressive rollout step")
    plt.ylabel("RMSE")
    plt.title("Per-step autoregressive RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_per_step_autoregressive_rmse.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.5, 4.2))
    plt.hist(sample_r2, bins=50)
    plt.xlabel("Sample-level autoregressive R2")
    plt.ylabel("Count")
    plt.title("Sample-level rollout quality")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_sample_autoregressive_r2_hist.png", dpi=220)
    plt.close()


def write_text_summary(path, name, args, metrics):
    lines = [
        f"Experiment: {name}",
        f"Stepwise mat: {args.stepwise_mat}",
        f"Autoregressive mat: {args.autoregressive_mat}",
        f"test_evo={args.test_evo}, steps={args.steps}, window_size={args.window_size}",
        f"stepwise_r2={metrics['stepwise_r2']:.6f}",
        f"autoregressive_r2={metrics['autoregressive_r2']:.6f}",
        f"final_autoregressive_r2={metrics['final_autoregressive_r2']:.6f}",
        f"sample_autoreg_r2_quantiles={metrics['sample_autoreg_r2_quantiles']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_summary_csv(path, name, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "experiment",
                    "stepwise_r2",
                    "autoregressive_r2",
                    "final_autoregressive_r2",
                    "median_sample_autoregressive_r2",
                    "q05_sample_autoregressive_r2",
                    "q95_sample_autoregressive_r2",
                ]
            )
        writer.writerow(
            [
                name,
                metrics["stepwise_r2"],
                metrics["autoregressive_r2"],
                metrics["final_autoregressive_r2"],
                metrics["sample_autoreg_r2_quantiles"]["50"],
                metrics["sample_autoreg_r2_quantiles"]["5"],
                metrics["sample_autoreg_r2_quantiles"]["95"],
            ]
        )


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(args.experiment_name)

    step_true, step_pred = load_pair(args.stepwise_mat)
    auto_true_flat, auto_pred_flat = load_pair(args.autoregressive_mat)
    if args.wavelength_points is not None and auto_true_flat.shape[1] != args.wavelength_points:
        raise ValueError(
            f"wavelength_points={args.wavelength_points}, but data has {auto_true_flat.shape[1]}"
        )
    auto_true = reshape_evolution(auto_true_flat, args.test_evo, args.steps)
    auto_pred = reshape_evolution(auto_pred_flat, args.test_evo, args.steps)
    step_true_map = reshape_evolution(step_true, args.test_evo, args.steps)
    step_pred_map = reshape_evolution(step_pred, args.test_evo, args.steps)

    metrics, per_step_r2, per_step_rmse, sample_r2 = compute_metrics(
        step_true, step_pred, auto_true, auto_pred
    )
    payload = {
        "experiment_name": args.experiment_name,
        "stepwise_mat": args.stepwise_mat,
        "autoregressive_mat": args.autoregressive_mat,
        "test_evo": args.test_evo,
        "steps": args.steps,
        "window_size": args.window_size,
        "normalization_label": args.normalization_label,
        **metrics,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_text_summary(out_dir / "metrics_summary.txt", args.experiment_name, args, metrics)
    if args.summary_csv:
        append_summary_csv(args.summary_csv, args.experiment_name, metrics)

    plot_curves(out_dir, prefix, per_step_r2, per_step_rmse, sample_r2)
    for idx in args.sample_indices:
        if idx < 0 or idx >= auto_true.shape[0]:
            continue
        plot_map_triplet(
            out_dir / f"sample_{idx:03d}_stepwise_evolution.png",
            f"{args.experiment_name} sample {idx}: stepwise",
            step_true_map[idx],
            step_pred_map[idx],
            args.normalization_label,
        )
        plot_map_triplet(
            out_dir / f"sample_{idx:03d}_autoregressive_evolution.png",
            f"{args.experiment_name} sample {idx}: autoregressive",
            auto_true[idx],
            auto_pred[idx],
            args.normalization_label,
        )
        plot_final_spectrum(
            out_dir / f"sample_{idx:03d}_final_spectrum.png",
            f"{args.experiment_name} sample {idx}: final spectrum",
            auto_true[idx, -1],
            step_pred_map[idx, -1],
            auto_pred[idx, -1],
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
