#!/usr/bin/env python3
"""Evaluate and visualize Luna RNNnonlinear-style prediction files.

This script reads the existing train_luna_rnn.py outputs and produces
train_mlp.py-style final-spectrum and temporal-evolution diagnostics for both
teacher-forced one-step predictions and autoregressive rollout predictions.
"""

import argparse
import json
import logging
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


LUNA_STYLE_DB_MIN = -40.0
LUNA_STYLE_CMAP = "viridis"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Luna RNN evaluation outputs")
    parser.add_argument("--results-dir", required=True, help="Directory containing RNN metrics/history/prediction .mat files")
    parser.add_argument("--output-dir", default=None, help="Directory for generated evaluation figures")
    parser.add_argument("--wavelength-min-nm", type=float, default=200.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=2500.0)
    parser.add_argument("--fiber-length-cm", type=float, default=50.0)
    parser.add_argument("--early-z-max-cm", type=float, default=10.0)
    parser.add_argument("--uv-min-nm", type=float, default=200.0)
    parser.add_argument("--uv-max-nm", type=float, default=700.0)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Test-evolution indices to visualize")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mat_predictions(path):
    """Load Y_submit/Y_test from MATLAB v5 or HDF5/v7.3-style files."""
    scipy_error = None
    try:
        mat = sio.loadmat(path)
        return np.asarray(mat["Y_submit"], dtype=np.float32), np.asarray(mat["Y_test"], dtype=np.float32)
    except Exception as exc:
        scipy_error = exc

    try:
        with h5py.File(path, "r") as f:
            return np.asarray(f["Y_submit"][:], dtype=np.float32), np.asarray(f["Y_test"][:], dtype=np.float32)
    except Exception as h5_error:
        raise RuntimeError(
            f"Failed to load {path} as MATLAB v5 or HDF5. "
            f"scipy error: {scipy_error}; hdf5 error: {h5_error}"
        ) from h5_error


def reshape_temporal(flat, metrics):
    """Return temporal array as (n_evolutions, n_steps_minus_one, n_lambda)."""
    n_evo = int(metrics["test_evolutions"])
    n_lambda = int(metrics["data_shape"][1])
    n_steps = int(metrics["data_shape"][2]) - 1
    arr = np.asarray(flat, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape == (n_evo, n_steps, n_lambda):
            return arr
        if arr.shape == (n_evo, n_lambda, n_steps):
            return np.transpose(arr, (0, 2, 1))
    if arr.ndim != 2:
        raise ValueError(f"Unsupported prediction shape: {arr.shape}")
    expected_rows = n_evo * n_steps
    if arr.shape != (expected_rows, n_lambda):
        raise ValueError(f"Expected {(expected_rows, n_lambda)} from metrics, got {arr.shape}")
    return arr.reshape(n_evo, n_steps, n_lambda)


def finite_arrays(y_true, y_pred):
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def r2_score_np(y_true, y_pred, eps=1e-12):
    true, pred = finite_arrays(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + eps))


def regression_metrics(y_true, y_pred, prefix):
    true, pred = finite_arrays(y_true, y_pred)
    if true.size == 0:
        return {
            f"{prefix}_MSE": float("nan"),
            f"{prefix}_RMSE": float("nan"),
            f"{prefix}_MAE": float("nan"),
            f"{prefix}_R2": float("nan"),
        }
    mse = float(np.mean((true - pred) ** 2))
    return {
        f"{prefix}_MSE": mse,
        f"{prefix}_RMSE": float(np.sqrt(mse)),
        f"{prefix}_MAE": float(np.mean(np.abs(true - pred))),
        f"{prefix}_R2": r2_score_np(y_true, y_pred),
    }


def normalized_to_luna_db(values, dBmin=LUNA_STYLE_DB_MIN):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    return dBmin + (0.0 - dBmin) * values


def choose_indices(n_available, requested, n_samples):
    if requested:
        indices = [idx for idx in requested if 0 <= idx < n_available]
        if indices:
            return indices[:n_samples]
        logging.warning("No valid requested indices; falling back to evenly spaced samples")
    n_samples = min(n_samples, n_available)
    if n_samples <= 0:
        return []
    return np.linspace(0, n_available - 1, n_samples, dtype=int).tolist()


def plot_training_curves(history, metrics, output_path):
    epochs = [row["epoch"] for row in history]
    train = [row["train_mse"] for row in history]
    step_mse = [row["test_stepwise_mse"] for row in history]
    step_r2 = [row["test_stepwise_r2"] for row in history]
    rollout_mse = [row.get("train_rollout_mse", np.nan) for row in history]
    autoreg_epochs = [
        row["epoch"] for row in history
        if row.get("test_autoregressive_r2") is not None
    ]
    autoreg_r2 = [
        row["test_autoregressive_r2"] for row in history
        if row.get("test_autoregressive_r2") is not None
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(epochs, train, label="Train one-step MSE", linewidth=2)
    if np.any(np.isfinite(rollout_mse)):
        ax1.plot(epochs, rollout_mse, label="Train rollout MSE", linewidth=1.7)
    ax1.plot(epochs, step_mse, label="Test stepwise MSE", linewidth=2)
    ax1.axhline(metrics.get("autoregressive_mse", np.nan), color="tab:red", linestyle="--",
                label="Autoregressive MSE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.set_title("RNN One-Step Training")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, step_r2, label="Test stepwise R2", linewidth=2, color="tab:blue")
    if autoreg_epochs:
        ax2.plot(autoreg_epochs, autoreg_r2, "o-", label="Periodic autoregressive R2",
                 color="tab:purple", markersize=4)
    ax2.axhline(metrics.get("autoregressive_r2", np.nan), color="tab:red", linestyle="--",
                label="Autoregressive R2")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R2")
    ax2.set_title("Stepwise vs Autoregressive R2")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_final_spectra(y_true, y_pred, output_path, wavelength_range, indices, y_scale="normalized"):
    n_lambda = y_true.shape[-1]
    wl = np.linspace(wavelength_range[0], wavelength_range[1], n_lambda)
    fig, axes = plt.subplots(len(indices), 1, figsize=(13, 3.0 * len(indices)))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        true = y_true[idx, -1]
        pred = y_pred[idx, -1]
        if y_scale == "db":
            true = normalized_to_luna_db(true)
            pred = normalized_to_luna_db(pred)
            ylabel = "Luna-like SED (dB)"
        else:
            ylabel = "Normalized Log Power"
        ax.plot(wl, true, label="Ground Truth", linewidth=2)
        ax.plot(wl, pred, label="RNN Prediction", linewidth=1.8, linestyle="--")
        ax.set_xlim(wavelength_range)
        ax.set_ylabel(ylabel)
        ax.set_title(f"test evolution[{idx}] final spectrum")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Wavelength (nm)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_temporal_evolution(y_true, y_pred, output_path, wavelength_range, z_cm, indices,
                            z_range_cm=None, title_suffix=""):
    n_lambda = y_true.shape[-1]
    wl = np.linspace(wavelength_range[0], wavelength_range[1], n_lambda)
    fig, axes = plt.subplots(len(indices), 3, figsize=(18, 4.0 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        true = y_true[idx]
        pred = y_pred[idx]
        z_plot = z_cm
        if z_range_cm is not None:
            mask = (z_cm >= z_range_cm[0]) & (z_cm <= z_range_cm[1])
            if np.count_nonzero(mask) < 2:
                logging.warning("Too few z points in requested zoom; using full z range for sample %s", idx)
                mask = np.ones_like(z_cm, dtype=bool)
            true = true[mask]
            pred = pred[mask]
            z_plot = z_cm[mask]

        true_db = normalized_to_luna_db(true)
        pred_db = normalized_to_luna_db(pred)
        diff_db = np.abs(pred_db - true_db)

        panels = [
            (true_db, "Ground Truth", LUNA_STYLE_CMAP, LUNA_STYLE_DB_MIN, 0.0, "SED (dB)"),
            (pred_db, "Prediction", LUNA_STYLE_CMAP, LUNA_STYLE_DB_MIN, 0.0, "SED (dB)"),
            (diff_db, "|Prediction - Truth|", "Reds", 0.0, max(float(np.nanmax(diff_db)), 1.0),
             "|Delta SED| (dB)"),
        ]
        for col, (data, title, cmap, vmin, vmax, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            im = ax.pcolormesh(wl, z_plot, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Distance (cm)\ntest[{idx}]")
            if row == len(indices) - 1:
                ax.set_xlabel("Wavelength (nm)")
            ax.set_xlim(wavelength_range)
            if z_range_cm is not None:
                ax.set_ylim(z_range_cm)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)

    title = "RNN Temporal Evolution"
    if title_suffix:
        title += f" ({title_suffix})"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def evaluate_prediction_set(name, y_true, y_pred, wavelength_range, z_cm, uv_range, early_z_max_cm):
    metrics = {}
    metrics.update(regression_metrics(y_true[:, -1, :], y_pred[:, -1, :], f"{name}_Final"))
    metrics.update(regression_metrics(y_true, y_pred, f"{name}_Temporal"))

    wl = np.linspace(wavelength_range[0], wavelength_range[1], y_true.shape[-1])
    uv_mask = (wl >= uv_range[0]) & (wl <= uv_range[1])
    if np.any(uv_mask):
        metrics.update(regression_metrics(y_true[:, -1, uv_mask], y_pred[:, -1, uv_mask], f"{name}_Final_UV"))
        metrics.update(regression_metrics(y_true[..., uv_mask], y_pred[..., uv_mask], f"{name}_Temporal_UV"))

    early_mask = z_cm <= early_z_max_cm
    if np.any(early_mask):
        metrics.update(regression_metrics(y_true[:, early_mask, :], y_pred[:, early_mask, :], f"{name}_Early_Z"))
        if np.any(uv_mask):
            metrics.update(regression_metrics(
                y_true[:, early_mask, :][..., uv_mask],
                y_pred[:, early_mask, :][..., uv_mask],
                f"{name}_Early_UV"
            ))
    return metrics


def process_prediction_file(label, mat_path, base_metrics, args, output_dir, indices, z_cm):
    logging.info("Loading %s predictions from %s", label, mat_path)
    y_pred_flat, y_true_flat = load_mat_predictions(mat_path)
    y_pred = reshape_temporal(y_pred_flat, base_metrics)
    y_true = reshape_temporal(y_true_flat, base_metrics)
    logging.info("%s temporal shape: %s", label, y_pred.shape)

    wavelength_range = (args.wavelength_min_nm, args.wavelength_max_nm)
    metrics = evaluate_prediction_set(
        label, y_true, y_pred, wavelength_range, z_cm,
        (args.uv_min_nm, args.uv_max_nm), args.early_z_max_cm
    )
    plot_final_spectra(
        y_true, y_pred, os.path.join(output_dir, f"predictions_{label}.png"),
        wavelength_range, indices, y_scale="normalized"
    )
    plot_final_spectra(
        y_true, y_pred, os.path.join(output_dir, f"predictions_{label}_db.png"),
        wavelength_range, indices, y_scale="db"
    )
    plot_temporal_evolution(
        y_true, y_pred, os.path.join(output_dir, f"temporal_evolution_{label}.png"),
        wavelength_range, z_cm, indices, title_suffix=label
    )
    plot_temporal_evolution(
        y_true, y_pred, os.path.join(output_dir, f"temporal_evolution_{label}_z0_2cm.png"),
        wavelength_range, z_cm, indices, z_range_cm=(0.0, 2.0), title_suffix=f"{label}, z=0-2 cm"
    )
    plot_temporal_evolution(
        y_true, y_pred, os.path.join(output_dir, f"temporal_evolution_{label}_z0_10cm.png"),
        wavelength_range, z_cm, indices, z_range_cm=(0.0, 10.0), title_suffix=f"{label}, z=0-10 cm"
    )
    return metrics


def main():
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(args.log_file or os.path.join(output_dir, "evaluate_luna_rnn.log"))

    metrics_path = os.path.join(args.results_dir, "metrics.json")
    history_path = os.path.join(args.results_dir, "history.json")
    base_metrics = load_json(metrics_path)
    history = load_json(history_path)
    logging.info("Loaded base metrics from %s", metrics_path)

    n_steps = int(base_metrics["data_shape"][2]) - 1
    z_cm = np.linspace(args.fiber_length_cm / n_steps, args.fiber_length_cm, n_steps)
    indices = choose_indices(int(base_metrics["test_evolutions"]), args.indices, args.n_samples)
    logging.info("Visualization indices: %s", indices)

    plot_training_curves(history, base_metrics, os.path.join(output_dir, "training_curves.png"))

    all_metrics = {
        "source_results_dir": args.results_dir,
        "wavelength_range_nm": [args.wavelength_min_nm, args.wavelength_max_nm],
        "fiber_length_cm": args.fiber_length_cm,
        "early_z_max_cm": args.early_z_max_cm,
        "base_training_metrics": base_metrics,
    }
    all_metrics.update(process_prediction_file(
        "stepwise",
        os.path.join(args.results_dir, "stepwise_predictions.mat"),
        base_metrics, args, output_dir, indices, z_cm
    ))
    all_metrics.update(process_prediction_file(
        "autoregressive",
        os.path.join(args.results_dir, "autoregressive_predictions.mat"),
        base_metrics, args, output_dir, indices, z_cm
    ))

    output_metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logging.info("Saved RNN evaluation metrics to %s", output_metrics_path)


if __name__ == "__main__":
    main()
