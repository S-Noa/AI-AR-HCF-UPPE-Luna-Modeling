#!/usr/bin/env python3
"""Compare data dynamics of the original SC and Luna RNN tasks.

The comparison intentionally uses the original global dBm mapping for both
targets. It therefore separates differences in the recurrent target space from
differences caused by using two unrelated display normalizations.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-data-h5", required=True)
    parser.add_argument("--luna-raw-mat", required=True)
    parser.add_argument("--original-autoreg-mat", required=True)
    parser.add_argument("--luna-autoreg-mat", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--original-test-evolutions", type=int, default=50)
    parser.add_argument("--original-steps", type=int, default=200)
    parser.add_argument("--luna-test-evolutions", type=int, default=778)
    parser.add_argument("--luna-steps", type=int, default=51)
    parser.add_argument("--db-floor", type=float, default=-55.0)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=10_000)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def load_h5(path):
    with h5py.File(path, "r") as handle:
        return handle["data"][:].astype(np.float32)


def load_mat_data(path):
    return sio.loadmat(path)["data"].astype(np.float32)


def original_dbm_scale(raw_power, floor):
    raw_power = np.asarray(raw_power, dtype=np.float64)
    reference = float(np.max(np.abs(raw_power)))
    if reference <= 0:
        raise ValueError("Luna raw-power reference is non-positive")
    scaled = np.maximum(raw_power / reference, np.finfo(np.float64).tiny)
    db = np.clip(10.0 * np.log10(scaled), floor, 0.0)
    return (db / abs(floor) + 1.0).astype(np.float32), reference


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    return float("nan") if denom <= 0 else float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def maybe_limit(data, limit, rng):
    if limit <= 0 or data.shape[0] <= limit:
        return data
    indices = rng.choice(data.shape[0], size=limit, replace=False)
    return data[np.sort(indices)]


def per_step_r2(path, test_evolutions, steps):
    result = sio.loadmat(path)
    true = result["Y_test"].astype(np.float32)
    pred = result["Y_submit"].astype(np.float32)
    expected = test_evolutions * (steps - 1)
    if true.shape != pred.shape or true.shape[0] != expected:
        raise ValueError(f"Unexpected prediction shape in {path}: {true.shape}")
    true = true.reshape(test_evolutions, steps - 1, -1)
    pred = pred.reshape(test_evolutions, steps - 1, -1)
    return np.asarray([r2_score(true[:, step], pred[:, step]) for step in range(steps - 1)])


def summarize(name, data, threshold=0.1):
    dz = np.abs(np.diff(data, axis=2))
    dlambda = np.abs(np.diff(data, axis=1))
    step_change = dz.mean(axis=(0, 1))
    sample_early_change = dz[:, :, : min(10, dz.shape[2])].mean(axis=(1, 2))
    sample_total_change = dz.mean(axis=(1, 2))
    occupancy = (data > threshold).mean(axis=(1, 2))
    spectral_roughness = dlambda.mean(axis=(1, 2))
    initial_final_change = np.abs(data[:, :, -1] - data[:, :, 0]).mean(axis=1)
    return {
        "name": name,
        "shape": list(data.shape),
        "fraction_zero": float((data <= 0.0).mean()),
        "fraction_low": float((data <= threshold).mean()),
        "mean_occupancy_above_threshold": float(occupancy.mean()),
        "mean_abs_dz": float(dz.mean()),
        "mean_abs_dlambda": float(dlambda.mean()),
        "max_z_change": float(step_change.max()),
        "max_z_change_normalized_position": float(step_change.argmax() / max(1, step_change.size - 1)),
        "early_to_total_change_ratio": float(sample_early_change.mean() / max(sample_total_change.mean(), 1e-12)),
        "mean_initial_final_change": float(initial_final_change.mean()),
        "sample_early_change": sample_early_change,
        "sample_occupancy": occupancy,
        "sample_spectral_roughness": spectral_roughness,
        "step_change": step_change,
    }


def select_representatives(summary):
    values = summary["sample_early_change"]
    quantiles = [10, 50, 90, 99]
    return quantiles, [int(np.argmin(np.abs(values - np.percentile(values, q)))) for q in quantiles]


def sample_windows(data, window_size, max_windows, rng):
    if data.shape[2] <= window_size:
        raise ValueError("window_size must be smaller than the z dimension")
    candidates = data.shape[0] * (data.shape[2] - window_size)
    count = min(max_windows, candidates)
    flat = rng.choice(candidates, size=count, replace=False)
    starts_per_sample = data.shape[2] - window_size
    samples = flat // starts_per_sample
    starts = flat % starts_per_sample
    histories = np.empty((count, window_size * data.shape[1]), dtype=np.float32)
    targets = np.empty((count, data.shape[1]), dtype=np.float32)
    for row, (sample, start) in enumerate(zip(samples, starts)):
        histories[row] = data[sample, :, start : start + window_size].T.reshape(-1)
        targets[row] = data[sample, :, start + window_size]
    return histories, targets


def nearest_neighbor_pairs(embedding, targets):
    distances, indices = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(embedding).kneighbors(embedding)
    return distances[:, 1], np.abs(targets - targets[indices[:, 1]]).mean(axis=1)


def summarize_neighbor_pairs(history_distance, target_difference, edges):
    edges = np.unique(edges)
    centers, means, counts = [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (history_distance >= left) & (history_distance <= right)
        if mask.any():
            centers.append(float(np.median(history_distance[mask])))
            means.append(float(target_difference[mask].mean()))
            counts.append(int(mask.sum()))
    return {
        "median_history_distance": float(np.median(history_distance)),
        "median_next_step_difference": float(np.median(target_difference)),
        "mean_next_step_difference": float(target_difference.mean()),
        "p90_next_step_difference": float(np.percentile(target_difference, 90)),
        "bin_centers": centers,
        "bin_mean_next_step_difference": means,
        "bin_counts": counts,
    }


def local_ambiguity_pair(original, luna, window_size, max_windows, rng):
    original_histories, original_targets = sample_windows(original, window_size, max_windows, rng)
    luna_histories, luna_targets = sample_windows(luna, window_size, max_windows, rng)
    combined = np.vstack((original_histories, luna_histories))
    standardized = StandardScaler().fit_transform(combined)
    components = min(16, standardized.shape[0] - 1, standardized.shape[1])
    embedding = PCA(n_components=components, svd_solver="randomized", random_state=0).fit_transform(standardized)
    original_embedding = embedding[: original_histories.shape[0]]
    luna_embedding = embedding[original_histories.shape[0] :]
    original_distance, original_difference = nearest_neighbor_pairs(original_embedding, original_targets)
    luna_distance, luna_difference = nearest_neighbor_pairs(luna_embedding, luna_targets)
    edges = np.quantile(np.concatenate((original_distance, luna_distance)), np.linspace(0, 1, 11))
    return (
        summarize_neighbor_pairs(original_distance, original_difference, edges),
        summarize_neighbor_pairs(luna_distance, luna_difference, edges),
        components,
    )


def plot_representatives(original, luna, original_indices, luna_indices, quantiles, path):
    figure, axes = plt.subplots(4, 2, figsize=(11, 11), sharex="col", sharey=True, constrained_layout=True)
    for row, (quantile, original_index, luna_index) in enumerate(zip(quantiles, original_indices, luna_indices)):
        for column, (data, index, title) in enumerate(((original, original_index, "Original SC"), (luna, luna_index, "Luna AR-HCF"))):
            image = axes[row, column].imshow(data[index].T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
            axes[row, column].set_title(f"{title}: early-change P{quantile}")
            axes[row, column].set_ylabel("Wavelength index")
            axes[row, column].set_xlabel("Normalized propagation")
    figure.colorbar(image, ax=axes, shrink=0.75, label="Target value after common dBm scaling")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_distribution(original, luna, path):
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    bins = np.linspace(0, 1, 80)
    axis.hist(original.reshape(-1), bins=bins, density=True, alpha=0.6, label="Original SC")
    axis.hist(luna.reshape(-1), bins=bins, density=True, alpha=0.6, label="Luna AR-HCF")
    axis.set(xlabel="Target value after common dBm scaling", ylabel="Density", title="Target-state distribution")
    axis.legend()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_z_dynamics(original_summary, luna_summary, path):
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.plot(np.linspace(0, 1, original_summary["step_change"].size), original_summary["step_change"], label="Original SC")
    axis.plot(np.linspace(0, 1, luna_summary["step_change"].size), luna_summary["step_change"], label="Luna AR-HCF")
    axis.set(xlabel="Normalized propagation", ylabel="Mean |Delta target|", title="Propagation-direction dynamics")
    axis.legend()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_occupancy(original_summary, luna_summary, path):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    labels = ["Original SC", "Luna AR-HCF"]
    axes[0].bar(labels, [original_summary["mean_occupancy_above_threshold"], luna_summary["mean_occupancy_above_threshold"]])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction above 0.1")
    axes[0].set_title("Effective spectral occupancy")
    axes[1].boxplot([original_summary["sample_spectral_roughness"], luna_summary["sample_spectral_roughness"]], tick_labels=labels, showfliers=False)
    axes[1].set_ylabel("Mean |Delta spectrum|")
    axes[1].set_title("Wavelength-direction roughness")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_ambiguity(original, luna, path):
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.plot(original["bin_centers"], original["bin_mean_next_step_difference"], marker="o", label="Original SC")
    axis.plot(luna["bin_centers"], luna["bin_mean_next_step_difference"], marker="o", label="Luna AR-HCF")
    axis.set(xlabel="Nearest-neighbor history distance in PCA space", ylabel="Mean next-step spectral difference", title="Local 10-step window ambiguity")
    axis.legend()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_overview(original, luna, original_summary, luna_summary, original_r2, luna_r2, original_index, luna_index, path):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, data, index, title in zip(axes[0, :2], (original, luna), (original_index, luna_index), ("Original SC: median early change", "Luna AR-HCF: median early change")):
        image = axis.imshow(data[index].T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axis.set(title=title, xlabel="Normalized propagation", ylabel="Wavelength index")
    figure.colorbar(image, ax=axes[0, :2], shrink=0.8, label="dBm-scaled target")
    bins = np.linspace(0, 1, 70)
    axes[0, 2].hist(original.reshape(-1), bins=bins, density=True, alpha=0.55, label="Original")
    axes[0, 2].hist(luna.reshape(-1), bins=bins, density=True, alpha=0.55, label="Luna")
    axes[0, 2].set(title="Target distribution", xlabel="Value", ylabel="Density")
    axes[0, 2].legend(fontsize=8)
    axes[1, 0].plot(np.linspace(0, 1, original_summary["step_change"].size), original_summary["step_change"], label="Original")
    axes[1, 0].plot(np.linspace(0, 1, luna_summary["step_change"].size), luna_summary["step_change"], label="Luna")
    axes[1, 0].set(title="z-direction change", xlabel="Normalized propagation", ylabel="Mean |Delta|")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].boxplot([original_summary["sample_early_change"], luna_summary["sample_early_change"]], tick_labels=["Original", "Luna"], showfliers=False)
    inset = axes[1, 1].inset_axes([0.57, 0.52, 0.40, 0.40])
    inset.bar([0, 1], [original_summary["mean_occupancy_above_threshold"], luna_summary["mean_occupancy_above_threshold"]], color=["C0", "C1"])
    inset.set_xticks([0, 1], ["O", "L"])
    inset.set_ylim(0, 1)
    inset.set_title("occupancy", fontsize=8)
    axes[1, 1].set(title="Early-z change across samples", ylabel="Mean |Delta|")
    axes[1, 2].plot(np.linspace(0, 1, original_r2.size), original_r2, label="Original")
    axes[1, 2].plot(np.linspace(0, 1, luna_r2.size), luna_r2, label="Luna")
    axes[1, 2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 2].set(title="Autoregressive R2 by rollout step", xlabel="Normalized rollout", ylabel="R2")
    axes[1, 2].legend(fontsize=8)
    figure.savefig(path, dpi=240)
    plt.close(figure)


def write_report(path, original, luna, ambiguity_original, ambiguity_luna):
    text = f"""# Original SC vs Luna AR-HCF Data-Dynamics Comparison

## Scope

Both targets use the original RNN paper's global dBm mapping. The original SC
and Luna propagation axes are shown in normalized coordinates; this is a
comparison of recurrent target dynamics, not a claim that their physical length
units are identical.

## Quantitative Summary

| Metric | Original SC | Luna AR-HCF |
|---|---:|---:|
| Target values clipped to zero | {original['fraction_zero']:.2%} | {luna['fraction_zero']:.2%} |
| Target values at or below 0.1 | {original['fraction_low']:.2%} | {luna['fraction_low']:.2%} |
| Mean effective occupancy above 0.1 | {original['mean_occupancy_above_threshold']:.4f} | {luna['mean_occupancy_above_threshold']:.4f} |
| Mean adjacent-z change | {original['mean_abs_dz']:.5f} | {luna['mean_abs_dz']:.5f} |
| Normalized position of maximum z change | {original['max_z_change_normalized_position']:.3f} | {luna['max_z_change_normalized_position']:.3f} |
| Early/total z-change ratio | {original['early_to_total_change_ratio']:.4f} | {luna['early_to_total_change_ratio']:.4f} |
| Mean wavelength roughness | {original['mean_abs_dlambda']:.5f} | {luna['mean_abs_dlambda']:.5f} |
| Mean initial-to-final spectral change | {original['mean_initial_final_change']:.5f} | {luna['mean_initial_final_change']:.5f} |
| Median nearest-history distance in shared PCA space | {ambiguity_original['median_history_distance']:.5f} | {ambiguity_luna['median_history_distance']:.5f} |
| Median local next-step difference | {ambiguity_original['median_next_step_difference']:.5f} | {ambiguity_luna['median_next_step_difference']:.5f} |
| P90 local next-step difference | {ambiguity_original['p90_next_step_difference']:.5f} | {ambiguity_luna['p90_next_step_difference']:.5f} |

## Interpretation

The plots establish descriptive differences in target sparsity, early-z
restructuring, spectral roughness, and local-window next-step variation. The
local-window analysis standardizes both datasets and projects their histories
into one shared PCA space. Larger nearest-history distances or larger next-step
differences therefore indicate a broader or less locally stable recurrent
target distribution under this representation. They do not, on their own,
establish a unique physical causal mechanism.
"""
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original = maybe_limit(load_h5(args.original_data_h5), args.max_trajectories, rng)
    luna_raw = maybe_limit(load_mat_data(args.luna_raw_mat), args.max_trajectories, rng)
    luna, reference = original_dbm_scale(luna_raw, args.db_floor)
    original_summary = summarize("Original SC", original)
    luna_summary = summarize("Luna AR-HCF", luna)
    original_quantiles, original_indices = select_representatives(original_summary)
    luna_quantiles, luna_indices = select_representatives(luna_summary)
    original_r2 = per_step_r2(args.original_autoreg_mat, args.original_test_evolutions, args.original_steps)
    luna_r2 = per_step_r2(args.luna_autoreg_mat, args.luna_test_evolutions, args.luna_steps)
    ambiguity_original, ambiguity_luna, ambiguity_components = local_ambiguity_pair(
        original, luna, args.window_size, args.max_windows, rng
    )

    plot_representatives(original, luna, original_indices, luna_indices, original_quantiles, out_dir / "representative_targets_original_vs_luna.png")
    plot_distribution(original, luna, out_dir / "target_distribution_original_vs_luna.png")
    plot_z_dynamics(original_summary, luna_summary, out_dir / "z_dynamics_original_vs_luna.png")
    plot_occupancy(original_summary, luna_summary, out_dir / "spectral_occupancy_original_vs_luna.png")
    plot_ambiguity(ambiguity_original, ambiguity_luna, out_dir / "local_window_ambiguity_original_vs_luna.png")
    plot_overview(original, luna, original_summary, luna_summary, original_r2, luna_r2, original_indices[1], luna_indices[1], out_dir / "data_dynamics_ppt_overview.png")

    metrics = {
        "comparison": {"db_floor": args.db_floor, "window_size": args.window_size, "seed": args.seed, "luna_raw_reference_max": reference},
        "original_sc": {key: value for key, value in original_summary.items() if not isinstance(value, np.ndarray)},
        "luna_ar_hcf": {key: value for key, value in luna_summary.items() if not isinstance(value, np.ndarray)},
        "representative_quantiles": {"quantiles": original_quantiles, "original_indices": original_indices, "luna_indices": luna_indices},
        "local_window_ambiguity": {
            "shared_standardized_pca_components": ambiguity_components,
            "original_sc": ambiguity_original,
            "luna_ar_hcf": ambiguity_luna,
        },
    }
    (out_dir / "data_dynamics_comparison_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(out_dir / "data_dynamics_comparison.md", metrics["original_sc"], metrics["luna_ar_hcf"], ambiguity_original, ambiguity_luna)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
