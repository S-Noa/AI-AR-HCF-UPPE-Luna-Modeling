#!/usr/bin/env python3
"""Create a quantile-stratified gallery of Simple benchmark target maps."""

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


C_LIGHT = 299_792_458.0


def read_map(path, target_lambda):
    with h5py.File(path, "r") as handle:
        field = np.asarray(handle["Eω"][:])
        z_cm = np.asarray(handle["z"][:], dtype=float).reshape(-1) * 100.0
        omega = np.asarray(handle["grid/ω"][:], dtype=float).reshape(-1)
    if field.shape[0] != z_cm.size:
        field = field.T
    if field.shape[0] != z_cm.size:
        raise ValueError(f"Cannot align field and z axes for {path}")
    valid = np.isfinite(omega) & (omega > 0.0)
    wavelength = 2.0 * np.pi * C_LIGHT / omega[valid] * 1e9
    power = np.abs(field[:, valid]) ** 2
    order = np.argsort(wavelength)
    wavelength, power = wavelength[order], power[:, order]
    mapped = np.array([
        np.interp(target_lambda, wavelength, row, left=0.0, right=0.0) for row in power
    ], dtype=np.float32)
    return z_cm, 10.0 * np.log10(np.maximum(mapped / max(float(mapped.max()), 1e-300), 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--db-floor", type=float, default=-50.0)
    args = parser.parse_args()
    if args.examples < 2:
        raise ValueError("--examples must be at least 2")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        simple_rows = [row for row in csv.DictReader(handle) if row["class"] == "simple"]
    simple_rows.sort(key=lambda row: float(row["complexity_score"]))
    if len(simple_rows) < args.examples:
        raise ValueError("Not enough Simple samples in manifest")

    positions = np.linspace(0, len(simple_rows) - 1, args.examples, dtype=int)
    selected = [simple_rows[index] for index in positions]
    quantiles = positions / (len(simple_rows) - 1) * 100.0
    target_lambda = np.linspace(200.0, 2500.0, 251)
    columns = 4
    rows = int(np.ceil(args.examples / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.4 * rows),
                                sharex=True, sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    image = None
    for axis, row, quantile in zip(axes, selected, quantiles):
        z_cm, spectral_db = read_map(Path(row["source_path"]), target_lambda)
        image = axis.imshow(spectral_db, origin="lower", aspect="auto",
                            extent=[200.0, 2500.0, z_cm[0], z_cm[-1]],
                            vmin=args.db_floor, vmax=0.0, cmap="turbo")
        axis.set_title(
            f"q={quantile:.0f}%, C={float(row['complexity_score']):.3f}\n"
            f"E={float(row['energy_uj']):.2f} uJ, tau={float(row['tau_fs']):.1f} fs\n"
            f"p={float(row['pressure_bar']):.1f} bar, d={float(row['diameter_um']):.1f} um",
            fontsize=8,
        )
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Distance (cm)")
    for axis in axes[len(selected):]:
        axis.set_visible(False)
    colorbar = figure.colorbar(image, ax=axes[:len(selected)], shrink=0.9, pad=0.01)
    colorbar.set_label("Relative spectral power (dB)")
    figure.suptitle("Simple benchmark targets across complexity quantiles", fontsize=14)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "simple_target_quantile_gallery.png", dpi=220)
    plt.close(figure)

    scores = np.array([float(row["complexity_score"]) for row in simple_rows])
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.hist(scores, bins=40, color="#2c7fb8", alpha=0.85)
    axis.scatter([float(row["complexity_score"]) for row in selected],
                 np.zeros(len(selected)), color="#d95f0e", label="gallery selections", zorder=3)
    axis.set(xlabel="Benchmark complexity score", ylabel="Simple-sample count",
             title="Simple benchmark complexity distribution")
    axis.legend()
    figure.savefig(args.output_dir / "simple_complexity_distribution.png", dpi=220)

    with (args.output_dir / "simple_target_gallery_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["quantile_percent", *selected[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for quantile, row in zip(quantiles, selected):
            writer.writerow({"quantile_percent": f"{quantile:.3f}", **row})


if __name__ == "__main__":
    main()
