#!/usr/bin/env python3
"""Create a comparable spectral-evolution montage for successful RL/Luna candidates."""

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


C_LIGHT = 299_792_458.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda-points", type=int, default=500)
    parser.add_argument("--db-floor", type=float, default=-50.0)
    return parser.parse_args()


def wavelength_axis_nm(handle, count):
    grid = handle["grid"]
    for key in ("lambda", "λ"):
        if key in grid:
            axis = np.asarray(grid[key][:], dtype=float).reshape(-1)
            if axis.size == count:
                return axis * 1e9
    for key in ("omega", "ω", "w"):
        if key in grid:
            omega = np.asarray(grid[key][:], dtype=float).reshape(-1)
            if omega.size == count:
                with np.errstate(divide="ignore", invalid="ignore"):
                    return 2.0 * np.pi * C_LIGHT / omega * 1e9
    raise ValueError("No usable wavelength axis")


def read_map(path, target_lambda):
    with h5py.File(path, "r") as handle:
        field = np.asarray(handle["Eω"][:])
        z_cm = np.asarray(handle["z"][:], dtype=float).reshape(-1) * 100.0
        if field.shape[0] != z_cm.size:
            field = field.T
        if field.shape[0] != z_cm.size:
            raise ValueError(f"Cannot align Eω={field.shape} with z={z_cm.size}")
        wavelength = wavelength_axis_nm(handle, field.shape[1])
    power = np.abs(field) ** 2
    valid = np.isfinite(wavelength) & (wavelength > 0.0)
    wavelength, power = wavelength[valid], power[:, valid]
    order = np.argsort(wavelength)
    wavelength, power = wavelength[order], power[:, order]
    mapped = np.empty((z_cm.size, target_lambda.size), dtype=np.float32)
    for index, row in enumerate(power):
        mapped[index] = np.interp(target_lambda, wavelength, row, left=0.0, right=0.0)
    relative_db = 10.0 * np.log10(np.maximum(mapped / max(float(mapped.max()), 1e-300), 1e-12))
    return z_cm, relative_db


def main():
    args = parse_args()
    validation_dir = Path(args.validation_dir)
    analysis_dir = Path(args.analysis_dir)
    with (analysis_dir / "luna_validation_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    target_lambda = np.linspace(200.0, 2500.0, args.lambda_points)
    columns = 3
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.4 * rows), sharex=True, sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    image = None
    for axis, metric in zip(axes, metrics):
        rank = int(metric["rank"])
        seed = int(metric["seed"])
        h5_path = validation_dir / "validation_h5" / f"candidate_rank{rank:03d}_seed{seed}.h5"
        z_cm, spectral_db = read_map(h5_path, target_lambda)
        image = axis.imshow(
            spectral_db,
            origin="lower",
            aspect="auto",
            extent=[target_lambda[0], target_lambda[-1], z_cm[0], z_cm[-1]],
            vmin=args.db_floor,
            vmax=0.0,
            cmap="turbo",
        )
        axis.axvspan(200.0, 700.0, color="white", alpha=0.08)
        axis.set_title(
            "rank {rank}: E={energy:.2f} uJ, tau={tau:.1f} fs\n"
            "p={pressure:.1f} bar, d={diameter:.1f} um; T={transmission:.1e}".format(
                rank=rank,
                energy=float(metric["energy_uj"]), tau=float(metric["tau_fs"]),
                pressure=float(metric["pressure_bar"]), diameter=float(metric["diameter_um"]),
                transmission=float(metric["luna_energy_transmission"]),
            ),
            fontsize=8,
        )
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Distance (cm)")
    for axis in axes[len(metrics):]:
        axis.set_visible(False)
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes[:len(metrics)], shrink=0.92, pad=0.01)
        colorbar.set_label("Relative spectral power (dB)")
    figure.suptitle("Successful raw4 RL candidates: Luna spectral evolution (per-sample relative scale)", fontsize=13)
    figure.savefig(args.output, dpi=220)
    print(args.output)


if __name__ == "__main__":
    main()
