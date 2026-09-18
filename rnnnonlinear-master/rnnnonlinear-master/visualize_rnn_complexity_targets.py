#!/usr/bin/env python3
"""Plot representative physical spectral-evolution targets for the RNN benchmark."""

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
        z_cm = np.asarray(handle["z"][:]).reshape(-1) * 100.0
        omega = np.asarray(handle["grid/ω"][:]).reshape(-1)
    if field.shape[0] != z_cm.size:
        field = field.T
    wavelength = 2.0 * np.pi * C_LIGHT / omega * 1e9
    power = np.abs(field) ** 2
    order = np.argsort(wavelength)
    mapped = np.array([np.interp(target_lambda, wavelength[order], row[order], left=0.0, right=0.0)
                       for row in power])
    return z_cm, 10.0 * np.log10(np.maximum(mapped / max(float(mapped.max()), 1e-300), 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    target_lambda = np.linspace(200.0, 2500.0, 251)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for class_index, label in enumerate(("simple", "complex")):
        choices = sorted((row for row in rows if row["class"] == label), key=lambda row: float(row["complexity_score"]))
        for axis, index in zip(axes[class_index], np.linspace(0, len(choices) - 1, 3, dtype=int)):
            row = choices[index]
            z_cm, spectral_db = read_map(Path(row["source_path"]), target_lambda)
            image = axis.imshow(spectral_db, origin="lower", aspect="auto",
                                extent=[200, 2500, z_cm[0], z_cm[-1]], vmin=-50, vmax=0, cmap="turbo")
            axis.set_title(f"{label.capitalize()}, C={float(row['complexity_score']):.3f}\n"
                           f"E={float(row['energy_uj']):.2f} uJ, tau={float(row['tau_fs']):.1f} fs, "
                           f"p={float(row['pressure_bar']):.1f} bar, d={float(row['diameter_um']):.1f} um", fontsize=8)
            axis.set_xlabel("Wavelength (nm)")
            axis.set_ylabel("Distance (cm)")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, pad=0.01)
    colorbar.set_label("Relative spectral power (dB)")
    fig.suptitle("Controlled RNN benchmark: representative target spectral evolution", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
