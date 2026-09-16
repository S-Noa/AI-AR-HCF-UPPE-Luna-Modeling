#!/usr/bin/env python3
"""Compare raw4 RL surrogate candidates with completed Luna validations."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


C_LIGHT = 299_792_458.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_scalar(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    raise ValueError("No wavelength-compatible frequency grid in validation HDF5")


def final_spectrum_metrics(path):
    with h5py.File(path, "r") as handle:
        field = np.asarray(handle["Eω"][:])
        z = np.asarray(handle["z"][:]).reshape(-1)
        if field.ndim != 2:
            raise ValueError(f"Expected 2D Eω, got {field.shape}")
        final = field[-1, :] if field.shape[0] == z.size else field[:, -1]
        wavelength = wavelength_axis_nm(handle, final.size)
        power = np.abs(final) ** 2
        valid = np.isfinite(wavelength) & np.isfinite(power) & (wavelength > 0.0)
        wavelength, power = wavelength[valid], power[valid]
        order = np.argsort(wavelength)
        wavelength, power = wavelength[order], power[order]
        total = (wavelength >= 200.0) & (wavelength <= 2500.0)
        uv = (wavelength >= 200.0) & (wavelength <= 700.0)
        if total.sum() < 2 or uv.sum() < 2:
            raise ValueError("Insufficient samples in total or UV wavelength range")
        # ``trapz`` is available in the cloud's NumPy version, unlike the
        # newer ``trapezoid`` alias.
        total_integral = float(np.trapz(power[total], wavelength[total]))
        uv_integral = float(np.trapz(power[uv], wavelength[uv]))
        final_peak = float(wavelength[total][np.argmax(power[total])])
        uv_peak = float(wavelength[uv][np.argmax(power[uv])])
        input_energy = float(handle["input/energy"][()]) if "input/energy" in handle else np.nan
        final_energy = float(np.asarray(handle["stats/energy"][:]).reshape(-1)[-1]) if "stats/energy" in handle else np.nan
        transmission = final_energy / input_energy if input_energy > 0.0 else np.nan
        max_density = float(np.nanmax(handle["stats/electrondensity"][:])) if "stats/electrondensity" in handle else np.nan
        max_ion_rate = float(np.nanmax(handle["stats/peak_ionisation_rate"][:])) if "stats/peak_ionisation_rate" in handle else np.nan
        return {
            "luna_uv_fraction_200_700": uv_integral / total_integral if total_integral > 0.0 else np.nan,
            "luna_final_peak_nm": final_peak,
            "luna_uv_peak_nm": uv_peak,
            "luna_input_energy_j": input_energy,
            "luna_final_energy_j": final_energy,
            "luna_energy_transmission": transmission,
            "luna_uv_energy_proxy_j": final_energy * uv_integral / total_integral if total_integral > 0.0 else np.nan,
            "luna_max_electron_density_m3": max_density,
            "luna_max_ionisation_rate_s1": max_ion_rate,
            "wavelength_nm": wavelength,
            "power": power,
        }


def load_candidates(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {int(row["rank"]): row for row in csv.DictReader(handle)}


def main():
    args = parse_args()
    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(validation_dir / "luna_candidates.csv")
    failure_ranks = set()
    failure_file = validation_dir / "luna_validation_failures.csv"
    if failure_file.is_file():
        with failure_file.open(newline="", encoding="utf-8") as handle:
            failure_ranks = {int(row["rank"]) for row in csv.DictReader(handle)}

    rows, spectra = [], []
    for path in sorted((validation_dir / "validation_h5").glob("candidate_rank*_seed*.h5")):
        rank = int(path.name.split("_rank", 1)[1].split("_", 1)[0])
        metrics = final_spectrum_metrics(path)
        candidate = candidates.get(rank)
        if candidate is None:
            raise KeyError(f"No CSV candidate for {path.name}")
        row = {
            "rank": rank,
            "seed": int(candidate["seed"]),
            "surrogate_uv_fraction": float(candidate.get("surrogate_uv_fraction", candidate["uv_fraction_surrogate"])),
            "luna_uv_fraction_200_700": metrics["luna_uv_fraction_200_700"],
            "luna_final_peak_nm": metrics["luna_final_peak_nm"],
            "luna_uv_peak_nm": metrics["luna_uv_peak_nm"],
            "luna_max_electron_density_m3": metrics["luna_max_electron_density_m3"],
            "luna_max_ionisation_rate_s1": metrics["luna_max_ionisation_rate_s1"],
            "luna_input_energy_j": metrics["luna_input_energy_j"],
            "luna_final_energy_j": metrics["luna_final_energy_j"],
            "luna_energy_transmission": metrics["luna_energy_transmission"],
            "luna_uv_energy_proxy_j": metrics["luna_uv_energy_proxy_j"],
            "energy_uj": float(candidate["energy_uj"]),
            "tau_fs": float(candidate["tau_fs"]),
            "pressure_bar": float(candidate["pressure_bar"]),
            "diameter_um": float(candidate["diameter_um"]),
        }
        row["uv_fraction_error"] = row["luna_uv_fraction_200_700"] - row["surrogate_uv_fraction"]
        rows.append(row)
        spectra.append((row, metrics["wavelength_nm"], metrics["power"]))

    rows.sort(key=lambda row: row["rank"])
    fieldnames = list(rows[0]) if rows else ["rank"]
    with (output_dir / "luna_validation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_requested": len(candidates),
        "n_successful": len(rows),
        "n_failed": len(failure_ranks),
        "successful_fraction": len(rows) / len(candidates) if candidates else np.nan,
        "surrogate_uv_fraction_range": [min((row["surrogate_uv_fraction"] for row in rows), default=np.nan), max((row["surrogate_uv_fraction"] for row in rows), default=np.nan)],
        "luna_uv_fraction_range": [min((row["luna_uv_fraction_200_700"] for row in rows), default=np.nan), max((row["luna_uv_fraction_200_700"] for row in rows), default=np.nan)],
        "mean_uv_fraction_error": float(np.mean([row["uv_fraction_error"] for row in rows])) if rows else np.nan,
        "luna_energy_transmission_range": [min((row["luna_energy_transmission"] for row in rows), default=np.nan), max((row["luna_energy_transmission"] for row in rows), default=np.nan)],
        "luna_max_electron_density_range_m3": [min((row["luna_max_electron_density_m3"] for row in rows), default=np.nan), max((row["luna_max_electron_density_m3"] for row in rows), default=np.nan)],
    }
    with (output_dir / "luna_validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        rank = [row["rank"] for row in rows]
        axes[0].plot(rank, [row["surrogate_uv_fraction"] for row in rows], "o-", label="Surrogate")
        axes[0].plot(rank, [row["luna_uv_fraction_200_700"] for row in rows], "s-", label="Luna")
        axes[0].set(xlabel="Candidate rank", ylabel="UV fraction (200-700 nm)", ylim=(-0.02, 1.02))
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].scatter([row["surrogate_uv_fraction"] for row in rows], [row["luna_uv_fraction_200_700"] for row in rows])
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[1].set(xlabel="Surrogate UV fraction", ylabel="Luna UV fraction", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axes[1].grid(alpha=0.3)
        axes[2].scatter(rank, [row["luna_energy_transmission"] for row in rows], label="Transmission")
        axes[2].set(xlabel="Candidate rank", ylabel="Final / input energy")
        axes[2].grid(alpha=0.3)
        fig.savefig(output_dir / "surrogate_vs_luna_uv_fraction.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for row, wavelength, power in spectra:
            mask = (wavelength >= 200.0) & (wavelength <= 2500.0)
            relative_db = 10.0 * np.log10(np.maximum(power[mask] / np.max(power[mask]), 1e-12))
            ax.plot(wavelength[mask], relative_db, linewidth=1, label=f"rank {row['rank']}")
        ax.set(xlabel="Wavelength (nm)", ylabel="Relative spectral power (dB)", ylim=(-60, 2))
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=7)
        fig.savefig(output_dir / "successful_luna_final_spectra.png", dpi=220)
        plt.close(fig)

    report = [
        "# Raw4 RL Luna Validation Summary", "",
        f"- Requested candidates: {summary['n_requested']}",
        f"- Successful Luna propagations: {summary['n_successful']}",
        f"- Non-runnable candidates: {summary['n_failed']}",
        f"- Successful fraction: {summary['successful_fraction']:.3f}",
        f"- Surrogate UV-fraction range: {summary['surrogate_uv_fraction_range']}",
        f"- Luna UV-fraction range: {summary['luna_uv_fraction_range']}",
        f"- Mean Luna minus surrogate UV fraction: {summary['mean_uv_fraction_error']:.6g}",
        f"- Luna energy-transmission range: {summary['luna_energy_transmission_range']}",
        f"- Luna maximum electron-density range (m^-3): {summary['luna_max_electron_density_range_m3']}",
        "",
        "The surrogate objective is a screening score. Luna values are the physical validation values.",
        "Candidates that exceeded the PPT rate-table limit are counted as non-runnable, not as successful high-UV solutions.",
    ]
    (output_dir / "luna_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
