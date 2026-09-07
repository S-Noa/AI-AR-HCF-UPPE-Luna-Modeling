#!/usr/bin/env python3
"""Export Luna HDF5 spectra as raw power maps for original RNNnonlinear code.

The Salmela RNN baseline applies its own global-max/dB normalization inside
``load_data.py``.  This helper keeps the Luna side as close as possible to that
interface by exporting positive linear power:

    data: (num_evolutions, n_lambda, n_z)

Only light resampling is performed: wavelength interpolation to a fixed
200--2500 nm grid, optional uniform wavelength downsampling, and optional
z-prefix selection/downsampling.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


C = 299792458.0
LOGGER = logging.getLogger("export_luna_raw_power_mat")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Luna HDF5 temporal spectra as raw linear-power .mat data"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing Luna .h5 files")
    parser.add_argument("--output", required=True, help="Output MATLAB v5 .mat file")
    parser.add_argument("--sample-filter", choices=["normal", "earlydense", "all"], default="earlydense")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--wavelength-min-nm", type=float, default=200.0)
    parser.add_argument("--wavelength-max-nm", type=float, default=2500.0)
    parser.add_argument("--target-points", type=int, default=1000,
                        help="Interpolation grid before optional lambda downsampling")
    parser.add_argument("--lambda-target-points", type=int, default=None,
                        help="Optional uniform wavelength downsampling after interpolation")
    parser.add_argument("--z-max-fraction", type=float, default=None,
                        help="Keep z_norm <= this value, e.g. 0.2 for first 10 cm of 50 cm")
    parser.add_argument("--z-target-points", type=int, default=None,
                        help="Optional uniform z downsampling after z-prefix selection")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args()


def setup_logging(log_file, log_level):
    level = getattr(logging, log_level)
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.setLevel(level)
    LOGGER.addHandler(stream)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        LOGGER.addHandler(fh)


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def keep_file(path, sample_filter):
    is_early = path.name.endswith("_earlydense.h5")
    if sample_filter == "earlydense":
        return is_early
    if sample_filter == "normal":
        return not is_early
    return True


def select_uniform_indices(n_points, target_points, name):
    if target_points is None:
        return np.arange(n_points, dtype=np.int64)
    if target_points < 2:
        raise ValueError(f"--{name}-target-points must be at least 2")
    if target_points > n_points:
        raise ValueError(f"--{name}-target-points={target_points} exceeds {n_points}")
    return np.rint(np.linspace(0, n_points - 1, target_points)).astype(np.int64)


def select_z_indices(z, z_max_fraction, z_target_points):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max <= z_min:
        z_norm = np.linspace(0.0, 1.0, z.size)
    else:
        z_norm = (z - z_min) / (z_max - z_min)
    if z_max_fraction is None:
        candidate = np.arange(z.size, dtype=np.int64)
    else:
        if not (0.0 < z_max_fraction <= 1.0):
            raise ValueError("--z-max-fraction must be in (0, 1]")
        candidate = np.flatnonzero(z_norm <= z_max_fraction + 1e-7).astype(np.int64)
        if candidate.size < 2:
            raise ValueError("--z-max-fraction leaves fewer than 2 z points")
    if z_target_points is None:
        return candidate, z_norm[candidate]
    if z_target_points > candidate.size:
        raise ValueError(f"--z-target-points={z_target_points} exceeds {candidate.size}")
    selected = candidate[np.rint(np.linspace(0, candidate.size - 1, z_target_points)).astype(np.int64)]
    return selected, z_norm[selected]


def get_dataset(group, candidates):
    for name in candidates:
        if name in group:
            return group[name][:]
    raise KeyError(f"None of {candidates} found")


def read_field_and_grid(path):
    with h5py.File(path, "r") as f:
        field = get_dataset(f, ["Eω", "Ew", "E"])
        z = get_dataset(f, ["z"])
        omega = None
        if "grid" in f:
            grid = f["grid"]
            for candidate in ["ω", "omega", "w"]:
                if candidate in grid and grid[candidate].shape != ():
                    maybe = grid[candidate][:]
                    if maybe.size in field.shape:
                        omega = maybe
                        break
        if omega is None:
            raise KeyError(f"No compatible omega grid found in {path}")
    return field, z, omega


def field_as_z_lambda_power(field, z, omega, target_wavelength_m, z_indices):
    field = np.asarray(field)
    if field.shape[0] == z.size:
        z_freq = field
    elif field.shape[1] == z.size:
        z_freq = field.T
    else:
        raise ValueError(f"Cannot match field shape {field.shape} to z_len={z.size}")

    if omega.size != z_freq.shape[1]:
        raise ValueError(f"omega_len={omega.size} does not match frequency axis {z_freq.shape[1]}")
    valid = np.abs(omega) > 1e10
    wavelength = np.full(omega.shape, np.nan, dtype=np.float64)
    wavelength[valid] = 2.0 * np.pi * C / omega[valid]
    mask = np.isfinite(wavelength)
    mask &= (wavelength >= target_wavelength_m[0]) & (wavelength <= target_wavelength_m[-1])
    if np.count_nonzero(mask) < 10:
        raise ValueError("Fewer than 10 wavelength points in selected range")
    order = np.argsort(wavelength[mask])
    wl = wavelength[mask][order]

    selected = []
    for zi in z_indices:
        power = np.abs(z_freq[zi, mask]) ** 2
        power = power[order]
        interp = np.interp(target_wavelength_m, wl, power, left=0.0, right=0.0)
        selected.append(interp.astype(np.float32))
    return np.stack(selected, axis=1)  # (n_lambda, n_z)


def main():
    args = parse_args()
    setup_logging(args.log_file, args.log_level)
    start = time.time()
    input_dir = Path(args.input_dir)
    files = sorted([p for p in input_dir.glob("*.h5") if keep_file(p, args.sample_filter)],
                   key=natural_key)
    if args.max_samples is not None:
        files = files[:args.max_samples]
    if not files:
        raise FileNotFoundError(f"No .h5 files matched in {input_dir}")

    target_wavelength = np.linspace(
        args.wavelength_min_nm * 1e-9,
        args.wavelength_max_nm * 1e-9,
        args.target_points,
        dtype=np.float64,
    )
    lambda_indices = select_uniform_indices(args.target_points, args.lambda_target_points, "lambda")
    target_selected = target_wavelength[lambda_indices]

    LOGGER.info("input_dir=%s", input_dir)
    LOGGER.info("output=%s", args.output)
    LOGGER.info("matched_files=%d sample_filter=%s", len(files), args.sample_filter)
    LOGGER.info("target wavelength grid: %d -> %d points, %.1f--%.1f nm",
                args.target_points, target_selected.size,
                target_selected[0] * 1e9, target_selected[-1] * 1e9)

    first_field, first_z, first_omega = read_field_and_grid(files[0])
    z_indices, z_norm = select_z_indices(first_z, args.z_max_fraction, args.z_target_points)
    LOGGER.info("z selection: original=%d selected=%d z_norm=[%.6g, %.6g]",
                first_z.size, z_indices.size, z_norm[0], z_norm[-1])

    data = np.empty((len(files), target_selected.size, z_indices.size), dtype=np.float32)
    source_files = []
    for i, path in enumerate(files):
        field, z, omega = (first_field, first_z, first_omega) if i == 0 else read_field_and_grid(path)
        if z.size != first_z.size:
            raise ValueError(f"{path} z_len={z.size}, expected {first_z.size}")
        full_grid_map = field_as_z_lambda_power(field, z, omega, target_wavelength, z_indices)
        data[i] = full_grid_map[lambda_indices, :]
        source_files.append(path.name)
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            LOGGER.info("processed %d/%d files", i + 1, len(files))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("saving mat5 data shape=%s dtype=%s size=%.2f MB",
                data.shape, data.dtype, data.nbytes / 1024 / 1024)
    sio.savemat(args.output, {
        "data": data,
        "z_norm": z_norm.astype(np.float32),
        "wavelength_nm": (target_selected * 1e9).astype(np.float32),
    })

    meta = {
        "input_dir": str(input_dir),
        "output": args.output,
        "sample_filter": args.sample_filter,
        "num_samples": len(files),
        "data_shape": list(data.shape),
        "z_selected_points": int(z_indices.size),
        "lambda_selected_points": int(target_selected.size),
        "normalization": "raw_linear_power",
        "source_files_head": source_files[:10],
        "elapsed_seconds": time.time() - start,
    }
    meta_path = os.path.splitext(args.output)[0] + "_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOGGER.info("saved metadata=%s", meta_path)
    LOGGER.info("done in %.2fs", time.time() - start)


if __name__ == "__main__":
    main()
