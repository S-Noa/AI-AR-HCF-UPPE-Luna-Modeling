#!/usr/bin/env python3
"""Select and export controlled Luna Simple/Moderate/Complex RNN datasets."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat


C_LIGHT = 299_792_458.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simple-dir", required=True)
    parser.add_argument("--complex-dir", required=True)
    parser.add_argument(
        "--moderate-dir",
        default=None,
        help=(
            "Optional Moderate candidate pool.  Moderate samples are selected "
            "from the center of their measured complexity-score distribution."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-per-class", type=int, default=1300)
    parser.add_argument("--train-evolutions", type=int, default=1250)
    parser.add_argument("--test-evolutions", type=int, default=50)
    parser.add_argument("--z-points", type=int, default=200)
    parser.add_argument("--lambda-points", type=int, default=251)
    parser.add_argument("--lambda-min-nm", type=float, default=200.0)
    parser.add_argument("--lambda-max-nm", type=float, default=2500.0)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def setup_logging(path):
    handlers = [logging.StreamHandler()]
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def read_field_map(path, lambda_nm, z_points):
    with h5py.File(path, "r") as handle:
        eomega = np.asarray(handle["Eω"][:])
        z = np.asarray(handle["z"][:], dtype=np.float64).reshape(-1)
        if eomega.shape[1] == z.size:
            eomega = eomega.T
        elif eomega.shape[0] != z.size:
            raise ValueError(f"Cannot align Eω={eomega.shape} with z={z.size}")
        omega = None
        if "grid" in handle:
            grid = handle["grid"]
            for key in ("ω", "omega", "w"):
                if key in grid:
                    candidate = np.asarray(grid[key][:], dtype=np.float64).reshape(-1)
                    if candidate.size == eomega.shape[1]:
                        omega = candidate
                        break
        if omega is None:
            raise ValueError("Missing usable frequency grid")
        params = {key: handle["benchmark_params"][key][()] for key in handle.get("benchmark_params", {})}

    # RealGrid includes a DC bin.  Only positive finite angular frequencies
    # have a physical wavelength and are valid for wavelength interpolation.
    valid_frequency = np.isfinite(omega) & (omega > 0.0)
    omega = omega[valid_frequency]
    eomega = eomega[:, valid_frequency]
    if omega.size < 2:
        raise ValueError("Frequency grid has fewer than two positive finite bins")
    source_lambda = C_LIGHT * 2.0 * np.pi / omega * 1e9
    order = np.argsort(source_lambda)
    source_lambda = source_lambda[order]
    power = np.abs(eomega[:, order]) ** 2
    selected_z = np.linspace(float(z[0]), float(z[-1]), z_points)
    z_power = np.empty((z_points, power.shape[1]), dtype=np.float64)
    for column in range(power.shape[1]):
        z_power[:, column] = np.interp(selected_z, z, power[:, column])
    output = np.empty((z_points, lambda_nm.size), dtype=np.float32)
    for row in range(z_points):
        output[row] = np.interp(lambda_nm, source_lambda, z_power[row], left=0.0, right=0.0)
    return np.maximum(output, 1e-30), selected_z, params


def complexity_metrics(power, lambda_nm):
    log_power = np.log10(np.maximum(power, 1e-30))
    lo, hi = np.min(log_power), np.max(log_power)
    normalized = (log_power - lo) / max(hi - lo, 1e-12)
    dz = float(np.mean(np.abs(np.diff(normalized, axis=0))))
    dlambda = float(np.mean(np.abs(np.diff(normalized, axis=1))))
    occupancy = float(np.mean(normalized > 0.1))
    uv_mask = (lambda_nm >= 200.0) & (lambda_nm <= 700.0)
    uv_occupancy = float(np.mean(normalized[:, uv_mask] > 0.1))
    score = dz + dlambda + 0.25 * occupancy + 0.50 * uv_occupancy
    return {"complexity_score": score, "mean_abs_dz": dz, "mean_abs_dlambda": dlambda,
            "occupancy": occupancy, "uv_occupancy": uv_occupancy}


def load_candidates(directory, label, lambda_nm, z_points):
    rows = []
    for path in sorted(Path(directory).glob("candidate_*.h5")):
        if not os.path.exists(f"{path}.done"):
            continue
        try:
            power, z, params = read_field_map(path, lambda_nm, z_points)
            row = {"class": label, "source_path": str(path), "power": power, "z": z}
            row.update(complexity_metrics(power, lambda_nm))
            for key, value in params.items():
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if np.asarray(value).shape == ():
                    value = np.asarray(value).item()
                row[key] = value
            rows.append(row)
        except Exception as exc:
            logging.warning("Skipping %s: %s", path, exc)
    logging.info("Loaded %d valid %s candidates", len(rows), label)
    return rows


def select_rows(rows, label, keep):
    if len(rows) < keep:
        raise ValueError(f"{label} has {len(rows)} valid candidates; need {keep}")
    ordered = sorted(rows, key=lambda row: row["complexity_score"])
    if label == "simple":
        return ordered[:keep]
    if label == "complex":
        # Preserve the historical export order: the hardest retained map is
        # first, matching the former ``reverse=True`` implementation.
        return list(reversed(ordered[-keep:]))
    if label == "moderate":
        # Keep the central score band rather than the easiest or hardest
        # candidates.  For 1600 candidates and keep=1300 this excludes the
        # lowest/highest 150 maps, preserving the physical bridge population.
        start = (len(ordered) - keep) // 2
        return ordered[start:start + keep]
    raise ValueError(f"Unsupported benchmark class: {label}")


def ordered_split(rows, train_count, test_count, seed):
    if train_count + test_count != len(rows):
        raise ValueError("keep-per-class must equal train-evolutions + test-evolutions")
    order = np.random.default_rng(seed).permutation(len(rows))
    shuffled = [rows[index] for index in order]
    return shuffled[:train_count] + shuffled[train_count:train_count + test_count]


def export_group(rows, label, output_dir, lambda_nm, train_count, test_count, seed):
    selected = select_rows(rows, label, train_count + test_count)
    ordered = ordered_split(selected, train_count, test_count, seed)
    raw = np.stack([row["power"].T for row in ordered], axis=0).astype(np.float32)
    global_max = np.max(raw)
    original_dbm = 10.0 * np.log10(np.maximum(raw / max(global_max, 1e-30), 1e-30))
    original_dbm = np.clip(original_dbm, -55.0, 0.0) / 55.0 + 1.0
    log_power = np.log10(np.maximum(raw, 1e-30))
    sample_lo = np.min(log_power, axis=(1, 2), keepdims=True)
    sample_hi = np.max(log_power, axis=(1, 2), keepdims=True)
    minmax = ((log_power - sample_lo) / np.maximum(sample_hi - sample_lo, 1e-12)).astype(np.float32)
    os.makedirs(output_dir, exist_ok=True)
    savemat(os.path.join(output_dir, f"{label}_raw_power.mat"), {"data": raw})
    savemat(os.path.join(output_dir, f"{label}_original_dbm.mat"), {"data": original_dbm.astype(np.float32)})
    savemat(os.path.join(output_dir, f"{label}_per_sample_minmax.mat"), {"data": minmax})
    return ordered


def write_manifest(groups, path, train_count):
    rows = [row for _, group in groups for row in group]
    keys = sorted({key for row in rows for key in row if key not in {"power", "z"}})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "split_index"] + keys)
        writer.writeheader()
        for _, group in groups:
            for index, row in enumerate(group):
                payload = {key: row.get(key, "") for key in keys}
                payload["split"] = "train" if index < train_count else "test"
                payload["split_index"] = index
                writer.writerow(payload)


def main():
    args = parse_args()
    setup_logging(args.log_file)
    if args.keep_per_class != args.train_evolutions + args.test_evolutions:
        raise ValueError("--keep-per-class must equal --train-evolutions + --test-evolutions")
    lambda_nm = np.linspace(args.lambda_min_nm, args.lambda_max_nm, args.lambda_points)
    simple = load_candidates(args.simple_dir, "simple", lambda_nm, args.z_points)
    complex_rows = load_candidates(args.complex_dir, "complex", lambda_nm, args.z_points)
    simple_ordered = export_group(simple, "simple", args.output_dir, lambda_nm,
                                  args.train_evolutions, args.test_evolutions, args.seed)
    complex_ordered = export_group(complex_rows, "complex", args.output_dir, lambda_nm,
                                   args.train_evolutions, args.test_evolutions, args.seed + 1)
    groups = [("simple", simple_ordered), ("complex", complex_ordered)]
    if args.moderate_dir:
        moderate = load_candidates(args.moderate_dir, "moderate", lambda_nm, args.z_points)
        moderate_ordered = export_group(moderate, "moderate", args.output_dir, lambda_nm,
                                        args.train_evolutions, args.test_evolutions, args.seed + 2)
        groups.append(("moderate", moderate_ordered))
    write_manifest(groups, os.path.join(args.output_dir, "manifest.csv"), args.train_evolutions)
    summary = {
        "z_points": args.z_points, "z_range_cm": [0.0, 5.0],
        "lambda_points": args.lambda_points, "lambda_range_nm": [args.lambda_min_nm, args.lambda_max_nm],
        "train_evolutions": args.train_evolutions, "test_evolutions": args.test_evolutions,
        "simple_score_range": [min(row["complexity_score"] for row in simple_ordered), max(row["complexity_score"] for row in simple_ordered)],
        "complex_score_range": [min(row["complexity_score"] for row in complex_ordered), max(row["complexity_score"] for row in complex_ordered)],
    }
    if args.moderate_dir:
        summary["moderate_selection"] = "central complexity-score band"
        summary["moderate_score_range"] = [
            min(row["complexity_score"] for row in moderate_ordered),
            max(row["complexity_score"] for row in moderate_ordered),
        ]
    with open(os.path.join(args.output_dir, "benchmark_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logging.info("Exported benchmark data to %s", args.output_dir)


if __name__ == "__main__":
    main()
