#!/usr/bin/env python3
"""Create a raw4 final-spectrum dataset at a fixed propagation distance.

The source global-log dataset stores standardized temporal maps. This tool
extracts one physical z slice, re-fits global log-power statistics on the
training split at that distance, and writes a small independent final-spectrum
task for early-propagation inverse design.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
from numpy.lib.format import open_memmap


RAW4_NAMES = ["energy", "tau", "pressure", "diameter"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--z-cm", type=float, default=5.0)
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def setup_logging(path):
    handlers = [logging.StreamHandler()]
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def z_index_for_split(input_dir, split, target_m):
    z_values = np.load(os.path.join(input_dir, f"z_{split}.npy"), allow_pickle=True)
    first = np.asarray(z_values[0], dtype=np.float64)
    index = int(np.argmin(np.abs(first - target_m)))
    if not np.allclose(first[index], target_m, rtol=0.0, atol=1e-12):
        logging.warning("Requested z=%.6g m; using nearest saved z=%.6g m", target_m, first[index])
    for value in z_values[1:min(len(z_values), 32)]:
        candidate = np.asarray(value, dtype=np.float64)
        if candidate.shape != first.shape or not np.allclose(candidate[index], first[index], rtol=0.0, atol=1e-12):
            raise ValueError(f"Inconsistent z grids found in {split}")
    return index, float(first[index])


def main():
    args = parse_args()
    setup_logging(args.log_file)
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (input_dir / "processing_params.json").open(encoding="utf-8") as handle:
        source_params = json.load(handle)
    if source_params.get("input_features") != RAW4_NAMES:
        raise ValueError("Input must be the raw4 view with energy, tau, pressure, diameter")
    source_norm = source_params.get("output_normalization", {})
    source_mean, source_std = float(source_norm["log_mean"]), float(source_norm["log_std"])
    target_m = args.z_cm / 100.0
    indices = {}
    for split in ("train", "val", "test"):
        index, actual_m = z_index_for_split(input_dir, split, target_m)
        indices[split] = index
        logging.info("%s: z index=%d, z=%.6g cm", split, index, actual_m * 100.0)
    if len(set(indices.values())) != 1:
        raise ValueError(f"Split z indices differ: {indices}")
    z_index = indices["train"]

    # Fit target-specific global log statistics from train only.
    total_count = total_sum = total_sumsq = 0.0
    y_train = np.load(input_dir / "y_temporal_train.npy", mmap_mode="r")
    for start in range(0, y_train.shape[0], args.batch_rows):
        standardized = np.asarray(y_train[start:start + args.batch_rows, z_index, :], dtype=np.float64)
        log_power = standardized * source_std + source_mean
        total_count += log_power.size
        total_sum += float(log_power.sum())
        total_sumsq += float(np.square(log_power).sum())
    target_mean = total_sum / total_count
    variance = max(total_sumsq / total_count - target_mean * target_mean, 1e-12)
    target_std = float(np.sqrt(variance))
    logging.info("z=%.3f cm target normalization: log_mean=%.8g log_std=%.8g", args.z_cm, target_mean, target_std)

    for split in ("train", "val", "test"):
        x = np.load(input_dir / f"X_{split}.npy")
        np.save(output_dir / f"X_{split}.npy", x.astype(np.float32, copy=False))
        temporal = np.load(input_dir / f"y_temporal_{split}.npy", mmap_mode="r")
        result = open_memmap(output_dir / f"y_{split}.npy", mode="w+", dtype=np.float32,
                             shape=(temporal.shape[0], temporal.shape[2]))
        for start in range(0, temporal.shape[0], args.batch_rows):
            standardized = np.asarray(temporal[start:start + args.batch_rows, z_index, :], dtype=np.float64)
            log_power = standardized * source_std + source_mean
            result[start:start + standardized.shape[0]] = ((log_power - target_mean) / target_std).astype(np.float32)
        result.flush()
        logging.info("Wrote %s: X=%s y=%s", split, x.shape, result.shape)

    shutil.copy2(input_dir / "scaler_X.joblib", output_dir / "scaler_X.joblib")
    params = dict(source_params)
    params["data_format"] = {
        "X": "(N, 4) - standardized raw physical parameters",
        "y": "(N, n_lambda) - final spectrum at the fixed propagation slice",
    }
    params["output_normalization"] = {
        "mode": "global_log_standard",
        "log_mean": target_mean,
        "log_std": target_std,
    }
    params["z_slice_target_cm"] = float(args.z_cm)
    params["z_slice_actual_cm"] = float(target_m * 100.0)
    params["z_slice_index"] = int(z_index)
    params["z_slice_source_dataset"] = str(input_dir.resolve())
    params["architecture"] = f"Raw4 fixed-z={args.z_cm:g} cm inverse-design view"
    params["split_sizes"] = {split: int(np.load(output_dir / f"X_{split}.npy").shape[0]) for split in ("train", "val", "test")}
    with (output_dir / "processing_params.json").open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    summary = {
        "z_target_cm": args.z_cm,
        "z_actual_cm": target_m * 100.0,
        "z_index": z_index,
        "output_log_mean": target_mean,
        "output_log_std": target_std,
        "split_sizes": params["split_sizes"],
    }
    with (output_dir / "zslice_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logging.info("Completed fixed-z raw4 view: %s", output_dir)


if __name__ == "__main__":
    main()
