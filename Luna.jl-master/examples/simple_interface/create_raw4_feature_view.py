#!/usr/bin/env python3
"""Create a t0p6 raw-parameter-only view of an existing processed dataset."""

import argparse
import json
import os
import shutil

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


RAW4_NAMES = ["energy", "tau", "pressure", "diameter"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wallthickness-um", type=float, default=0.65)
    parser.add_argument("--tolerance-um", type=float, default=0.02)
    return parser.parse_args()


def copy_aligned_files(input_dir, output_dir, split, mask):
    prefixes = [
        "y_", "y_temporal_", "z_", "spectrum_scale_features_", "source_files_",
        "source_paths_", "source_indices_",
    ]
    for prefix in prefixes:
        source = os.path.join(input_dir, f"{prefix}{split}.npy")
        if not os.path.exists(source):
            continue
        values = np.load(source, allow_pickle=True)
        np.save(os.path.join(output_dir, os.path.basename(source)), values[mask], allow_pickle=True)
    for suffix in ("spectrum_quality", "suspicious_samples"):
        source = os.path.join(input_dir, f"{suffix}_{split}.csv")
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(output_dir, os.path.basename(source)))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.input_dir, "processing_params.json"), encoding="utf-8") as handle:
        params = json.load(handle)
    feature_names = params.get("input_features", [])
    required = ["energy", "tau", "pressure", "diameter", "wallthickness"]
    missing = [name for name in required if name not in feature_names]
    if missing:
        raise ValueError(f"Input dataset lacks required raw features: {missing}")
    raw_indices = [feature_names.index(name) for name in RAW4_NAMES]
    thickness_index = feature_names.index("wallthickness")
    scaler = joblib.load(os.path.join(args.input_dir, "scaler_X.joblib"))

    selected = {}
    for split in ("train", "val", "test"):
        x_scaled = np.load(os.path.join(args.input_dir, f"X_{split}.npy"))
        x_raw = scaler.inverse_transform(x_scaled)
        mask = np.abs(x_raw[:, thickness_index] - args.wallthickness_um) <= args.tolerance_um
        if not np.any(mask):
            raise ValueError(f"No {split} rows match wall thickness {args.wallthickness_um} um")
        selected[split] = (x_raw[mask][:, raw_indices], mask)

    raw4_scaler = StandardScaler().fit(selected["train"][0])
    for split, (raw4, mask) in selected.items():
        np.save(os.path.join(args.output_dir, f"X_{split}.npy"), raw4_scaler.transform(raw4).astype(np.float32))
        copy_aligned_files(args.input_dir, args.output_dir, split, mask)
    joblib.dump(raw4_scaler, os.path.join(args.output_dir, "scaler_X.joblib"))

    params["input_features"] = RAW4_NAMES
    params["feature_dim"] = 4
    params["data_format"]["X"] = "(N, 4) - standardized raw physical parameters"
    params["wallthickness_range"] = [args.wallthickness_um, args.wallthickness_um]
    params["architecture"] = "Raw4 t0p6 inverse-design view"
    params["raw4_source_dataset"] = os.path.abspath(args.input_dir)
    params["raw4_wallthickness_um"] = args.wallthickness_um
    params["raw4_split_sizes"] = {split: int(values[0].shape[0]) for split, values in selected.items()}
    with open(os.path.join(args.output_dir, "processing_params.json"), "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    print(json.dumps(params["raw4_split_sizes"], indent=2))


if __name__ == "__main__":
    main()
