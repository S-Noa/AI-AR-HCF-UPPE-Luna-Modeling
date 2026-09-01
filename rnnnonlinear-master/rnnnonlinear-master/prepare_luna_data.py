#!/usr/bin/env python3
"""Convert Luna processed temporal spectra to the RNNnonlinear .mat format.

    RNNnonlinear expects a MATLAB file with:

    data: (num_evolutions, n_grid, n_steps)

Optionally, this script can also export Luna input features for conditional
autoregressive RNN training:

    features: (num_evolutions, n_features)
    z_norm:   (n_steps,)

Luna preprocessing stores temporal labels as:

    y_temporal_<split>.npy: (num_samples, n_z, n_lambda)

This script concatenates selected splits and transposes the axes.
"""

import argparse
import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import scipy.io as sio


LOGGER = logging.getLogger("prepare_luna_data")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Luna temporal spectra for RNNnonlinear")
    parser.add_argument("--input-dir", required=True, help="Processed Luna data directory")
    parser.add_argument("--output", default="simulations/luna_temporal.mat", help="Output .mat path")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"], help="Splits to concatenate")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit after concatenation")
    parser.add_argument("--metadata-output", default=None, help="Optional metadata JSON path")
    parser.add_argument("--output-format", choices=["hdf5", "mat5"], default="hdf5",
                        help="Output format. hdf5 writes a MATLAB v7.3-compatible HDF5 file")
    parser.add_argument("--compression", choices=["gzip", "none"], default="gzip",
                        help="Compression for HDF5 output")
    parser.add_argument("--include-features", action="store_true",
                        help="Also export X_<split>.npy as /features for conditional RNN training")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO",
                        help="Logging verbosity")
    return parser.parse_args()


def setup_logging(log_file=None, log_level="INFO"):
    level = getattr(logging, log_level.upper())
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def log_array(name, arr):
    LOGGER.info(
        "%s: shape=%s dtype=%s size=%s",
        name,
        arr.shape,
        arr.dtype,
        format_bytes(arr.nbytes),
    )


def load_split(input_dir, split):
    path = os.path.join(input_dir, f"y_temporal_{split}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    start = time.perf_counter()
    LOGGER.info("Loading temporal split '%s' from %s", split, path)
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path} must have shape (N, n_z, n_lambda), got {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    LOGGER.info("Loaded temporal split '%s' in %.2fs", split, time.perf_counter() - start)
    log_array(f"y_temporal_{split}", arr)
    return arr


def load_feature_split(input_dir, split):
    path = os.path.join(input_dir, f"X_{split}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    start = time.perf_counter()
    LOGGER.info("Loading feature split '%s' from %s", split, path)
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} must have shape (N, n_features), got {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    LOGGER.info("Loaded feature split '%s' in %.2fs", split, time.perf_counter() - start)
    log_array(f"X_{split}", arr)
    return arr


def load_z_reference(input_dir, split, n_steps):
    """Load one z vector and normalize it to [0, 1] for conditional RNN input."""
    path = os.path.join(input_dir, f"z_{split}.npy")
    if not os.path.exists(path):
        LOGGER.warning("No %s found; using uniform normalized z grid", path)
        z = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        return z
    LOGGER.info("Loading z reference from %s", path)
    raw = np.load(path, allow_pickle=True)
    if len(raw) == 0:
        LOGGER.warning("%s is empty; using uniform normalized z grid", path)
        return np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    z = np.asarray(raw[0], dtype=np.float32).reshape(-1)
    if z.size != n_steps:
        raise ValueError(f"{path} first z vector has {z.size} steps, expected {n_steps}")
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max <= z_min:
        LOGGER.warning("%s has non-increasing z range; using uniform normalized z grid", path)
        return np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    z_norm = ((z - z_min) / (z_max - z_min)).astype(np.float32)
    LOGGER.info("Loaded z reference: raw range=[%.6g, %.6g], steps=%d", z_min, z_max, z_norm.size)
    return z_norm


def save_output(path, data, output_format, compression, features=None, z_norm=None, feature_names=None):
    """Save data as MATLAB v5 or HDF5-backed MATLAB v7.3-style .mat."""
    start = time.perf_counter()
    LOGGER.info("Writing output to %s using format=%s compression=%s", path, output_format, compression)
    if output_format == "mat5":
        payload = {"data": data}
        if features is not None:
            payload["features"] = features
        if z_norm is not None:
            payload["z_norm"] = z_norm
        sio.savemat(path, payload)
        LOGGER.info("Finished writing Mat5 output in %.2fs", time.perf_counter() - start)
        return "mat5"

    compression_kwargs = {}
    if compression != "none":
        compression_kwargs["compression"] = compression
        if compression == "gzip":
            compression_kwargs["compression_opts"] = 4

    with h5py.File(path, "w") as f:
        LOGGER.info("Creating HDF5 dataset /data")
        dset = f.create_dataset("data", data=data, **compression_kwargs)
        dset.attrs["description"] = "Luna temporal spectra, shape=(N,n_lambda,n_z)"
        if features is not None:
            LOGGER.info("Creating HDF5 dataset /features")
            f.create_dataset("features", data=features, **compression_kwargs)
            if feature_names:
                encoded = np.asarray([name.encode("utf-8") for name in feature_names])
                LOGGER.info("Creating HDF5 dataset /feature_names with %d names", len(feature_names))
                f.create_dataset("feature_names", data=encoded)
        if z_norm is not None:
            LOGGER.info("Creating HDF5 dataset /z_norm")
            f.create_dataset("z_norm", data=z_norm.astype(np.float32, copy=False))
        f.attrs["format"] = "hdf5_mat_v7_3_compatible"

    LOGGER.info("Finished writing HDF5 output in %.2fs", time.perf_counter() - start)
    return "hdf5_mat_v7_3_compatible"


def run(args):
    start_total = time.perf_counter()
    LOGGER.info("Starting Luna temporal data export")
    LOGGER.info("input_dir=%s", args.input_dir)
    LOGGER.info("output=%s", args.output)
    LOGGER.info("splits=%s", ",".join(args.splits))
    LOGGER.info("output_format=%s compression=%s", args.output_format, args.compression)
    LOGGER.info("include_features=%s max_samples=%s", args.include_features, args.max_samples)

    arrays = [load_split(args.input_dir, split) for split in args.splits]
    feature_arrays = [load_feature_split(args.input_dir, split) for split in args.splits] if args.include_features else None
    nonempty = [arr for arr in arrays if arr.shape[0] > 0]
    if not nonempty:
        raise ValueError("No samples found in selected splits")

    n_z_set = {arr.shape[1] for arr in nonempty}
    n_lambda_set = {arr.shape[2] for arr in nonempty}
    LOGGER.info("Split shape check: n_z=%s n_lambda=%s", sorted(n_z_set), sorted(n_lambda_set))
    if len(n_z_set) != 1 or len(n_lambda_set) != 1:
        raise ValueError(f"All splits must share shape; n_z={n_z_set}, n_lambda={n_lambda_set}")

    LOGGER.info("Concatenating %d non-empty temporal splits", len(nonempty))
    temporal = np.concatenate(nonempty, axis=0)
    log_array("temporal_concatenated", temporal)
    features = None
    if args.include_features:
        feature_nonempty = [arr for arr in feature_arrays if arr.shape[0] > 0]
        if len(feature_nonempty) != len(nonempty):
            raise ValueError("Temporal/features split emptiness mismatch")
        feature_dim_set = {arr.shape[1] for arr in feature_nonempty}
        LOGGER.info("Feature shape check: feature_dim=%s", sorted(feature_dim_set))
        if len(feature_dim_set) != 1:
            raise ValueError(f"All feature splits must share feature dimension, got {feature_dim_set}")
        for split, temporal_arr, feature_arr in zip(args.splits, arrays, feature_arrays):
            if temporal_arr.shape[0] != feature_arr.shape[0]:
                raise ValueError(
                    f"Split {split} sample mismatch: y_temporal has {temporal_arr.shape[0]}, "
                    f"X has {feature_arr.shape[0]}"
                )
        LOGGER.info("Concatenating %d non-empty feature splits", len(feature_nonempty))
        features = np.concatenate(feature_nonempty, axis=0)
        log_array("features_concatenated", features)
    if args.max_samples is not None:
        LOGGER.info("Applying max_samples=%d", args.max_samples)
        temporal = temporal[:args.max_samples]
        if features is not None:
            features = features[:args.max_samples]
        log_array("temporal_after_max_samples", temporal)
        if features is not None:
            log_array("features_after_max_samples", features)

    LOGGER.info("Transposing temporal data from (N,n_z,n_lambda) to (N,n_lambda,n_z)")
    data = np.transpose(temporal, (0, 2, 1))
    log_array("data", data)
    z_norm = load_z_reference(args.input_dir, args.splits[0], data.shape[2])
    log_array("z_norm", z_norm)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    feature_names = None

    processing_params_path = os.path.join(args.input_dir, "processing_params.json")
    processing_params = {}
    if os.path.exists(processing_params_path):
        LOGGER.info("Loading processing params from %s", processing_params_path)
        with open(processing_params_path, "r", encoding="utf-8") as f:
            processing_params = json.load(f)
        feature_names = processing_params.get("input_features")
        LOGGER.info("Loaded processing params; input_features=%s", len(feature_names) if feature_names else 0)
    else:
        LOGGER.warning("No processing_params.json found at %s", processing_params_path)

    output_format = save_output(
        args.output, data, args.output_format, args.compression,
        features=features, z_norm=z_norm, feature_names=feature_names
    )
    if os.path.exists(args.output):
        LOGGER.info("Output file size: %s", format_bytes(os.path.getsize(args.output)))

    elapsed_seconds = time.perf_counter() - start_total
    metadata = {
        "source_input_dir": args.input_dir,
        "splits": args.splits,
        "output_mat": args.output,
        "output_format": output_format,
        "compression": args.compression if args.output_format == "hdf5" else None,
        "data_shape": list(data.shape),
        "features_shape": list(features.shape) if features is not None else None,
        "feature_names": feature_names,
        "z_norm_shape": list(z_norm.shape),
        "conditioning_available": features is not None,
        "original_temporal_shape": list(temporal.shape),
        "wavelength_range_nm": processing_params.get("wavelength_range_nm"),
        "target_points": processing_params.get("target_points"),
        "sample_filter": processing_params.get("sample_filter"),
        "early_dense_detected": processing_params.get("early_dense_detected"),
        "n_z": processing_params.get("n_z"),
        "spectrum_normalization": processing_params.get("spectrum_normalization"),
        "output_normalization": processing_params.get("output_normalization"),
        "log_file": args.log_file,
        "elapsed_seconds": elapsed_seconds,
    }
    metadata_path = args.metadata_output
    if metadata_path is None:
        root, _ = os.path.splitext(args.output)
        metadata_path = root + "_metadata.json"
    metadata_dir = os.path.dirname(metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    LOGGER.info("Writing metadata JSON to %s", metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    LOGGER.info("Saved output: %s", args.output)
    LOGGER.info("Saved metadata: %s", metadata_path)
    LOGGER.info("data shape: %s", data.shape)
    LOGGER.info("output format: %s", output_format)
    LOGGER.info("Completed Luna temporal data export in %.2fs", elapsed_seconds)


def main():
    args = parse_args()
    setup_logging(args.log_file, args.log_level)
    try:
        run(args)
    except Exception:
        LOGGER.exception("prepare_luna_data failed")
        raise


if __name__ == "__main__":
    main()
