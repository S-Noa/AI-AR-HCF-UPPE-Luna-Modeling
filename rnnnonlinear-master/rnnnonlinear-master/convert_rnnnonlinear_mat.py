#!/usr/bin/env python3
"""Convert original RNNnonlinear MATLAB datasets to train_luna_rnn.py HDF5 input.

The original Salmela/RNNnonlinear datasets store:

    data: (num_evolutions, n_grid, n_steps)

This script keeps that axis order and writes a small HDF5 .mat file with /data,
which is the same layout consumed by train_luna_rnn.py. Optional normalization
matches the original load_data.py choices used by predRNN.py.
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


LOGGER = logging.getLogger("convert_rnnnonlinear_mat")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert original RNNnonlinear .mat data to HDF5")
    parser.add_argument("--input", required=True, help="Original RNNnonlinear .mat file")
    parser.add_argument("--output", required=True, help="Output HDF5 .mat path for train_luna_rnn.py")
    parser.add_argument("--normalization", choices=["none", "max", "dBm", "manual"], default="dBm",
                        help="Normalization matching load_data.py; default is dBm")
    parser.add_argument("--manual-max", type=float, default=10369993.175721595,
                        help="Manual max used when --normalization=manual")
    parser.add_argument("--db-limit", type=float, default=55.0,
                        help="dB dynamic range used when --normalization=dBm")
    parser.add_argument("--eps", type=float, default=1e-30,
                        help="Lower bound before log10 for dBm normalization")
    parser.add_argument("--max-evolutions", type=int, default=None,
                        help="Optional prefix limit for quick smoke tests")
    parser.add_argument("--compression", choices=["gzip", "none"], default="gzip")
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def load_data_array(path):
    scipy_error = None
    try:
        mat = sio.loadmat(path)
        if "data" not in mat:
            raise KeyError("MAT file does not contain variable 'data'")
        return np.asarray(mat["data"])
    except Exception as exc:
        scipy_error = exc

    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                raise KeyError("HDF5 file does not contain dataset /data")
            return np.asarray(f["data"])
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path} with scipy ({scipy_error}) or h5py ({exc})") from exc


def normalize_data(data, args):
    data = data.astype(np.float32, copy=False)
    if args.normalization == "none":
        return data, None
    if args.normalization == "manual":
        reference = float(args.manual_max)
        if reference <= 0:
            raise ValueError("--manual-max must be positive")
        return data / reference, reference

    reference = float(np.max(np.abs(data)))
    if reference <= 0:
        raise ValueError("Cannot normalize data with non-positive absolute maximum")
    data = data / reference
    if args.normalization == "max":
        return data, reference

    safe = np.maximum(data, float(args.eps))
    data_db = 10.0 * np.log10(safe)
    floor = -float(args.db_limit)
    data_db = np.clip(data_db, floor, 0.0)
    scaled = data_db / float(args.db_limit) + 1.0
    return scaled.astype(np.float32, copy=False), reference


def main():
    args = parse_args()
    setup_logging(args.log_file)
    start = time.perf_counter()
    LOGGER.info("Loading original RNNnonlinear data from %s", args.input)
    data = load_data_array(args.input)
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (N,n_grid,n_steps), got {data.shape}")
    LOGGER.info("Raw data: shape=%s dtype=%s size=%s", data.shape, data.dtype, format_bytes(data.nbytes))

    if args.max_evolutions is not None:
        LOGGER.info("Applying max_evolutions=%d", args.max_evolutions)
        data = data[:args.max_evolutions]

    data, reference = normalize_data(data, args)
    LOGGER.info(
        "Converted data: shape=%s dtype=%s min=%.6g max=%.6g normalization=%s reference=%s",
        data.shape,
        data.dtype,
        float(np.min(data)),
        float(np.max(data)),
        args.normalization,
        reference,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    compression_kwargs = {}
    if args.compression != "none":
        compression_kwargs["compression"] = args.compression
        if args.compression == "gzip":
            compression_kwargs["compression_opts"] = 4

    LOGGER.info("Writing HDF5 output to %s", args.output)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("data", data=data, **compression_kwargs)
        f.create_dataset("z_norm", data=np.linspace(0.0, 1.0, data.shape[2], dtype=np.float32))
        f.attrs["format"] = "hdf5_mat_v7_3_compatible"
        f.attrs["source"] = "original_rnnnonlinear_mat"
        f.attrs["normalization"] = args.normalization

    metadata = {
        "source_mat": args.input,
        "output_mat": args.output,
        "data_shape": list(data.shape),
        "normalization": args.normalization,
        "normalization_reference": reference,
        "db_limit": args.db_limit if args.normalization == "dBm" else None,
        "max_evolutions": args.max_evolutions,
        "elapsed_seconds": time.perf_counter() - start,
    }
    metadata_path = args.metadata_output
    if metadata_path is None:
        root, _ = os.path.splitext(args.output)
        metadata_path = root + "_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info("Saved metadata to %s", metadata_path)
    LOGGER.info("Output file size: %s", format_bytes(os.path.getsize(args.output)))


if __name__ == "__main__":
    main()
