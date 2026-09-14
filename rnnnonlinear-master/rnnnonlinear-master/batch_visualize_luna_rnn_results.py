#!/usr/bin/env python3
"""Create uniform visualizations for completed Luna RNN training outputs."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = (
    "metrics.json",
    "stepwise_predictions.mat",
    "autoregressive_predictions.mat",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch visualize completed Luna RNN result directories."
    )
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_metrics(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    shape = payload.get("data_shape")
    test_evolutions = payload.get("test_evolutions")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("metrics.json has no valid data_shape=(N,n_lambda,n_steps)")
    if not isinstance(test_evolutions, int) or test_evolutions < 1:
        raise ValueError("metrics.json has no valid positive test_evolutions")
    if int(shape[1]) < 1 or int(shape[2]) < 2:
        raise ValueError("metrics.json has invalid wavelength or propagation dimensions")
    return payload, int(shape[1]), int(shape[2]), int(test_evolutions)


def discover(result_root, include_smoke):
    for result_dir in sorted(result_root.glob("results_*")):
        if not result_dir.is_dir():
            continue
        if result_dir.name.startswith("results_smoke_") and not include_smoke:
            yield result_dir, "skipped_smoke"
            continue
        missing = [name for name in REQUIRED_FILES if not (result_dir / name).is_file()]
        if missing:
            yield result_dir, "missing:" + ",".join(missing)
        else:
            yield result_dir, "eligible"


def write_manifest(output_root, rows):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    fields = [
        "experiment", "status", "reason", "n_lambda", "n_steps", "test_evolutions",
        "stepwise_r2", "autoregressive_r2", "autoregressive_final_r2", "output_dir",
    ]
    with (output_root / "batch_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    result_root = args.result_root.resolve()
    output_root = args.output_root.resolve()
    visualizer = Path(__file__).with_name("visualize_rnn_rollout_results.py")
    if not result_root.is_dir():
        raise FileNotFoundError(f"--result-root does not exist: {result_root}")
    if not visualizer.is_file():
        raise FileNotFoundError(f"Missing visualizer: {visualizer}")

    rows = []
    for result_dir, discovery_status in discover(result_root, args.include_smoke):
        row = {
            "experiment": result_dir.name,
            "status": discovery_status,
            "reason": "",
            "n_lambda": "",
            "n_steps": "",
            "test_evolutions": "",
            "stepwise_r2": "",
            "autoregressive_r2": "",
            "autoregressive_final_r2": "",
            "output_dir": str(output_root / result_dir.name),
        }
        if discovery_status != "eligible":
            row["reason"] = discovery_status
            rows.append(row)
            continue
        try:
            metrics, n_lambda, n_steps, test_evolutions = read_metrics(result_dir / "metrics.json")
            row.update({
                "n_lambda": n_lambda,
                "n_steps": n_steps,
                "test_evolutions": test_evolutions,
                "stepwise_r2": metrics.get("stepwise_r2", ""),
                "autoregressive_r2": metrics.get("autoregressive_r2", ""),
                "autoregressive_final_r2": metrics.get("autoregressive_final_r2", ""),
            })
        except Exception as exc:
            row["status"] = "skipped_invalid_metadata"
            row["reason"] = str(exc)
            rows.append(row)
            continue

        experiment_out = output_root / result_dir.name
        if (experiment_out / "COMPLETED").is_file() and not args.overwrite:
            row["status"] = "already_complete"
            rows.append(row)
            continue
        if args.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue

        command = [
            sys.executable, str(visualizer),
            "--stepwise-mat", str(result_dir / "stepwise_predictions.mat"),
            "--autoregressive-mat", str(result_dir / "autoregressive_predictions.mat"),
            "--output-dir", str(experiment_out),
            "--experiment-name", result_dir.name,
            "--test-evo", str(test_evolutions),
            "--steps", str(n_steps),
            "--wavelength-points", str(n_lambda),
            "--normalization-label", "saved RNN target space",
            "--sample-indices", *[str(index) for index in args.sample_indices],
        ]
        print("Running:", " ".join(command), flush=True)
        try:
            subprocess.run(command, check=True)
            experiment_out.mkdir(parents=True, exist_ok=True)
            (experiment_out / "COMPLETED").touch()
            row["status"] = "complete"
        except subprocess.CalledProcessError as exc:
            row["status"] = "failed"
            row["reason"] = f"visualizer exit code {exc.returncode}"
        rows.append(row)
        write_manifest(output_root, rows)

    write_manifest(output_root, rows)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"output_root": str(output_root), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
