#!/usr/bin/env python3
"""Run the unmodified RNNnonlinear Keras components on a benchmark .mat file."""

import argparse
import copy
import json
import os

import numpy as np
import scipy.io as sio
from sklearn.metrics import r2_score

from load_data import load_data
from make_RNN_model import make_RNN_model, update_RNN_model
from pred_evo import pred_evo


def fixed_horizon_metrics(model, truth, window_size, horizons):
    """Exact recursive k-step evaluation from true local histories for Keras."""
    output = {}
    n_evo, n_steps, n_grid = truth.shape
    for origin in ("all", "zero"):
        output[origin] = {}
        for horizon in horizons:
            starts = [0] if origin == "zero" else range(n_steps - horizon)
            predictions, targets = [], []
            for start in starts:
                prefix = np.repeat(truth[:, :1, :], window_size - 1, axis=1)
                padded = np.concatenate([prefix, truth], axis=1)
                current = padded[:, start:start + window_size, :]
                for _ in range(horizon):
                    prediction = model.predict_proba(current)
                    current = np.concatenate([current[:, 1:, :], prediction[:, None, :]], axis=1)
                predictions.append(prediction)
                targets.append(truth[:, start + horizon, :])
            prediction = np.concatenate(predictions, axis=0)
            target = np.concatenate(targets, axis=0)
            output[origin][str(horizon)] = {
                "horizon": horizon,
                "origin": origin,
                "n_start_positions": len(starts),
                "mse": float(np.mean((prediction - target) ** 2)),
                "r2": float(r2_score(target.reshape(-1), prediction.reshape(-1))),
            }
    return output


def evaluate_autoregressive_r2(model, x_data, y_data, evolutions, steps, window_size, n_grid):
    """Evaluate full rollout R2 on complete held-out propagation trajectories."""
    prediction = pred_evo(model, x_data, evolutions, steps, window_size, n_grid)
    return float(r2_score(y_data.reshape(-1), prediction.reshape(-1)))


def split_validation_evolutions(x_train, y_train, validation_evolutions, steps):
    """Hold out complete trajectories, never individual z windows, for selection."""
    if validation_evolutions <= 0:
        return x_train, y_train, None, None
    windows_per_evolution = steps - 1
    validation_windows = validation_evolutions * windows_per_evolution
    if validation_windows >= x_train.shape[0]:
        raise ValueError("--validation-evolutions leaves no training trajectories")
    return (
        x_train[:-validation_windows], y_train[:-validation_windows],
        x_train[-validation_windows:], y_train[-validation_windows:],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalization", choices=["none", "dBm"], required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-evolutions", type=int, default=1250)
    parser.add_argument("--test-evolutions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--epochs-stage1", type=int, default=50)
    parser.add_argument("--epochs-stage2", type=int, default=30)
    parser.add_argument(
        "--validation-evolutions", type=int, default=0,
        help=(
            "Number of complete trajectories held out from the training block for "
            "autoregressive checkpoint selection. Zero preserves the legacy final-epoch behavior."
        ),
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="Evaluate validation autoregressive R2 every N epochs; also evaluates epoch 1 and the final epoch.",
    )
    parser.add_argument(
        "--eval-fixed-horizons", type=int, nargs="+", default=list(range(1, 11)),
        help="Exact recursive horizons to evaluate from true histories (default: 1 through 10).",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    i_x, x_train, x_test, y_train, y_test = load_data(
        args.data, args.train_evolutions, args.test_evolutions, args.steps,
        args.window_size, args.normalization
    )
    if args.validation_evolutions < 0:
        raise ValueError("--validation-evolutions must be non-negative")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least one")
    x_fit, y_fit, x_validation, y_validation = split_validation_evolutions(
        x_train, y_train, args.validation_evolutions, args.steps
    )
    fit_validation_data = (x_validation, y_validation) if x_validation is not None else None

    model = make_RNN_model(args.window_size, i_x)
    total_epochs = args.epochs_stage1 + args.epochs_stage2
    best_validation_r2 = -float("inf")
    best_epoch = 0
    best_weights = None
    selection_history = []

    for epoch in range(1, total_epochs + 1):
        if epoch == args.epochs_stage1 + 1:
            model = update_RNN_model(model)
        history = model.fit(
            x_fit, y_fit, epochs=1, validation_data=fit_validation_data,
            verbose=2, shuffle=True,
        )
        should_select = (
            x_validation is not None
            and (epoch == 1 or epoch % args.checkpoint_every == 0 or epoch == total_epochs)
        )
        validation_r2 = None
        if should_select:
            validation_r2 = evaluate_autoregressive_r2(
                model, x_validation, y_validation, args.validation_evolutions,
                args.steps, args.window_size, i_x,
            )
            if np.isfinite(validation_r2) and validation_r2 > best_validation_r2:
                best_validation_r2 = validation_r2
                best_epoch = epoch
                best_weights = copy.deepcopy(model.get_weights())
                model.save_weights(
                    os.path.join(args.output_dir, "best_validation_autoregressive_weights.h5")
                )
        selection_history.append({
            "epoch": epoch,
            "stage": 1 if epoch <= args.epochs_stage1 else 2,
            "loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
            "validation_autoregressive_r2": validation_r2,
        })

    if best_weights is not None:
        model.set_weights(best_weights)
        selected_model = "best_validation_autoregressive"
        selected_epoch = best_epoch
    else:
        selected_model = "final_epoch_legacy"
        selected_epoch = total_epochs
    stepwise = model.predict_proba(x_test)
    autoreg = pred_evo(model, x_test, args.test_evolutions, args.steps, args.window_size, i_x)
    evo_size = args.steps - 1
    autoreg_true_map = y_test.reshape(args.test_evolutions, evo_size, i_x)
    autoreg_pred_map = autoreg.reshape(args.test_evolutions, evo_size, i_x)
    input_profiles = x_test[::evo_size, 0, :]
    truth = np.concatenate([input_profiles[:, None, :], autoreg_true_map], axis=1)
    fixed_horizon = fixed_horizon_metrics(
        model, truth, args.window_size, args.eval_fixed_horizons
    )
    metrics = {
        "stepwise_r2": float(r2_score(y_test.reshape(-1), stepwise.reshape(-1))),
        "autoregressive_r2": float(r2_score(y_test.reshape(-1), autoreg.reshape(-1))),
        "autoregressive_final_r2": float(r2_score(autoreg_true_map[:, -1].reshape(-1), autoreg_pred_map[:, -1].reshape(-1))),
        "normalization": args.normalization,
        "seed": args.seed,
        "steps": args.steps,
        "window_size": args.window_size,
        "validation_evolutions": args.validation_evolutions,
        "checkpoint_every": args.checkpoint_every,
        "selection_metric": "validation_autoregressive_r2" if x_validation is not None else "final_epoch_legacy",
        "selected_model": selected_model,
        "selected_epoch": selected_epoch,
        "best_validation_autoregressive_r2": best_validation_r2 if best_weights is not None else None,
        "fixed_horizon_metrics": fixed_horizon,
    }
    model.save(os.path.join(args.output_dir, "keras_model.h5"))
    sio.savemat(os.path.join(args.output_dir, "stepwise_predictions.mat"), {"Y_submit": stepwise, "Y_test": y_test})
    sio.savemat(os.path.join(args.output_dir, "autoregressive_predictions.mat"), {"Y_submit": autoreg, "Y_test": y_test})
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(args.output_dir, "fixed_horizon_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(fixed_horizon, handle, indent=2)
    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as handle:
        json.dump({"epochs": selection_history}, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
