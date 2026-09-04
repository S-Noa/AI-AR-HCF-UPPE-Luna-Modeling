# Common commands

Run commands from the outer project root unless a command explicitly changes directories.

## Git setup

```powershell
git init
git branch -M main
git remote add origin https://github.com/S-Noa/AI-AR-HCF-UPPE-Luna-Modeling.git
git status --short
git diff --cached --stat
git push -u origin main
```

## Julia package context

```powershell
cd Luna.jl-master
julia --project=. -e "using Pkg; Pkg.status()"
```

## Cloud runtime setup

Use the Git-synced cloud code checkout, while keeping large legacy datasets under
`/mnt/Luna.jl-master`.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$LUNA_PROJECT"
julia --project=. -e 'using Luna'
```

For Python training scripts on the cloud, use the same environment script so
PyTorch can find the Corex CUDA runtime:

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$LUNA_PROJECT/examples/simple_interface"
python3 train_mlp.py --help
```

## Visualize all extreme samples with z zooms

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$LUNA_PROJECT/examples/simple_interface"

nohup julia --project="$LUNA_PROJECT" visualize_hdf5.jl \
  --extreme-samples \
  --all \
  --data-dir "$LUNA_LEGACY_DATA_ROOT/extreme_search_t650_all_runnable" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/extreme_samples_output/all_zzooms" \
  --z-zooms-cm 50 10 5 1 \
  --skip-highres-rerun \
  > visualize_extreme_all_zzooms.log 2>&1 &
```

## Transformer temporal fine-tuning example

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$LUNA_PROJECT/examples/simple_interface"

python3 train_mlp.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/processed_data_t650_v7" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/models_t650_transformer_temporal_finetune_v4" \
  --model transformer \
  --transformer-mode temporal \
  --transformer-d-model 192 \
  --transformer-heads 4 \
  --transformer-layers 4 \
  --transformer-use-z-embedding \
  --transformer-band-boundaries-nm 200 700 1200 1800 2500 \
  --finetune-from "$LUNA_LEGACY_DATA_ROOT/models_t650_transformer_temporal_v3/best_val_r2_model.pth" \
  --learning-rate 5e-5 \
  --epochs 100 \
  --batch-size 4 \
  --selection-metric val_temporal_r2 \
  --early-z-max-cm 10 \
  --early-z-loss-weight 1.2 \
  --lambda-gradient-weight 0.03 \
  --uv-loss-weight 1.2 \
  --long-wavelength-weight 1.3 \
  --long-wavelength-cutoff-nm 1200 \
  --visualize
```

## Prepare Luna RNN conditional data

```bash
cd /mnt/Luna.jl-master/rnnnonlinear-master/rnnnonlinear-master

nohup python3 prepare_luna_data.py \
  --input-dir ../../processed_data_t650_v7 \
  --output simulations/luna_t650_temporal_conditional.mat \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  --log-file prepare_luna_t650_conditional.log \
  > prepare_luna_t650_conditional.nohup.log 2>&1 &
```

## Luna RNN conditional baseline

```bash
python3 train_luna_rnn.py \
  --data simulations/luna_t650_temporal_conditional.mat \
  --output-dir results_luna_rnn_features_z_v2 \
  --conditioning features_z \
  --training-mode one_step \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 1e-4 \
  --epochs 80 \
  --batch-size 128 \
  --log-file results_luna_rnn_features_z_v2/train.log
```

## t0p6 early-dense preprocessing for RNN baseline

Use relative-intensity targets for this baseline so it remains comparable with
the existing normalized-shape RNN/Transformer/CNN experiments. The raw
early-dense directory contains two z grids, so preprocess only the filtered
`z_len=1401` symlink directory.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$LUNA_PROJECT/examples/simple_interface"

python3 - <<'PY'
import csv, os
manifest = '/mnt/Luna.jl-master/earlydense_manifests/t0p6_earlydense_zlen_manifest.csv'
links = '/mnt/Luna.jl-master/training_data_ar_t0p6_earlydense_z1401_links'
os.makedirs(links, exist_ok=True)
count = 0
with open(manifest, newline='') as fh:
    for row in csv.DictReader(fh):
        if row['z_len'] == '1401' and row['early_dense_saveN'] == '1001' and row['early_dense_zmax_cm'] == '10.0':
            dst = os.path.join(links, row['filename'])
            if not os.path.exists(dst):
                os.symlink(row['path'], dst)
            count += 1
print(f'z1401 links: {count}')
PY

nohup python3 data_preprocessing.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/training_data_ar_t0p6_earlydense_z1401_links" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_v1" \
  --sample-filter earlydense \
  --single-thickness-mode \
  --thickness 0.65 \
  --target-points 1000 \
  --spectrum-normalization per_sample_minmax \
  --test-size 0.15 \
  --val-size 0.15 \
  --batch-size 16 \
  > "$LUNA_LEGACY_DATA_ROOT/preprocess_t0p6_earlydense_z1401_v1.log" 2>&1 &

tail -f "$LUNA_LEGACY_DATA_ROOT/preprocess_t0p6_earlydense_z1401_v1.log"
```

## Export t0p6 early-dense data for Luna RNN

Run this only after preprocessing has completed and
`processed_t0p6_earlydense_z1401_v1/processing_params.json` exists.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
mkdir -p "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations"
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

nohup python3 prepare_luna_data.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_v1" \
  --output "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat" \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_t0p6_earlydense_z1401_conditional.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_t0p6_earlydense_z1401_conditional.nohup.log" 2>&1 &

tail -f "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_t0p6_earlydense_z1401_conditional.log"
```

## t0p6 early-dense Luna RNN smoke test

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_smoke_legacy_z1401" \
  --training-mode open_source_legacy \
  --conditioning none \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 1e-4 \
  --epochs 2 \
  --batch-size 128 \
  --train-evolutions 200 \
  --test-evolutions 50 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 16 \
  --autoregressive-eval-batch-size 8 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_smoke_legacy_z1401/train.log"
```

## t0p6 early-dense Luna RNN baselines

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_legacy_baseline_z1401_v1" \
  --training-mode open_source_legacy \
  --conditioning none \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 1e-4 \
  --epochs 80 \
  --batch-size 128 \
  --eval-autoregressive-every 2 \
  --eval-autoregressive-samples 64 \
  --autoregressive-eval-batch-size 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_legacy_baseline_z1401_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_legacy_baseline_z1401_v1.nohup.log" 2>&1 &

nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_z1401_v1" \
  --training-mode conditional_legacy \
  --conditioning features_z \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 1e-4 \
  --epochs 80 \
  --batch-size 128 \
  --eval-autoregressive-every 2 \
  --eval-autoregressive-samples 64 \
  --autoregressive-eval-batch-size 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_z1401_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_z1401_v1.nohup.log" 2>&1 &
```

## t0p6 early-dense scheduled-sampling RNN

Use this after the `conditional_legacy` one-step comparison shows high
stepwise accuracy but unstable autoregressive rollout.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_smoke_features_z_scheduled_z1401" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --rollout-steps 50 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.5 \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 1e-4 \
  --epochs 2 \
  --batch-size 16 \
  --train-evolutions 200 \
  --test-evolutions 50 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 16 \
  --autoregressive-eval-batch-size 8 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_smoke_features_z_scheduled_z1401/train.log"

nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_scheduled_z1401_v1" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --rollout-steps 50 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.5 \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 1e-4 \
  --epochs 80 \
  --batch-size 16 \
  --eval-autoregressive-every 2 \
  --eval-autoregressive-samples 64 \
  --autoregressive-eval-batch-size 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_scheduled_z1401_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_scheduled_z1401_v1.nohup.log" 2>&1 &

tail -f "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_scheduled_z1401_v1/train.log"
```

## Warm-start scheduled-sampling RNN

Use this after a conditional one-step checkpoint exists. This preserves the
local one-step propagation mapping, then fine-tunes for autoregressive
stability with a lower learning rate and gentler feedback schedule.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_smoke_features_z_warm_scheduled_z1401" \
  --init-from "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_z1401_v1/best_stepwise_model.pth" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --rollout-steps 100 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.2 \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 2e-5 \
  --epochs 2 \
  --batch-size 8 \
  --train-evolutions 200 \
  --test-evolutions 50 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 16 \
  --autoregressive-eval-batch-size 8 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_smoke_features_z_warm_scheduled_z1401/train.log"

nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z1401_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_warm_scheduled_z1401_v1" \
  --init-from "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_z1401_v1/best_stepwise_model.pth" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --rollout-steps 100 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.2 \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 2e-5 \
  --epochs 40 \
  --batch-size 8 \
  --eval-autoregressive-every 2 \
  --eval-autoregressive-samples 64 \
  --autoregressive-eval-batch-size 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_warm_scheduled_z1401_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_warm_scheduled_z1401_v1.nohup.log" 2>&1 &
```

## Early-10cm downsampled RNN diagnostic

This creates a smaller RNN task from the same processed early-dense data:
front 20% of the propagation distance, downsampled to 201 z points.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

python3 prepare_luna_data.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_v1" \
  --output "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_201_conditional.mat" \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  --z-max-fraction 0.2 \
  --z-target-points 201 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_t0p6_earlydense_z10cm_201_conditional.log"

nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_201_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_scheduled_z10cm_201_v1" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --rollout-steps 100 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.3 \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 1e-4 \
  --epochs 60 \
  --batch-size 16 \
  --eval-autoregressive-every 2 \
  --eval-autoregressive-samples 64 \
  --autoregressive-eval-batch-size 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_scheduled_z10cm_201_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_features_z_scheduled_z10cm_201_v1.nohup.log" 2>&1 &
```

## RNN paper-style dB preprocessing

Use this route only for Luna RNN autoregressive diagnostics. It follows the
Salmela RNN baseline normalization style: global training-set maximum, dB
compression, `-55 dB` floor, and `[0,1]` targets.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$LUNA_PROJECT/examples/simple_interface"

nohup python3 data_preprocessing.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/training_data_ar_t0p6_earlydense_z1401_links" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_rnnpaperdb_v1" \
  --sample-filter earlydense \
  --single-thickness-mode \
  --thickness 0.65 \
  --target-points 1000 \
  --spectrum-normalization rnn_paper_db \
  --test-size 0.15 \
  --val-size 0.15 \
  --batch-size 16 \
  > "$LUNA_LEGACY_DATA_ROOT/preprocess_t0p6_earlydense_z1401_rnnpaperdb_v1.nohup.log" 2>&1 &

tail -f "$LUNA_LEGACY_DATA_ROOT/preprocess_t0p6_earlydense_z1401_rnnpaperdb_v1.nohup.log"
```

## RNN paper-style z exports

Run after `processed_t0p6_earlydense_z1401_rnnpaperdb_v1/processing_params.json`
exists.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

python3 prepare_luna_data.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_rnnpaperdb_v1" \
  --output "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_101_rnnpaperdb_conditional.mat" \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  --z-max-fraction 0.2 \
  --z-target-points 101 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_z10cm_101_rnnpaperdb.log"

python3 prepare_luna_data.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_rnnpaperdb_v1" \
  --output "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_201_rnnpaperdb_conditional.mat" \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  --z-max-fraction 0.2 \
  --z-target-points 201 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_z10cm_201_rnnpaperdb.log"

python3 prepare_luna_data.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_rnnpaperdb_v1" \
  --output "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z50cm_501_rnnpaperdb_conditional.mat" \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  --z-target-points 501 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_z50cm_501_rnnpaperdb.log"
```

## RNN paper-style autoregressive runs

Direct scheduled-sampling baseline on the shortest diagnostic task:

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_101_rnnpaperdb_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_101_rnnpaperdb_v1" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --prediction-target direct \
  --output-activation sigmoid \
  --rollout-steps-start 20 \
  --rollout-steps-end 100 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.15 \
  --rollout-start-mode mixed \
  --zero-start-prob 0.5 \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 5e-5 \
  --epochs 80 \
  --batch-size 16 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 128 \
  --autoregressive-eval-batch-size 8 \
  --early-stop-on-autoreg \
  --autoreg-patience 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_101_rnnpaperdb_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_101_rnnpaperdb_v1.nohup.log" 2>&1 &
```

Residual scheduled-sampling baseline on the same task:

```bash
nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_101_rnnpaperdb_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_residual_z10cm_101_rnnpaperdb_v1" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --prediction-target residual \
  --output-activation identity \
  --rollout-steps-start 20 \
  --rollout-steps-end 100 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.15 \
  --rollout-start-mode mixed \
  --zero-start-prob 0.5 \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 5e-5 \
  --epochs 80 \
  --batch-size 16 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 128 \
  --autoregressive-eval-batch-size 8 \
  --early-stop-on-autoreg \
  --autoreg-patience 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_residual_z10cm_101_rnnpaperdb_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_residual_z10cm_101_rnnpaperdb_v1.nohup.log" 2>&1 &
```

## RNN normalization diagnostics and stable short-rollout baseline

The original `rnn_paper_db` setting with a `-55 dB` floor is too sparse for
the Luna early-dense spectra. Less aggressive variants can be tested without
changing the default behavior:

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$LUNA_PROJECT/examples/simple_interface"

nohup python3 data_preprocessing.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/training_data_ar_t0p6_earlydense_z1401_links" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_rnnpaperdb_m80_v1" \
  --sample-filter earlydense \
  --single-thickness-mode \
  --thickness 0.65 \
  --target-points 1000 \
  --spectrum-normalization rnn_paper_db \
  --rnn-db-reference-mode max \
  --rnn-db-floor -80 \
  --test-size 0.15 \
  --val-size 0.15 \
  --batch-size 16 \
  > "$LUNA_LEGACY_DATA_ROOT/preprocess_t0p6_earlydense_z1401_rnnpaperdb_m80_v1.nohup.log" 2>&1 &
```

Current preferred route for a usable autoregressive RNN baseline is the
relative-shape task with a shorter front-10-cm rollout:

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

python3 prepare_luna_data.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_v1" \
  --output "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_101_conditional.mat" \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  --z-max-fraction 0.2 \
  --z-target-points 101 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_t0p6_earlydense_z10cm_101_conditional.log"

nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_101_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_101_perminmax_stable_v1" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --prediction-target direct \
  --output-activation sigmoid \
  --rollout-steps-start 100 \
  --rollout-steps-end 100 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.05 \
  --rollout-start-mode mixed \
  --zero-start-prob 0.7 \
  --no-detach-feedback \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 5e-5 \
  --grad-clip 1.0 \
  --epochs 30 \
  --batch-size 16 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 256 \
  --autoregressive-eval-batch-size 8 \
  --early-stop-on-autoreg \
  --autoreg-patience 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_101_perminmax_stable_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_101_perminmax_stable_v1.nohup.log" 2>&1 &
```

If the 101-point task remains positive, run the 201-point front-10-cm
comparison:

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_201_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_201_perminmax_stable_v1" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --prediction-target direct \
  --output-activation sigmoid \
  --rollout-steps-start 100 \
  --rollout-steps-end 100 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.05 \
  --rollout-start-mode mixed \
  --zero-start-prob 0.7 \
  --no-detach-feedback \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 5e-5 \
  --grad-clip 1.0 \
  --epochs 30 \
  --batch-size 16 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 256 \
  --autoregressive-eval-batch-size 8 \
  --early-stop-on-autoreg \
  --autoreg-patience 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_201_perminmax_stable_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_201_perminmax_stable_v1.nohup.log" 2>&1 &
```

Queued follow-up after the 201-point run: warm-start the 101-point model and
test a coarser 51-point front-10-cm rollout.

```bash
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

# Export the coarser z10cm_51 dataset if missing.
python3 prepare_luna_data.py \
  --input-dir "$LUNA_LEGACY_DATA_ROOT/processed_t0p6_earlydense_z1401_v1" \
  --output "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_51_conditional.mat" \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  --z-max-fraction 0.2 \
  --z-target-points 51 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/prepare_t0p6_earlydense_z10cm_51_conditional.log"

# A1: short one-step warm-start on z10cm_101.
python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_101_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_onestep_z10cm_101_perminmax_warm_v1" \
  --training-mode conditional_legacy \
  --conditioning features_z \
  --prediction-target direct \
  --output-activation sigmoid \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 1e-4 \
  --grad-clip 1.0 \
  --epochs 5 \
  --batch-size 128 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 128 \
  --autoregressive-eval-batch-size 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_onestep_z10cm_101_perminmax_warm_v1/train.log"

# A2: scheduled sampling initialized from the one-step checkpoint.
nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_101_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_warm_scheduled_z10cm_101_perminmax_v1" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --init-from "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_onestep_z10cm_101_perminmax_warm_v1/best_stepwise_model.pth" \
  --prediction-target direct \
  --output-activation sigmoid \
  --rollout-steps-start 20 \
  --rollout-steps-end 60 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.02 \
  --rollout-start-mode mixed \
  --zero-start-prob 0.8 \
  --no-detach-feedback \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 3e-5 \
  --grad-clip 1.0 \
  --epochs 30 \
  --batch-size 16 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 256 \
  --autoregressive-eval-batch-size 8 \
  --early-stop-on-autoreg \
  --autoreg-patience 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_warm_scheduled_z10cm_101_perminmax_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_warm_scheduled_z10cm_101_perminmax_v1.nohup.log" 2>&1 &

# B: coarser z10cm_51 direct scheduled-sampling baseline.
nohup python3 train_luna_rnn.py \
  --data "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/simulations/luna_t0p6_earlydense_z10cm_51_conditional.mat" \
  --output-dir "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_51_perminmax_stable_v1" \
  --training-mode scheduled_sampling \
  --conditioning features_z \
  --prediction-target direct \
  --output-activation sigmoid \
  --rollout-steps-start 50 \
  --rollout-steps-end 50 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.05 \
  --rollout-start-mode mixed \
  --zero-start-prob 0.8 \
  --no-detach-feedback \
  --window-size 10 \
  --hidden 250 \
  --learning-rate 5e-5 \
  --grad-clip 1.0 \
  --epochs 30 \
  --batch-size 16 \
  --eval-autoregressive-every 1 \
  --eval-autoregressive-samples 256 \
  --autoregressive-eval-batch-size 8 \
  --early-stop-on-autoreg \
  --autoreg-patience 8 \
  --checkpoint-every 1 \
  --log-file "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_51_perminmax_stable_v1/train.log" \
  > "$LUNA_LEGACY_DATA_ROOT/rnn_earlydense/results_direct_z10cm_51_perminmax_stable_v1.nohup.log" 2>&1 &
```
