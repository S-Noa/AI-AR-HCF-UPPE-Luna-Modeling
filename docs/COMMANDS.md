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
