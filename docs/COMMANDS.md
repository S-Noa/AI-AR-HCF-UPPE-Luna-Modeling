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

## Visualize all extreme samples with z zooms

```bash
cd /mnt/Luna.jl-master/Luna.jl-master/examples/simple_interface

nohup julia --project=../.. visualize_hdf5.jl \
  --extreme-samples \
  --all \
  --data-dir ../../extreme_search_t650_all_runnable \
  --output-dir extreme_samples_output/all_zzooms \
  --z-zooms-cm 50 10 5 1 \
  --skip-highres-rerun \
  > visualize_extreme_all_zzooms.log 2>&1 &
```

## Transformer temporal fine-tuning example

```bash
cd /mnt/Luna.jl-master/Luna.jl-master/examples/simple_interface

python3 train_mlp.py \
  --input-dir processed_data_t650_v7 \
  --output-dir models_t650_transformer_temporal_finetune_v4 \
  --model transformer \
  --transformer-mode temporal \
  --transformer-d-model 192 \
  --transformer-heads 4 \
  --transformer-layers 4 \
  --transformer-use-z-embedding \
  --transformer-band-boundaries-nm 200 700 1200 1800 2500 \
  --finetune-from models_t650_transformer_temporal_v3/best_val_r2_model.pth \
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
