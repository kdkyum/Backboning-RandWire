# Searching Wiring Patterns of Neural Networks with Graph Backboning

## Requirements

```
pip install -r requirements.txt
```

* If you want to calculate FLOPs of the model, you should replace the `thop` pacakage's `thop/profile.py` to `utils/profile.py`

## Training base models

* DATA_DIR : Path of data directory.
* DATASET : Name of dataset {'cifar10', 'cifar100', 'tiny_imagenet'}
* MODEL_DIR : Path of model will be saved.
* SEED : Random seed.

```
python train.py {DATA_DIR} \\
    --arch complete \\
    --dataset {DATASET} -j 1 \\
    --epochs 100 \\
    --lr 0.05 \\
    --wd 3e-4 \\
    --graph-wd 0 \\
    --model-dir {SAVE_DIR} \\
    --batch-size 96 \\
    --print-freq 100 \\
    --N 32 \\
    --drop-path 0.1 \\
    --noScheduleDrop \\
    --channels 68 \\
    --edge-act softplus \\
    --init_norm \\
    --in-nodes 5 \\
    --out-nodes 5 \\
    --seed {SEED}
```

## Graph backboning

* DATA_DIR : Path of data directory.
* DATASET : Name of dataset {'cifar10', 'cifar100', 'tiny_imagenet'}
* BASE_DIR : Path of base model will be backboned.
* SAVE_DIR : Path of backboned model will be saved.
* CRITERION : Backboning criteria {'naive', 'in_disparity', 'BC', 'random'}
* EDGES : Target number of edges after graph backboning.
* SEED : Random seed.

```
python prune.py {DATA_DIR} \\
    -a complete  \\
    --dataset {DATASET} -j 1 \\
    --resume {BASE_DIR} \\
    --save {SAVE_DIR} \\
    --prune-criterion {CRITERION} \\
    --channels 68 \\
    --N 32 \\
    --in-nodes 5 \\
    --out-nodes 5 \\
    --edge-act softplus \\
    --init_norm \\
    --prune-edges {EDGES}
    --seed {SEED} \\
```

## Training the backboned graph model

* DATA_DIR : Path of data directory.
* DATASET : Name of dataset {'cifar10', 'cifar100', 'tiny_imagenet'}
* T : Training epochs
* MODEL_DIR : Path of model will be saved.
* RESUME_DIR : Path of the backboned model is saved.
* SEED : Random seed.

```
python -u train.py {DATA_DIR} \\
    --arch complete \\
    --dataset {DATASET} -j 1 \\
    --epochs {T} \\
    --lr 0.05 \\
    --wd 3e-4 \\
    --graph-wd 0 \\
    --model-dir {MODEL_DIR} \\
    --resume {RESUME_DIR} \\
    --batch-size 96 \\
    --print-freq 100 \\
    --drop-path 0.1 \\
    --noScheduleDrop \\
    --channels 68 \\
    --N 32 \\
    --in-nodes 5 \\
    --out-nodes 5 \\
    --edge-act softplus \\
    --init_norm \\
    --seed {SEED}
```