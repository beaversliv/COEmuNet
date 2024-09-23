# Project title
The goal of this project is to build a surrogate model of [P3DROSLO](https://github.com/Magritte-code/pomme) to slove the forward problem of Radiative Transfer Equation without scattering effect.

Frederik currently works on the paper Probabilistic 3D-Reconstruction of Spectral Line Observations (p3droslo) which first implements the forward function in PyTorch and benefits from automatic differentiable functionality and will then solves the inverse problem of reconstructing a model. Presently, the PyTorch implementation of the forward problem in this study affords Shiqi the opportunity to generate simulated data needed to train a network.

## Requirements
- Python 3.12.4
- Packages
    ```
    pip install -r requirements.txt
    ```
## Data Generation
We have three types of data: single direction dataset, multi directions dataset and multi frequencies datset.
- Generate single direction dataset
    ```
    python /physical_informed/data/data_gen/data_gen.py \
        --type original \
        -- model_grid 64 --num_rotations 100
    ```
- Generate multi directions dataset
    ```
    python /physical_informed/data/data_gen/data_gen.py \
        --type rotation \
        -- model_grid 64 --num_rotations 100
    ```
- Generate multi frequencies dataset
    ```
    python /physical_informed/data/data_gen/data_gen.py \
        --type rotation \
        -- model_grid 64 --num_rotations 100 --mulfreq
    ```
## Train a 3DResNet based autoencoder
The configurations for different datasets are written as 
`/config/faceon_dataset.yaml`,\
`/config/rotation_dataset.yaml`,\
`/config/mulfreq_dataset.yaml`.
- Take single direction case for example:
    ```
    python single.py --config /config/faceon_dataset.yaml
    ```
-  Enable Multi-GPU training:
    ```
    python ddp.py --config /config/faceon_dataset.yaml
    ```
- Overwrite arguments
    ```
    python ddp.py --config /config/faceon_dataset.yaml \
        --batch_size 128 --seed 1234 --lr 0.004 \
        --epochs 500 --alpha 0.8 --beta 0.2 \
        --save_path <custom directory>
    ```
## Reproducing Experiment
The checkpoint can be downloaded from my [drive]