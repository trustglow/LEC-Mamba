# LEC-Mamba

PyTorch implementation of **LEC-Mamba** for multimodal remote sensing image classification.

LEC-Mamba integrates local structural feature extraction, Mamba sequence modeling, a multi-scale encoder–decoder, and center-guided detail enhancement to classify the center pixel of spatially co-registered hyperspectral and auxiliary-modality image patches. This release provides the implementation and training script for the MUUFL Gulfport dataset.

## Requirements

- Python 3
- PyTorch with CUDA support
- NumPy
- SciPy
- scikit-learn
- einops
- tqdm

Install a CUDA-compatible PyTorch build following the [official installation instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
pip install numpy scipy scikit-learn einops tqdm
```

The current differential convolution implementation requires a CUDA-capable GPU. The Mamba implementation supports an optional `mamba-ssm` backend and uses the included PyTorch implementation when that backend is unavailable.

## Data Preparation

Prepare the spatially co-registered MUUFL hyperspectral image, LiDAR image, and ground-truth map as MATLAB files. Place them in the following directory relative to `main.py`:

```text
Datasets/
└── MUUFL/
    ├── HSI.mat
    ├── LiDAR.mat
    └── muufl_gt.mat
```

| File | Variable | Description |
| --- | --- | --- |
| `HSI.mat` | `HSI` | Hyperspectral image |
| `LiDAR.mat` | `LiDAR` | LiDAR image |
| `muufl_gt.mat` | `gt` | Ground-truth labels |

The default model uses 64 HSI channels, 2 auxiliary channels, 13 × 13 input patches, and 11 classes. The spatial dimensions of both images and the ground-truth map must match. Dataset loading is defined in `load_data.py`.

## Usage

Run the following command from the directory containing `main.py`:

```bash
python main.py --learning_rate 0.001 --train_bs 64
```

| Argument | Description | Default |
| --- | --- | --- |
| `--learning_rate` | Initial learning rate | `0.001` |
| `--train_bs` | Batch size for training and evaluation | `64` |

The script trains the model and evaluates it after each epoch. The checkpoint with the highest evaluation accuracy is saved to `/tmp/output/model.pth`, including the model and optimizer states.

Additional experiment settings are defined in the source files:

- `main.py`: dataset selection, data split, random seeds, number of epochs, and checkpoint path.
- `model.py`: input channels, patch size, number of classes, and network configuration.

The provided MUUFL configuration uses 200 epochs, the AdamW optimizer, and a stratified split with 5% of labeled samples for training and 95% for evaluation. The evaluation split is also used to select the checkpoint.

## Code Structure

| File | Description |
| --- | --- |
| `main.py` | Training and evaluation entry point |
| `model.py` | Complete LEC-Mamba network |
| `mtdc.py` | Multi-topology differential convolution module |
| `lpg.py` | Local-pointwise gating module |
| `c2r.py` | Center-contrast refinement module |
| `bimamba.py` | Mamba sequence modeling |
| `pscan.py` | Parallel scan implementation |
| `load_data.py` | Dataset loading |
| `utils.py` | Data preprocessing, patch construction, and data loading utilities |

## Citation

If you find this code useful for your research, please cite the accompanying LEC-Mamba paper. The bibliographic information and BibTeX entry will be added when available.

## Acknowledgements

We thank the authors of [Mamba](https://github.com/state-spaces/mamba) and [mamba-minimal](https://github.com/johnma2006/mamba-minimal) for making their implementations publicly available.
