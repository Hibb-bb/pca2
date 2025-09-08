# OOD Experiment Files

## Quick Start:
1. Make sure you have the conda environment set up with PyTorch, wandb, etc.
2. Run: `sbatch run_ood_experiments.sh`

## What this does:
- Generates MNIST datasets split by digits (0-4 train, 5-9 test; even/odd splits)
- Trains transformer models on training digits
- Tests on out-of-distribution digits
- Logs results to wandb

## Files:
- `run_ood_experiments.sh` - Main SLURM script to run everything
- `data_prepare_ood.py` - Generates OOD datasets
- `main_realworld_0in1.py` - Main training script
- `model.py` - Transformer model definitions
- `loss.py` - Loss functions  
- `data_generation.py` - Data utilities

## Key Settings:
- Model: 12 layers, 256 embedding, 8 heads
- Data: D=10, N=10, k=3,4,5
- Seeds: 1234, 1235, 1236
- Evaluation every 5000 steps

## Expected Output:
- Datasets saved to `dataset/` folder
- Model checkpoints in `ckpt/` folder
- Results logged to wandb project "transformer_pca"
- Evaluation metrics: `eigenvector_cos_similarity_1`, etc.
