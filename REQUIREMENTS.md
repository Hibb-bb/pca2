# Requirements for OOD Transformer-PCA Experiments

## Python Dependencies (Tested Versions):
python==3.12.7
torch==2.5.1+cu124
torchvision==0.20.1+cu124
transformers==4.46.2
numpy==2.1.3
scikit-learn==1.5.2
matplotlib==3.9.3
wandb==0.18.7
tqdm==4.66.6
joblib==1.4.2

## Minimum Compatible Versions:
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.20.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.0.0
wandb>=0.12.0
tqdm>=4.60.0
joblib>=1.0.0


## Quick Version Check:
Run this to verify all packages are installed correctly:
```bash
python check_versions.py
```

## Wandb Setup:
Make sure to set your wandb API key in the script or as environment variable:
```bash
export WANDB_API_KEY=your_api_key_here
```

## Directory Structure:
The script will create these directories automatically:
- `dataset/` - for generated OOD datasets
- `ckpt/` - for model checkpoints  
- `error/` - for SLURM error logs
- `output/` - for SLURM output logs
