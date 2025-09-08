#!/usr/bin/env python3
"""
Script to check all package versions for the OOD Transformer-PCA experiment
"""

import sys
import subprocess

def get_package_version(package_name):
    """Get version of a package using multiple methods"""
    try:
        # Method 1: Import and check __version__
        module = __import__(package_name)
        if hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, 'VERSION'):
            return module.VERSION
        else:
            return "Version attribute not found"
    except ImportError:
        return "Not installed"
    except Exception as e:
        return f"Error: {e}"

def get_pip_version(package_name):
    """Get version using pip show"""
    try:
        result = subprocess.run(['pip', 'show', package_name], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':', 1)[1].strip()
        return "Not found in pip"
    except:
        return "Not available via pip"

def main():
    print("=" * 60)
    print("PACKAGE VERSION CHECK FOR OOD TRANSFORMER-PCA EXPERIMENT")
    print("=" * 60)
    
    # Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Essential packages for the experiment
    packages = [
        'torch',
        'torchvision', 
        'transformers',
        'numpy',
        'sklearn',  # scikit-learn imports as sklearn
        'matplotlib',
        'wandb',
        'tqdm',
        'joblib'
    ]
    
    print("Package Versions:")
    print("-" * 40)
    
    for package in packages:
        version = get_package_version(package)
        if version == "Not installed":
            # Try alternative import names
            if package == 'sklearn':
                version = get_package_version('scikit-learn')
        
        pip_version = get_pip_version(package if package != 'sklearn' else 'scikit-learn')
        
        print(f"{package:15}: {version:20} (pip: {pip_version})")
    
    print("\n" + "=" * 60)
    print("CUDA AVAILABILITY CHECK")
    print("=" * 60)
    
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No CUDA GPUs available")
    except ImportError:
        print("PyTorch not installed - cannot check CUDA")
    
    print("\n" + "=" * 60)
    print("COMPATIBILITY CHECK")
    print("=" * 60)
    
    # Check if we can import key modules from our experiment
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        import wandb
        from transformers import GPT2Config
        print("✅ All critical imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
    
    print("\nIf you see any 'Not installed' packages above, install them with:")
    print("pip install torch torchvision transformers numpy scikit-learn matplotlib wandb tqdm joblib")

if __name__ == "__main__":
    main()
