#!/usr/bin/env python3
"""
setup_tpu.py
Setup script for TPU training environment with all necessary dependencies.
Run this script on your TPU instance to install required packages.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n[SETUP] {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def check_tpu_environment():
    """Check if running on TPU environment."""
    print("\n[CHECK] Verifying TPU environment...")
    
    # Check for TPU environment variables
    tpu_name = os.environ.get('TPU_NAME')
    colab_tpu = os.environ.get('COLAB_TPU_ADDR')
    
    if tpu_name or colab_tpu:
        print(f"✓ TPU environment detected!")
        if tpu_name:
            print(f"  TPU_NAME: {tpu_name}")
        if colab_tpu:
            print(f"  COLAB_TPU_ADDR: {colab_tpu}")
        return True
    else:
        print("⚠ Warning: TPU environment variables not found.")
        print("  Make sure you're running on a TPU instance.")
        return False

def install_pytorch_xla():
    """Install PyTorch XLA for TPU support."""
    print("\n[INSTALL] Installing PyTorch XLA...")
    
    # For TPU v3/v4 with Python 3.8+
    commands = [
        "pip install torch torchvision torchaudio",
        "pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html",
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Installing: {cmd.split()[2]}"):
            return False
    
    return True

def install_dependencies():
    """Install other required dependencies."""
    print("\n[INSTALL] Installing additional dependencies...")
    
    dependencies = [
        "numpy",
        "pyyaml", 
        "tqdm",
        "wandb",  # Optional for experiment tracking
        "datasets",  # Optional for data processing
    ]
    
    cmd = f"pip install {' '.join(dependencies)}"
    return run_command(cmd, "Installing Python dependencies")

def verify_installation():
    """Verify that PyTorch XLA is properly installed."""
    print("\n[VERIFY] Testing PyTorch XLA installation...")
    
    test_script = """
import torch
import torch_xla
import torch_xla.core.xla_model as xm

print(f"PyTorch version: {torch.__version__}")
print(f"XLA version: {torch_xla.__version__}")

device = xm.xla_device()
print(f"XLA device: {device}")

# Test basic tensor operations
x = torch.randn(3, 3, device=device)
y = torch.randn(3, 3, device=device)
z = x + y
print(f"Test tensor operation successful: {z.shape}")

print("✓ PyTorch XLA is working correctly!")
"""
    
    try:
        exec(test_script)
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file for TPU training."""
    config_content = """# sample_tpu_config.yaml
# Basic configuration for TPU training

model:
  max_seq_len: 512  # Smaller for testing
  vocab_size: 50304
  n_layers: 6       # Smaller model for testing
  n_heads: 8
  d_model: 512
  dropout: 0.1
  bias: true

data_dir: "data/"
sequence_length: 512
batch_size: 16        # Adjust based on TPU memory
tpu_cores: 8
mixed_precision: true
learning_rate: 1e-4
max_steps: 10000
warmup_steps: 1000
save_interval: 500
log_interval: 10
out_dir: "checkpoints_tpu"
"""
    
    with open("sample_tpu_config.yaml", "w") as f:
        f.write(config_content)
    
    print("\n[SETUP] Created sample_tpu_config.yaml")
    print("Edit this file according to your needs before training.")

def main():
    print("=" * 60)
    print("         LUNARIS CODEX TPU SETUP")
    print("=" * 60)
    
    # Check TPU environment
    check_tpu_environment()
    
    # Install PyTorch XLA
    if not install_pytorch_xla():
        print("\n✗ Failed to install PyTorch XLA")
        sys.exit(1)
    
    # Install other dependencies
    if not install_dependencies():
        print("\n✗ Failed to install dependencies")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Installation verification failed")
        sys.exit(1)
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "=" * 60)
    print("✓ TPU SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Prepare your data in .npy format in the 'data/' directory")
    print("2. Edit sample_tpu_config.yaml according to your needs")
    print("3. Run: python train_tpu.py sample_tpu_config.yaml")
    print("\nFor Google Colab TPU, make sure to:")
    print("- Enable TPU in Runtime > Change runtime type")
    print("- Run this setup script first")
    print("- Use the provided config file")

if __name__ == "__main__":
    main()
