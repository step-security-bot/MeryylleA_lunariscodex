# train_tpu.py
# A robust, feature-rich training script for the LunarisCodex model optimized for Google TPUs.
# Based on the original train.py but adapted for PyTorch XLA and TPU-specific optimizations.

import os
import time
import math
import glob
from dataclasses import dataclass, field
from typing import Optional
from contextlib import nullcontext

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# TPU-specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from torch_xla.amp import autocast, GradScaler

# Assuming model.py contains the LunarisCodex and LunarisCodexConfig classes
from model import LunarisCodex, LunarisCodexConfig

# --- Configuration Dataclass ---
@dataclass
class TrainConfig:
    # Model configuration
    model: LunarisCodexConfig = field(default_factory=LunarisCodexConfig)

    # Data configuration
    data_dir: str = "data/"
    sequence_length: int = 1024

    # Optimizer configuration
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # Scheduler configuration
    warmup_steps: int = 2000
    max_steps: int = 600000

    # Training configuration
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1  # Set to a large number for step-based training
    grad_clip: float = 1.0
    compile_model: bool = False  # Disabled for TPU compatibility

    # TPU-specific configuration
    tpu_cores: int = 8  # Number of TPU cores to use
    mixed_precision: bool = True  # Use mixed precision training
    dataloader_num_workers: int = 4

    # I/O and Logging
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000

    # W&B configuration
    wandb_project: Optional[str] = "lunaris-codex-tpu"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"tpu-run-{time.strftime('%Y-%m-%d-%H-%M')}"

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file, ensuring correct types."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        model_config_dict = config_dict.pop("model", {})
        model_config = LunarisCodexConfig(**model_config_dict)
        config_dict['model'] = model_config

        float_fields = ['learning_rate', 'weight_decay', 'beta1', 'beta2', 'grad_clip']
        int_fields = ['warmup_steps', 'max_steps', 'batch_size', 'gradient_accumulation_steps', 
                     'num_epochs', 'save_interval', 'log_interval', 'tpu_cores', 'dataloader_num_workers']

        for key in float_fields:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
        for key in int_fields:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])

        return cls(**config_dict)


# --- Sharded Memory-Mapped Dataset (TPU-optimized) ---
class ShardDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.shards:
            raise ValueError(f"No .npy files found in directory: {data_dir}")

        # For TPU, we want to minimize memory-mapped file access during training
        # Load all shards into memory if they fit, otherwise use memory mapping
        self.mmap_shards = []
        self.shard_lengths = []
        
        for shard_path in self.shards:
            try:
                # Try to load into memory first (better for TPU)
                shard_data = np.load(shard_path)
                self.mmap_shards.append(shard_data)
                self.shard_lengths.append(len(shard_data))
            except MemoryError:
                # Fall back to memory mapping if too large
                shard_data = np.load(shard_path, mmap_mode='r')
                self.mmap_shards.append(shard_data)
                self.shard_lengths.append(len(shard_data))

        self.cumulative_lengths = np.cumsum(self.shard_lengths)
        self.total_length = max(0, self.cumulative_lengths[-1] - self.sequence_length - 1)

        print(f"[DATA] Loaded {len(self.shards)} shards. Effective total samples: {self.total_length}. "
              f"Total tokens: {self.cumulative_lengths[-1] / 1e9:.2f}B.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which shard contains this index
        shard_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_idx = idx if shard_idx == 0 else idx - self.cumulative_lengths[shard_idx - 1]

        # Handle cross-shard sequences
        if local_idx + self.sequence_length + 1 > self.shard_lengths[shard_idx]:
            remaining_len = self.shard_lengths[shard_idx] - local_idx
            seq_part1 = self.mmap_shards[shard_idx][local_idx : local_idx + remaining_len]

            needed_from_next = self.sequence_length + 1 - remaining_len
            seq_part2 = self.mmap_shards[shard_idx + 1][:needed_from_next]

            seq = np.concatenate((seq_part1, seq_part2))
        else:
            seq = self.mmap_shards[shard_idx][local_idx : local_idx + self.sequence_length + 1]

        seq_tensor = torch.from_numpy(seq.astype(np.int64))
        x, y = seq_tensor[:-1], seq_tensor[1:]
        return x, y


# --- Learning Rate Scheduler ---
def get_lr(step, config: TrainConfig):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.learning_rate * 0.01

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)


# --- Robust checkpoint key unwrapping ---
def unwrap_model_keys(state_dict):
    """Remove XLA and other prefixes from model state dict keys."""
    unwrapped = {}
    prefixes_to_remove = ['_orig_mod.', 'module.']

    for k, v in state_dict.items():
        new_k = k
        for prefix in prefixes_to_remove:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        unwrapped[new_k] = v
    return unwrapped


# --- TPU Training Function ---
def train_on_tpu(rank, config: TrainConfig):
    """Main training function that runs on each TPU core."""
    
    # Get TPU device
    device = xm.xla_device()
    is_master_process = xm.is_master_ordinal()
    world_size = xm.xrt_world_size()
    
    # Set random seed for reproducibility
    torch.manual_seed(1337 + rank)
    
    if is_master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print("-" * 50)
        print(" " * 15 + "LUNARIS CODEX TPU TRAINING")
        print("-" * 50)
        print(f"Model: {config.model}")
        print(f"Data: {config.data_dir}")
        print(f"Batch size per core: {config.batch_size}")
        print(f"Total batch size: {config.batch_size * world_size}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max steps: {config.max_steps}")
        print(f"TPU cores: {world_size}")
        print(f"Mixed precision: {config.mixed_precision}")
        print("-" * 50)

    # Initialize W&B on master process
    if is_master_process and config.wandb_project:
        import wandb
        wandb.init(
            project=config.wandb_project, 
            entity=config.wandb_entity, 
            name=config.wandb_run_name,
            config=config.__dict__
        )

    # Load dataset
    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    
    # Create data sampler for TPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.dataloader_num_workers,
        pin_memory=False,  # TPU doesn't need pin_memory
        drop_last=True
    )

    # Create parallel loader for TPU
    para_loader = pl.ParallelLoader(train_loader, [device])

    # Initialize model
    model = LunarisCodex(config.model).to(device)

    if is_master_process:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[MODEL] Number of parameters: {num_params:.2f}M")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )

    # Initialize mixed precision scaler if enabled
    scaler = GradScaler() if config.mixed_precision else None

    # Training state
    current_step = 0
    current_epoch = 0
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        if is_master_process:
            print(f"[SETUP] Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint on CPU first, then move to TPU
        state = torch.load(checkpoint_path, map_location='cpu')
        unwrapped_state_dict = unwrap_model_keys(state['model'])
        model.load_state_dict(unwrapped_state_dict)
        optimizer.load_state_dict(state['optimizer'])
        current_step = state['step']
        current_epoch = state.get('epoch', 0)
        
        if scaler and 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
            
        if is_master_process:
            print(f"[SETUP] Resumed successfully. Starting from step {current_step}")

    # Zero gradients
    optimizer.zero_grad()

    # Set epoch for sampler
    train_sampler.set_epoch(current_epoch)

    # Training progress bar
    if is_master_process:
        print(f"\n[TRAIN] Starting training from step {current_step} up to {config.max_steps} steps...")
        pbar = tqdm(total=config.max_steps, desc="Training Steps", initial=current_step, ncols=120)

    # Training loop
    data_iter = iter(para_loader.per_device_loader(device))
    
    while current_step < config.max_steps:
        current_step += 1

        # Update learning rate
        lr = get_lr(current_step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        accumulated_loss = 0.0

        # Gradient accumulation loop
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                current_epoch += 1
                train_sampler.set_epoch(current_epoch)
                data_iter = iter(para_loader.per_device_loader(device))
                x, y = next(data_iter)

            # Forward pass with mixed precision
            if config.mixed_precision:
                with autocast():
                    logits, loss = model(x, y)
                    loss = loss / config.gradient_accumulation_steps
                
                accumulated_loss += loss.item()
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
            else:
                logits, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps
                accumulated_loss += loss.item()
                loss.backward()

        # Gradient clipping and optimizer step
        if config.mixed_precision:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # TPU-specific optimizer step
            xm.optimizer_step(optimizer, barrier=True)

        optimizer.zero_grad()

        # Progress bar update
        if is_master_process:
            pbar.update(1)

            # Logging
            if current_step % config.log_interval == 0:
                log_loss = accumulated_loss

                # Calculate perplexity
                if log_loss < 100:
                    try:
                        perplexity = math.exp(log_loss)
                    except (OverflowError, ValueError):
                        perplexity = float('inf')
                else:
                    perplexity = float('inf')

                current_lr = lr

                postfix_data = {
                    "loss": f"{log_loss:.3f}",
                    "ppl": f"{perplexity:.2f}" if perplexity != float('inf') else "inf",
                    "lr": f"{current_lr:.2e}",
                    "gnorm": f"{grad_norm.item():.2f}"
                }
                pbar.set_postfix(postfix_data)

                # W&B logging
                if config.wandb_project:
                    wandb.log({
                        "step": current_step,
                        "loss": log_loss,
                        "perplexity": perplexity,
                        "lr": current_lr,
                        "grad_norm": grad_norm.item(),
                        "epoch": current_epoch
                    })

            # Checkpointing
            if current_step > 0 and current_step % config.save_interval == 0:
                # Save checkpoint on master process
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config.__dict__,
                    'step': current_step,
                    'epoch': current_epoch,
                }
                
                if scaler:
                    checkpoint['scaler'] = scaler.state_dict()

                save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
                
                # Use XLA-aware saving
                xm.save(checkpoint, save_path, master_only=True)
                
                latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
                xm.save(checkpoint, latest_path, master_only=True)
                
                if is_master_process:
                    print(f"\n[CHECKPOINT] Saved checkpoint to {save_path}")

    # Cleanup
    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()


# --- Main Training Function ---
def train(config_path: str):
    """Main function that spawns TPU training processes."""
    config = TrainConfig.from_yaml(config_path)
    
    print(f"[SETUP] Starting TPU training with {config.tpu_cores} cores")
    print(f"[SETUP] Total global batch size: {config.batch_size * config.tpu_cores}")
    
    # Spawn training on TPU cores
    xmp.spawn(train_on_tpu, args=(config,), nprocs=config.tpu_cores, start_method='fork')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex model on TPU.")
    parser.add_argument("config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    
    # Verify TPU availability
    if not torch_xla._XLAC._xla_runtime_is_initialized():
        print("Warning: XLA runtime not initialized. Make sure you're running on a TPU.")
    
    train(args.config)
