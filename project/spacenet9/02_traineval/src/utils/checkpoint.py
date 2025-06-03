#!/usr/bin/env python3
"""
Checkpoint management utilities
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        keep_best: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            keep_best: Always keep the best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.checkpoints = []
        self.best_checkpoint = None
        
        # Load existing checkpoints
        self._scan_checkpoints()
    
    def _scan_checkpoints(self):
        """Scan directory for existing checkpoints"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        # Sort by epoch number
        def get_epoch(path):
            return int(path.stem.split('_')[-1])
        
        self.checkpoints = sorted(checkpoint_files, key=get_epoch)
        
        # Find best checkpoint
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        if best_path.exists():
            self.best_checkpoint = best_path
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save checkpoint
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Optional scheduler state
            metrics: Optional metrics dictionary
            is_best: Whether this is the best checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        # Save best checkpoint
        if is_best and self.keep_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            shutil.copy(checkpoint_path, best_path)
            self.best_checkpoint = best_path
            print(f"Saved best checkpoint at epoch {epoch}")
        
        # Remove old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by epoch
            def get_epoch(path):
                return int(path.stem.split('_')[-1])
            
            self.checkpoints.sort(key=get_epoch)
            
            # Remove oldest checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    print(f"Removed old checkpoint: {old_checkpoint.name}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint (if None, loads best)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint to
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            if self.best_checkpoint is not None and self.best_checkpoint.exists():
                checkpoint_path = self.best_checkpoint
            else:
                raise ValueError("No checkpoint path provided and no best checkpoint found")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        if not self.checkpoints:
            return None
        
        def get_epoch(path):
            return int(path.stem.split('_')[-1])
        
        return max(self.checkpoints, key=get_epoch)
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        return self.best_checkpoint


def save_model_only(
    model: nn.Module,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save only model weights
    
    Args:
        model: Model to save
        save_path: Path to save model
        metadata: Optional metadata to include
    """
    save_dict = {
        'model_state_dict': model.state_dict()
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, save_path)
    print(f"Saved model to {save_path}")


def load_model_only(
    model: nn.Module,
    load_path: Path,
    device: str = 'cpu',
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load only model weights
    
    Args:
        model: Model to load weights into
        load_path: Path to saved model
        device: Device to load to
        strict: Whether to strictly enforce parameter names match
    
    Returns:
        Loaded dictionary
    """
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded model from {load_path}")
    
    return checkpoint


if __name__ == "__main__":
    # Test checkpoint manager
    import tempfile
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        
        # Create checkpoint manager
        manager = CheckpointManager(checkpoint_dir, max_checkpoints=3)
        
        # Create dummy model and optimizer
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters())
        
        # Save some checkpoints
        for epoch in range(5):
            metrics = {'loss': 1.0 / (epoch + 1)}
            is_best = epoch == 2  # Epoch 2 is best
            
            manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                is_best=is_best
            )
        
        # Check remaining checkpoints
        print(f"\nRemaining checkpoints: {len(manager.checkpoints)}")
        for cp in manager.checkpoints:
            print(f"  {cp.name}")
        
        # Load best checkpoint
        best_path = manager.get_best_checkpoint()
        if best_path:
            checkpoint = manager.load_checkpoint(best_path)
            print(f"\nBest checkpoint from epoch: {checkpoint['epoch']}")
            print(f"Best metrics: {checkpoint.get('metrics', {})}")