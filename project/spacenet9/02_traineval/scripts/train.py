#!/usr/bin/env python3
"""
Training script for SpaceNet9 keypoint detection
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import create_dataloaders
from src.models import create_model, create_loss
from src.utils.metrics import calculate_metrics
from src.utils.checkpoint import CheckpointManager
from src.utils.early_stopping import EarlyStopping
from src.utils.visualization import visualize_predictions


class Trainer:
    """
    Trainer class for SpaceNet9 models
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Create experiment directory
        self.exp_dir = Path(config['exp_dir']) / config['exp_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_logging()
        
    def _setup_data(self):
        """Setup data loaders"""
        self.dataloaders = create_dataloaders(
            data_root=self.config['data']['data_root'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task=self.config['data']['task'],
            regions=self.config['data'].get('regions'),
            val_ratio=self.config['data'].get('val_ratio', 0.2),
            test_ratio=self.config['data'].get('test_ratio', 0.1),
            seed=self.config.get('seed', 42)
        )
        
    def _setup_model(self):
        """Setup model and loss"""
        # Create model
        self.model = create_model(
            model_name=self.config['model']['name'],
            encoder_name=self.config['model']['encoder'],
            encoder_weights=self.config['model'].get('encoder_weights', 'imagenet'),
            in_channels=self.config['model'].get('in_channels', 3),
            classes=self.config['model'].get('classes', 1)
        )
        self.model.to(self.device)
        
        # Create loss
        self.criterion = create_loss(self.config['loss'])
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def _setup_training(self):
        """Setup training components"""
        # Optimizer
        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
        
        # Learning rate scheduler
        scheduler_config = self.config['scheduler']
        if scheduler_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config.get('eta_min', 0)
            )
        elif scheduler_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config['type'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            self.scheduler = None
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.exp_dir / 'checkpoints',
            max_checkpoints=self.config['training'].get('max_checkpoints', 5)
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['training'].get('early_stopping_patience', 20),
            min_delta=self.config['training'].get('early_stopping_delta', 1e-4)
        )
        
    def _setup_logging(self):
        """Setup logging"""
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'spacenet9'),
                name=self.config['exp_name'],
                config=self.config
            )
            wandb.watch(self.model)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_metrics = {}
        
        pbar = tqdm(self.dataloaders['train'], desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            if isinstance(self.criterion, nn.Module):
                loss = self.criterion(outputs, targets)
                if isinstance(loss, dict):
                    loss_value = loss['total']
                else:
                    loss_value = loss
            else:
                loss_value = self.criterion(outputs, targets)
            
            # Backward pass
            loss_value.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss_value.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_value.item():.4f}'})
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss_value.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Calculate epoch metrics
        epoch_metrics['loss'] = epoch_loss / len(self.dataloaders['train'])
        
        return epoch_metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        val_loss = 0
        val_metrics = {}
        
        with torch.no_grad():
            pbar = tqdm(self.dataloaders['val'], desc=f'Epoch {epoch+1} [Val]')
            
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                if isinstance(self.criterion, nn.Module):
                    loss = self.criterion(outputs, targets)
                    if isinstance(loss, dict):
                        loss_value = loss['total']
                    else:
                        loss_value = loss
                else:
                    loss_value = self.criterion(outputs, targets)
                
                val_loss += loss_value.item()
                
                # Visualize first batch
                if batch_idx == 0 and epoch % self.config['training'].get('vis_interval', 10) == 0:
                    vis_path = self.exp_dir / 'visualizations' / f'epoch_{epoch+1}.png'
                    vis_path.parent.mkdir(exist_ok=True)
                    
                    # Visualize predictions
                    # TODO: Implement visualization function
                    
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_value.item():.4f}'})
        
        # Calculate validation metrics
        val_metrics['loss'] = val_loss / len(self.dataloaders['val'])
        
        return val_metrics
    
    def train(self):
        """Main training loop"""
        start_epoch = 0
        best_val_loss = float('inf')
        
        # Resume from checkpoint if specified
        if self.config['training'].get('resume'):
            checkpoint = self.checkpoint_manager.load_checkpoint(
                self.config['training']['resume'],
                self.model,
                self.optimizer,
                self.scheduler
            )
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'val/epoch_loss': val_metrics['loss'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
            
            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics={
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'best_val_loss': best_val_loss
                },
                is_best=is_best
            )
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final model
        torch.save(
            self.model.state_dict(),
            self.exp_dir / 'final_model.pth'
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train SpaceNet9 model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        help='Experiment name (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.exp_name:
        config['exp_name'] = args.exp_name
    config['device'] = args.device
    
    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()