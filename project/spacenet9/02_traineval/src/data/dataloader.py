#!/usr/bin/env python3
"""
DataLoader utilities for SpaceNet9
"""

from typing import Dict, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .spacenet9_dataset import (
    SpaceNet9Dataset, 
    get_train_transforms, 
    get_val_transforms,
    get_sar_train_transforms,
    get_sar_val_transforms,
    create_data_splits
)


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    task: str = 'opt_keypoint_detection',
    regions: Optional[list] = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_root: Root directory of the dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        task: 'opt_keypoint_detection' or 'sar_keypoint_detection'
        regions: List of regions to use (default: all available)
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    data_root = Path(data_root)
    
    # Get all regions if not specified
    if regions is None:
        regions = ['001', '002', '003']  # Default regions
    
    # Create data splits
    splits = create_data_splits(regions, val_ratio, test_ratio, seed)
    
    # Select transforms based on INPUT modality, not task
    # opt_keypoint_detection: input=SAR, so use SAR transforms
    # sar_keypoint_detection: input=optical, so use optical transforms
    if task == 'opt_keypoint_detection':
        # Input is SAR image
        train_transform = get_sar_train_transforms()
        val_transform = get_sar_val_transforms()
    else:
        # Input is optical image
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
    
    # Create datasets
    train_dataset = SpaceNet9Dataset(
        data_root=data_root,
        regions=splits['train'],
        task=task,
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = SpaceNet9Dataset(
        data_root=data_root,
        regions=splits['val'],
        task=task,
        transform=val_transform,
        mode='val'
    )
    
    # Only create test dataset if test regions exist
    if splits['test']:
        test_dataset = SpaceNet9Dataset(
            data_root=data_root,
            regions=splits['test'],
            task=task,
            transform=val_transform,
            mode='test'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
    }
    
    if test_loader is not None:
        dataloaders['test'] = test_loader
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"Test samples: {len(test_dataset)}")
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    else:
        print(f"Test samples: 0")
        total_samples = len(train_dataset) + len(val_dataset)
    print(f"Total samples: {total_samples}")
    
    return dataloaders


def create_single_dataloader(
    data_root: str,
    regions: list,
    batch_size: int = 16,
    num_workers: int = 4,
    task: str = 'opt_keypoint_detection',
    mode: str = 'val',
    shuffle: bool = False,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a single dataloader for specific regions
    
    Args:
        data_root: Root directory of the dataset
        regions: List of regions to include
        batch_size: Batch size
        num_workers: Number of workers for data loading
        task: 'opt_keypoint_detection' or 'sar_keypoint_detection'
        mode: 'train', 'val', or 'test'
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    data_root = Path(data_root)
    
    # Select transforms based on INPUT modality, not task
    # opt_keypoint_detection: input=SAR, so use SAR transforms
    # sar_keypoint_detection: input=optical, so use optical transforms
    if task == 'opt_keypoint_detection':
        # Input is SAR image
        if mode == 'train':
            transform = get_sar_train_transforms()
        else:
            transform = get_sar_val_transforms()
    else:
        # Input is optical image
        if mode == 'train':
            transform = get_train_transforms()
        else:
            transform = get_val_transforms()
    
    # Create dataset
    dataset = SpaceNet9Dataset(
        data_root=data_root,
        regions=regions,
        task=task,
        transform=transform,
        mode=mode
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == 'train')
    )
    
    print(f"\nCreated {mode} dataloader with {len(dataset)} samples")
    
    return dataloader


if __name__ == "__main__":
    # Test dataloader creation
    data_root = "/workspace/project/spacenet9/00_data"
    
    # Create all dataloaders
    dataloaders = create_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=0,  # Set to 0 for testing
        regions=['001'],  # Use only one region for testing
        val_ratio=0.3,
        test_ratio=0.0
    )
    
    # Test loading a batch
    for batch in dataloaders['train']:
        print(f"\nBatch loaded successfully!")
        print(f"Input shape: {batch['input'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        break