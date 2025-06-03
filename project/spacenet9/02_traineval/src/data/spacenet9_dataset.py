#!/usr/bin/env python3
"""
SpaceNet9 Dataset class for PyTorch
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SpaceNet9Dataset(Dataset):
    """
    PyTorch Dataset for SpaceNet9 keypoint detection
    
    Args:
        data_root: Root directory containing the dataset
        regions: List of region names (e.g., ['001', '002'])
        task: 'opt_keypoint_detection' or 'sar_keypoint_detection'
        transform: Albumentations transform pipeline
        mode: 'train', 'val', or 'test'
        heatmap_scale: Scale factor for heatmap (default: 1.0)
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        regions: List[str],
        task: str = 'opt_keypoint_detection',
        transform: Optional[Callable] = None,
        mode: str = 'train',
        heatmap_scale: float = 1.0
    ):
        self.data_root = Path(data_root)
        self.regions = regions
        self.task = task
        self.transform = transform
        self.mode = mode
        self.heatmap_scale = heatmap_scale
        
        # Validate task
        if task not in ['opt_keypoint_detection', 'sar_keypoint_detection']:
            raise ValueError(f"Invalid task: {task}")
        
        # Collect all tiles
        self.tiles = []
        self._collect_tiles()
        
        if len(self.tiles) == 0:
            raise ValueError(f"No tiles found for regions {regions} and task {task}")
        
        print(f"Loaded {len(self.tiles)} tiles for {mode} mode")
    
    def _collect_tiles(self):
        """Collect all available tiles from the specified regions"""
        task_dir = self.data_root / 'dataset' / self.task
        
        for region in self.regions:
            region_dir = task_dir / region
            if not region_dir.exists():
                print(f"Warning: Region directory {region_dir} does not exist")
                continue
            
            # Load dataset info
            info_path = region_dir / 'dataset_info.json'
            if not info_path.exists():
                print(f"Warning: Dataset info not found at {info_path}")
                continue
            
            with open(info_path, 'r') as f:
                dataset_info = json.load(f)
            
            # Add tiles from this region
            for tile_info in dataset_info['tiles']:
                tile_data = {
                    'region': region,
                    'region_dir': region_dir,
                    'index': tile_info['index'],
                    'optical_tile': region_dir / tile_info['optical_tile'],
                    'sar_tile': region_dir / tile_info['sar_tile'],
                    'heatmap': region_dir / tile_info['heatmap'],
                    'metadata': tile_info
                }
                self.tiles.append(tile_data)
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from GeoTIFF file"""
        with rasterio.open(path) as src:
            if src.count >= 3:
                # RGB image
                image = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
            else:
                # Grayscale image (SAR)
                image = src.read(1)
                # Convert to 3-channel for consistency
                image = np.stack([image, image, image], axis=-1)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            # Already float, ensure in [0, 1]
            image = image.astype(np.float32)
            image = np.clip(image, 0, 1)
        
        # For SAR images, apply percentile-based normalization for better dynamic range
        if 'sar' in str(path).lower():
            # Use percentile normalization to enhance contrast for all SAR images
            p1, p99 = np.percentile(image, [1, 99])
            if p99 > p1:
                image = (image - p1) / (p99 - p1)
                image = np.clip(image, 0, 1)
            
            # Additional contrast enhancement for very low dynamic range images
            if image.max() < 0.1:
                # Apply histogram equalization-like stretching
                image = image / (image.max() + 1e-8)
                # Apply gamma correction to enhance mid-tones
                image = np.power(image, 0.7)
        
        return image
    
    def _load_heatmap(self, path: Path) -> np.ndarray:
        """Load heatmap from GeoTIFF file"""
        with rasterio.open(path) as src:
            heatmap = src.read(1).astype(np.float32)
        
        # Apply scaling if needed
        if self.heatmap_scale != 1.0:
            heatmap = heatmap * self.heatmap_scale
        
        return heatmap
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        tile_data = self.tiles[idx]
        
        # Load images and heatmap
        optical_image = self._load_image(tile_data['optical_tile'])
        sar_image = self._load_image(tile_data['sar_tile'])
        heatmap = self._load_heatmap(tile_data['heatmap'])
        
        # Apply transforms
        if self.transform is not None:
            # Apply transform to the input modality and heatmap
            if self.task == 'opt_keypoint_detection':
                # Input is SAR image, apply SAR transforms
                transformed = self.transform(image=sar_image, mask=heatmap)
                sar_image = transformed['image']
                heatmap = transformed['mask']
                
                # Manually normalize optical image for visualization (ImageNet stats)
                optical_image = (optical_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                optical_image = torch.from_numpy(optical_image.transpose(2, 0, 1)).float()
            else:
                # Input is optical image, apply optical transforms
                transformed = self.transform(image=optical_image, mask=heatmap)
                optical_image = transformed['image']
                heatmap = transformed['mask']
                
                # Manually normalize SAR image for visualization (simple 0-1)
                sar_image = torch.from_numpy(sar_image.transpose(2, 0, 1)).float()
            
            # Ensure heatmap is proper tensor
            if isinstance(heatmap, np.ndarray):
                heatmap = torch.from_numpy(heatmap).float()
            if heatmap.dim() == 2:
                heatmap = heatmap.unsqueeze(0)
        else:
            # Default transform to tensor
            optical_image = torch.from_numpy(optical_image.transpose(2, 0, 1)).float()
            sar_image = torch.from_numpy(sar_image.transpose(2, 0, 1)).float()
            heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()
        
        # Prepare output based on task
        if self.task == 'opt_keypoint_detection':
            # Input: SAR, Target: optical keypoint heatmap
            return {
                'input': sar_image,
                'target': heatmap,
                'optical': optical_image,  # For visualization
                'metadata': tile_data['metadata']
            }
        else:
            # Input: optical, Target: SAR keypoint heatmap
            return {
                'input': optical_image,
                'target': heatmap,
                'sar': sar_image,  # For visualization
                'metadata': tile_data['metadata']
            }


def get_train_transforms(size: int = 256) -> A.Compose:
    """Get training augmentation pipeline"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=0.1, 
            rotate_limit=15, 
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=1.0
            ),
            A.RandomGamma(p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(size: int = 256) -> A.Compose:
    """Get validation augmentation pipeline"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_sar_train_transforms(size: int = 256) -> A.Compose:
    """Get SAR training augmentation pipeline"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=0.1, 
            rotate_limit=15, 
            p=0.5
        ),
        # SAR-specific augmentations (less aggressive on brightness/contrast)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        # Simple 0-1 normalization for SAR
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2()
    ])


def get_sar_val_transforms(size: int = 256) -> A.Compose:
    """Get SAR validation augmentation pipeline"""
    return A.Compose([
        # Simple 0-1 normalization for SAR
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2()
    ])


def create_data_splits(
    regions: List[str], 
    val_ratio: float = 0.2, 
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create train/val/test splits from regions
    
    Args:
        regions: List of all regions
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    np.random.seed(seed)
    regions = np.array(regions)
    np.random.shuffle(regions)
    
    n_total = len(regions)
    
    # For small number of regions, ensure at least one region per split
    if n_total >= 3:
        n_test = max(1, int(n_total * test_ratio)) if test_ratio > 0 else 0
        n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
        n_train = n_total - n_test - n_val
        
        # Ensure we have at least one training region
        if n_train <= 0:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
    else:
        # For very small datasets, put all in train unless specified otherwise
        n_test = 0
        n_val = 0
        n_train = n_total
    
    splits = {
        'train': regions[:n_train].tolist(),
        'val': regions[n_train:n_train+n_val].tolist() if n_val > 0 else [],
        'test': regions[n_train+n_val:].tolist() if n_test > 0 else []
    }
    
    return splits


if __name__ == "__main__":
    # Test the dataset
    data_root = Path("/workspace/project/spacenet9/00_data")
    
    # Create dataset
    dataset = SpaceNet9Dataset(
        data_root=data_root,
        regions=['001'],
        task='opt_keypoint_detection',
        transform=get_val_transforms(),
        mode='train'
    )
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input shape: {sample['input'].shape}")
        print(f"Target shape: {sample['target'].shape}")
        print(f"Metadata: {sample['metadata']}")