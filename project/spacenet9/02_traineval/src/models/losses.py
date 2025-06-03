#!/usr/bin/env python3
"""
Loss functions for keypoint detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for heatmap regression
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmap (B, 1, H, W)
            target: Target heatmap (B, 1, H, W)
        
        Returns:
            Loss value
        """
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance in keypoint detection
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmap (B, 1, H, W) - should be sigmoid activated
            target: Target heatmap (B, 1, H, W)
        
        Returns:
            Loss value
        """
        # Apply sigmoid if not already applied
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Compute focal loss
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        focal_weight = (1 - pt).pow(self.gamma)
        focal_loss = -alpha_t * focal_weight * pt.log()
        
        # Handle reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive focal loss that adjusts based on the number of positive samples
    """
    
    def __init__(
        self, 
        alpha_range: tuple = (0.1, 0.5),
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha_range = alpha_range
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmap (B, 1, H, W)
            target: Target heatmap (B, 1, H, W)
        
        Returns:
            Loss value
        """
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Calculate positive ratio
        pos_ratio = (target > 0.5).float().mean()
        
        # Adaptive alpha based on positive ratio
        alpha = self.alpha_range[0] + (self.alpha_range[1] - self.alpha_range[0]) * (1 - pos_ratio)
        
        # Compute focal loss with adaptive alpha
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        
        focal_weight = (1 - pt).pow(self.gamma)
        focal_loss = -alpha_t * focal_weight * pt.log()
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice loss for handling class imbalance in segmentation-like tasks
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmap (B, 1, H, W) - should be sigmoid activated
            target: Target heatmap (B, 1, H, W)
        
        Returns:
            Loss value
        """
        # Apply sigmoid if not already applied
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class BinaryDiceLoss(nn.Module):
    """
    Binary Dice loss with threshold
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmap (B, 1, H, W)
            target: Target heatmap (B, 1, H, W)
        
        Returns:
            Loss value
        """
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Binarize predictions and targets
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(pred_binary.size(0), -1)
        target_flat = target_binary.view(target_binary.size(0), -1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 loss (Huber loss) for robust regression
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmap (B, 1, H, W)
            target: Target heatmap (B, 1, H, W)
        
        Returns:
            Loss value
        """
        loss = F.smooth_l1_loss(pred, target, beta=self.beta, reduction=self.reduction)
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components
    """
    
    def __init__(
        self,
        losses: dict,
        weights: Optional[dict] = None
    ):
        """
        Args:
            losses: Dictionary of loss functions
            weights: Dictionary of loss weights (default: equal weights)
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        
        if weights is None:
            weights = {name: 1.0 for name in losses.keys()}
        self.weights = weights
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Args:
            pred: Predicted heatmap
            target: Target heatmap
        
        Returns:
            Dictionary with individual losses and total loss
        """
        loss_dict = {}
        total_loss = 0
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            loss_dict[name] = loss_value
            total_loss += self.weights.get(name, 1.0) * loss_value
        
        loss_dict['total'] = total_loss
        return loss_dict


class KeypointDetectionLoss(nn.Module):
    """
    Main loss function for keypoint detection combining multiple objectives
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        use_focal: bool = False,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_smoothl1: bool = False,
        smoothl1_weight: float = 0.3,
        smoothl1_beta: float = 0.5,
        use_dice: bool = False,
        dice_weight: float = 0.3,
        dice_smooth: float = 1e-6,
        use_binary_dice: bool = False,
        binary_dice_weight: float = 0.2,
        binary_dice_threshold: float = 0.5
    ):
        super().__init__()
        
        losses = {}
        weights = {}
        
        # Main loss
        if loss_type == 'mse':
            losses['mse'] = MSELoss()
            weights['mse'] = 1.0
        elif loss_type == 'smoothl1':
            losses['smoothl1'] = SmoothL1Loss(beta=smoothl1_beta)
            weights['smoothl1'] = 1.0
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Additional losses
        if use_focal:
            losses['focal'] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            weights['focal'] = focal_weight
            
        if use_smoothl1 and loss_type != 'smoothl1':
            losses['smoothl1_aux'] = SmoothL1Loss(beta=smoothl1_beta)
            weights['smoothl1_aux'] = smoothl1_weight
            
        if use_dice:
            losses['dice'] = DiceLoss(smooth=dice_smooth)
            weights['dice'] = dice_weight
            
        if use_binary_dice:
            losses['binary_dice'] = BinaryDiceLoss(
                threshold=binary_dice_threshold, 
                smooth=dice_smooth
            )
            weights['binary_dice'] = binary_dice_weight
        
        self.combined_loss = CombinedLoss(losses, weights)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Args:
            pred: Predicted heatmap
            target: Target heatmap
        
        Returns:
            Dictionary with loss values
        """
        return self.combined_loss(pred, target)


def create_loss(config: dict) -> nn.Module:
    """
    Factory function to create loss based on configuration
    
    Args:
        config: Loss configuration dictionary
    
    Returns:
        Loss module
    """
    loss_type = config.get('type', 'mse')
    
    if loss_type == 'keypoint_detection':
        return KeypointDetectionLoss(**config.get('params', {}))
    elif loss_type == 'mse':
        return MSELoss(**config.get('params', {}))
    elif loss_type == 'focal':
        return FocalLoss(**config.get('params', {}))
    elif loss_type == 'smoothl1':
        return SmoothL1Loss(**config.get('params', {}))
    elif loss_type == 'dice':
        return DiceLoss(**config.get('params', {}))
    elif loss_type == 'binary_dice':
        return BinaryDiceLoss(**config.get('params', {}))
    elif loss_type == 'combined':
        # Create individual losses
        losses = {}
        weights = {}
        
        for loss_config in config.get('losses', []):
            name = loss_config['name']
            loss_fn = create_loss(loss_config)
            losses[name] = loss_fn
            weights[name] = loss_config.get('weight', 1.0)
        
        return CombinedLoss(losses, weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test losses
    pred = torch.randn(2, 1, 256, 256)
    target = torch.zeros(2, 1, 256, 256)
    target[:, :, 100:110, 100:110] = 1.0  # Add some positive samples
    
    # Test individual losses
    mse_loss = MSELoss()
    print(f"MSE Loss: {mse_loss(pred, target):.4f}")
    
    focal_loss = FocalLoss()
    print(f"Focal Loss: {focal_loss(torch.sigmoid(pred), target):.4f}")
    
    # Test combined loss
    keypoint_loss = KeypointDetectionLoss(
        loss_type='mse',
        use_focal=True,
        focal_weight=0.5
    )
    
    loss_dict = keypoint_loss(pred, target)
    print("\nCombined loss:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.4f}")