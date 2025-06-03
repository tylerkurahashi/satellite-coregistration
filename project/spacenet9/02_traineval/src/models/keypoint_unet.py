#!/usr/bin/env python3
"""
U-Net based keypoint detection model for SpaceNet9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, List, Dict, Any


class KeypointUNet(nn.Module):
    """
    U-Net model for keypoint detection
    
    Args:
        encoder_name: Encoder backbone name from timm
        encoder_weights: Pretrained weights to use
        in_channels: Number of input channels
        classes: Number of output classes (1 for heatmap)
        activation: Activation function for output
        decoder_attention_type: Attention type for decoder
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        decoder_attention_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            decoder_attention_type=decoder_attention_type,
            **kwargs
        )
        
        # Store configuration
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def get_params_groups(self) -> Dict[str, Any]:
        """Get parameter groups for different learning rates"""
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params
        }


class KeypointUNetPlusPlus(nn.Module):
    """
    U-Net++ model for keypoint detection
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        decoder_attention_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            decoder_attention_type=decoder_attention_type,
            **kwargs
        )
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)


class KeypointFPN(nn.Module):
    """
    Feature Pyramid Network for keypoint detection
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)


def create_model(
    model_name: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create keypoint detection models
    
    Args:
        model_name: Model architecture ('unet', 'unetplusplus', 'fpn', 'fpn_attention', 'fpn_v2')
        encoder_name: Encoder backbone name
        encoder_weights: Pretrained weights
        in_channels: Number of input channels
        classes: Number of output classes
        **kwargs: Additional model parameters
    
    Returns:
        Model instance
    """
    # Import FPN models to avoid circular imports
    from .keypoint_fpn import KeypointFPNWithAttention, KeypointFPNv2
    
    models = {
        'unet': KeypointUNet,
        'unetplusplus': KeypointUNetPlusPlus,
        'fpn': KeypointFPN,
        'fpn_attention': KeypointFPNWithAttention,
        'fpn_v2': KeypointFPNv2
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(models.keys())}")
    
    model_class = models[model_name]
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs
    )


class HeatmapPostProcessor:
    """
    Post-processing for heatmap to extract keypoints
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        nms_radius: int = 3,
        max_keypoints: int = 100
    ):
        self.threshold = threshold
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
    
    def extract_keypoints(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Extract keypoint locations from heatmap
        
        Args:
            heatmap: Heatmap tensor of shape (B, 1, H, W)
        
        Returns:
            Keypoints tensor of shape (B, N, 2) where N is number of keypoints
        """
        batch_size = heatmap.shape[0]
        keypoints_batch = []
        
        for b in range(batch_size):
            # Apply NMS using max pooling
            pooled = F.max_pool2d(
                heatmap[b:b+1], 
                kernel_size=2*self.nms_radius+1, 
                stride=1, 
                padding=self.nms_radius
            )
            
            # Find peaks
            peaks = (heatmap[b:b+1] == pooled) & (heatmap[b:b+1] > self.threshold)
            
            # Get coordinates
            y_coords, x_coords = torch.where(peaks[0, 0])
            scores = heatmap[b, 0, y_coords, x_coords]
            
            # Sort by score and keep top k
            if len(scores) > self.max_keypoints:
                top_k_indices = torch.topk(scores, self.max_keypoints).indices
                y_coords = y_coords[top_k_indices]
                x_coords = x_coords[top_k_indices]
            
            # Stack coordinates
            if len(y_coords) > 0:
                keypoints = torch.stack([x_coords.float(), y_coords.float()], dim=1)
            else:
                keypoints = torch.zeros((0, 2), device=heatmap.device)
            
            keypoints_batch.append(keypoints)
        
        return keypoints_batch
    
    def refine_keypoints(self, heatmap: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Refine keypoint locations to sub-pixel accuracy
        
        Args:
            heatmap: Original heatmap
            keypoints: Coarse keypoint locations
        
        Returns:
            Refined keypoint locations
        """
        # Simple quadratic interpolation for sub-pixel refinement
        refined_keypoints = keypoints.clone()
        
        h, w = heatmap.shape[-2:]
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp[0]), int(kp[1])
            
            if 1 <= x < w-1 and 1 <= y < h-1:
                # Get 3x3 neighborhood
                patch = heatmap[0, y-1:y+2, x-1:x+2]
                
                # Check if patch is valid size (3x3)
                if patch.shape[0] == 3 and patch.shape[1] == 3:
                    # Compute gradients
                    dx = (patch[1, 2] - patch[1, 0]) / 2
                    dy = (patch[2, 1] - patch[0, 1]) / 2
                    
                    # Compute Hessian
                    dxx = patch[1, 2] - 2*patch[1, 1] + patch[1, 0]
                    dyy = patch[2, 1] - 2*patch[1, 1] + patch[0, 1]
                    dxy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4
                    
                    # Solve for offset
                    det = dxx * dyy - dxy * dxy
                    if abs(det) > 1e-6:
                        offset_x = -(dyy * dx - dxy * dy) / det
                        offset_y = -(dxx * dy - dxy * dx) / det
                        
                        # Limit offset magnitude
                        offset_x = torch.clamp(offset_x, -0.5, 0.5)
                        offset_y = torch.clamp(offset_y, -0.5, 0.5)
                        
                        refined_keypoints[i, 0] += offset_x
                        refined_keypoints[i, 1] += offset_y
        
        return refined_keypoints


class AdaptiveHeatmapPostProcessor(HeatmapPostProcessor):
    """
    Adaptive post-processor with dynamic thresholding and advanced NMS
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        nms_radius: int = 3,
        max_keypoints: int = 100,
        adaptive_threshold: bool = True,
        percentile_threshold: float = 95.0,
        min_threshold: float = 0.1,
        max_threshold: float = 0.8
    ):
        super().__init__(threshold, nms_radius, max_keypoints)
        self.adaptive_threshold = adaptive_threshold
        self.percentile_threshold = percentile_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def _calculate_adaptive_threshold(self, heatmap: torch.Tensor) -> float:
        """
        Calculate adaptive threshold based on heatmap statistics
        
        Args:
            heatmap: Single heatmap (1, H, W)
            
        Returns:
            Adaptive threshold value
        """
        if not self.adaptive_threshold:
            return self.threshold
        
        # Flatten heatmap and remove zeros/very low values
        flat_values = heatmap.flatten()
        valid_values = flat_values[flat_values > 0.01]
        
        if len(valid_values) == 0:
            return self.threshold
        
        # Use percentile-based threshold
        threshold = torch.quantile(valid_values, self.percentile_threshold / 100.0).item()
        
        # Clamp to reasonable range
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))
        
        return threshold
    
    def extract_keypoints(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Extract keypoints with adaptive thresholding
        
        Args:
            heatmap: Heatmap tensor of shape (B, 1, H, W)
        
        Returns:
            List of keypoints tensors for each batch item
        """
        batch_size = heatmap.shape[0]
        keypoints_batch = []
        
        for b in range(batch_size):
            current_heatmap = heatmap[b:b+1]
            
            # Calculate adaptive threshold for this heatmap
            if self.adaptive_threshold:
                threshold = self._calculate_adaptive_threshold(current_heatmap[0])
            else:
                threshold = self.threshold
            
            # Apply Gaussian blur for smoother peaks (skip if not available)
            try:
                smoothed = F.gaussian_blur(
                    current_heatmap, 
                    kernel_size=[3, 3], 
                    sigma=[1.0, 1.0]
                )
            except AttributeError:
                # Fallback if gaussian_blur is not available
                smoothed = current_heatmap
            
            # Apply NMS using max pooling
            pooled = F.max_pool2d(
                smoothed, 
                kernel_size=2*self.nms_radius+1, 
                stride=1, 
                padding=self.nms_radius
            )
            
            # Find peaks
            peaks = (smoothed == pooled) & (smoothed > threshold)
            
            # Get coordinates
            y_coords, x_coords = torch.where(peaks[0, 0])
            scores = current_heatmap[0, 0, y_coords, x_coords]
            
            # Sort by score and keep top k
            if len(scores) > self.max_keypoints:
                top_k_indices = torch.topk(scores, self.max_keypoints).indices
                y_coords = y_coords[top_k_indices]
                x_coords = x_coords[top_k_indices]
                scores = scores[top_k_indices]
            
            # Stack coordinates with scores
            if len(y_coords) > 0:
                keypoints = torch.stack([
                    x_coords.float(), 
                    y_coords.float(), 
                    scores.float()
                ], dim=1)
            else:
                keypoints = torch.zeros((0, 3), device=heatmap.device)
            
            keypoints_batch.append(keypoints)
        
        return keypoints_batch


class MultiScalePostProcessor:
    """
    Multi-scale post-processing for better keypoint detection
    """
    
    def __init__(
        self,
        scales: List[float] = [0.5, 1.0, 2.0],
        threshold: float = 0.3,
        nms_radius: int = 5,
        max_keypoints: int = 50
    ):
        self.scales = scales
        self.threshold = threshold
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        
        self.processors = [
            AdaptiveHeatmapPostProcessor(
                threshold=threshold,
                nms_radius=nms_radius,
                max_keypoints=max_keypoints * 2,  # More for merging
                adaptive_threshold=True,
                percentile_threshold=90.0
            ) for _ in scales
        ]
    
    def extract_keypoints(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Extract keypoints at multiple scales and merge
        
        Args:
            heatmap: Heatmap tensor of shape (B, 1, H, W)
        
        Returns:
            List of merged keypoints for each batch item
        """
        batch_size = heatmap.shape[0]
        all_keypoints = []
        
        for b in range(batch_size):
            batch_keypoints = []
            
            # Extract keypoints at different scales
            for scale, processor in zip(self.scales, self.processors):
                if scale != 1.0:
                    # Resize heatmap
                    original_size = heatmap.shape[-2:]
                    new_size = [int(s * scale) for s in original_size]
                    scaled_heatmap = F.interpolate(
                        heatmap[b:b+1], 
                        size=new_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # Extract keypoints
                    keypoints = processor.extract_keypoints(scaled_heatmap)[0]
                    
                    # Scale coordinates back
                    if len(keypoints) > 0:
                        keypoints[:, :2] /= scale
                else:
                    # Original scale
                    keypoints = processor.extract_keypoints(heatmap[b:b+1])[0]
                
                if len(keypoints) > 0:
                    batch_keypoints.append(keypoints)
            
            # Merge keypoints from different scales
            if batch_keypoints:
                merged_keypoints = self._merge_keypoints(batch_keypoints)
            else:
                merged_keypoints = torch.zeros((0, 3), device=heatmap.device)
            
            all_keypoints.append(merged_keypoints)
        
        return all_keypoints
    
    def _merge_keypoints(self, keypoint_lists: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge keypoints from different scales using NMS
        
        Args:
            keypoint_lists: List of keypoint tensors
            
        Returns:
            Merged keypoints tensor
        """
        if not keypoint_lists:
            return torch.zeros((0, 3))
        
        # Concatenate all keypoints
        all_keypoints = torch.cat(keypoint_lists, dim=0)
        
        if len(all_keypoints) == 0:
            return torch.zeros((0, 3))
        
        # Apply NMS
        keep_indices = []
        coords = all_keypoints[:, :2]
        scores = all_keypoints[:, 2]
        
        # Sort by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        
        for i in sorted_indices:
            if len(keep_indices) >= self.max_keypoints:
                break
                
            # Check distance to already kept keypoints
            if len(keep_indices) == 0:
                keep_indices.append(i)
            else:
                kept_coords = coords[keep_indices]
                current_coord = coords[i:i+1]
                
                # Calculate distances
                distances = torch.norm(kept_coords - current_coord, dim=1)
                
                # Keep if far enough from existing keypoints
                if torch.min(distances) > self.nms_radius:
                    keep_indices.append(i)
        
        if keep_indices:
            return all_keypoints[keep_indices]
        else:
            return torch.zeros((0, 3), device=all_keypoints.device)


if __name__ == "__main__":
    # Test model creation
    model = create_model(
        model_name="unet",
        encoder_name="resnet34",
        encoder_weights="imagenet"
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test keypoint extraction
    post_processor = HeatmapPostProcessor()
    heatmap = torch.sigmoid(y)  # Convert to probabilities
    keypoints = post_processor.extract_keypoints(heatmap)
    print(f"Extracted keypoints for batch 0: {len(keypoints[0])} points")