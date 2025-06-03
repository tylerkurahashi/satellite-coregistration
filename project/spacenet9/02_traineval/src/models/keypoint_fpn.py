#!/usr/bin/env python3
"""
Feature Pyramid Network (FPN) based keypoint detection model for SpaceNet9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, List, Dict, Any, Tuple

from .attention_modules import CBAM, MultiScaleAttention, PyramidPoolingModule


class KeypointFPNWithAttention(nn.Module):
    """
    FPN-based keypoint detection model with attention mechanisms
    
    Args:
        encoder_name: Encoder backbone name
        encoder_weights: Pretrained weights
        in_channels: Number of input channels
        pyramid_channels: Number of channels in FPN
        segmentation_channels: Number of channels in segmentation head
        classes: Number of output classes
        attention_type: Type of attention ('cbam', 'multi_scale', 'both')
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        classes: int = 1,
        attention_type: str = "both",
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        # Create FPN model
        self.fpn = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            pyramid_channels=pyramid_channels,
            segmentation_channels=segmentation_channels,
            classes=classes,
            dropout=dropout,
            activation=None,  # We'll add our own activation
            **kwargs
        )
        
        # Store configuration
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        self.attention_type = attention_type
        
        # Get encoder channels for attention modules
        encoder_channels = self.fpn.encoder.out_channels
        
        # Add attention modules to encoder blocks
        if attention_type in ["cbam", "both"]:
            self.encoder_attention = nn.ModuleList()
            for i, ch in enumerate(encoder_channels):
                if ch > 0 and i > 0:  # Skip first channel (usually 0 or very small)
                    self.encoder_attention.append(CBAM(ch))
                else:
                    self.encoder_attention.append(nn.Identity())
        else:
            self.encoder_attention = None
            
        # Multi-scale attention for FPN features
        if attention_type in ["multi_scale", "both"]:
            # FPN outputs features at multiple scales
            fpn_channels = [pyramid_channels] * 4  # P2, P3, P4, P5
            self.multi_scale_attention = MultiScaleAttention(
                fpn_channels, 
                out_channels=segmentation_channels
            )
        else:
            self.multi_scale_attention = None
            
        # Additional refinement layers
        self.refine_conv = nn.Sequential(
            nn.Conv2d(segmentation_channels, segmentation_channels, 3, padding=1),
            nn.BatchNorm2d(segmentation_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(segmentation_channels, classes, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output heatmap (B, classes, H, W)
        """
        # Extract encoder features
        features = self.fpn.encoder(x)
        
        # Apply attention to encoder features if enabled
        if self.encoder_attention is not None:
            features = [
                attn(feat) if feat is not None else None
                for attn, feat in zip(self.encoder_attention, features)
            ]
        
        # Generate FPN features
        # The FPN decoder expects features as a list, not unpacked
        decoder_output = self.fpn.decoder(features)
        
        # Apply multi-scale attention if enabled
        if self.multi_scale_attention is not None and isinstance(decoder_output, (list, tuple)):
            # If decoder returns multiple scales, use them
            decoder_output = self.multi_scale_attention(decoder_output)
        
        # Final segmentation head
        if hasattr(self.fpn, 'segmentation_head'):
            output = self.fpn.segmentation_head(decoder_output)
        else:
            # Use our refinement conv if no segmentation head
            output = self.refine_conv(decoder_output)
            
        return output
    
    def get_pyramid_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract pyramid features for visualization or analysis
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of pyramid features at different scales
        """
        features = self.fpn.encoder(x)
        
        if self.encoder_attention is not None:
            features = [
                attn(feat) if feat is not None else None
                for attn, feat in zip(self.encoder_attention, features)
            ]
        
        pyramid_features = self.fpn.decoder(*features)
        
        return pyramid_features if isinstance(pyramid_features, (list, tuple)) else [pyramid_features]


class MultiScaleKeypointHead(nn.Module):
    """
    Multi-scale prediction head for keypoint detection
    
    Generates predictions at multiple scales and fuses them
    
    Args:
        in_channels_list: List of input channels for each scale
        classes: Number of output classes
        scales: List of scale factors for each prediction
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        classes: int = 1,
        scales: List[float] = [1.0, 0.5, 0.25]
    ):
        super().__init__()
        
        self.scales = scales
        self.classes = classes
        
        # Create prediction heads for each scale
        self.heads = nn.ModuleList()
        for in_ch in in_channels_list:
            self.heads.append(nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, classes, 1)
            ))
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: List of feature maps at different scales
            
        Returns:
            Fused multi-scale predictions
        """
        # Generate predictions at each scale
        predictions = []
        target_size = features[0].shape[-2:]
        
        for feat, head, scale in zip(features, self.heads, self.scales):
            # Generate prediction
            pred = head(feat)
            
            # Resize to target size if needed
            if pred.shape[-2:] != target_size:
                pred = F.interpolate(
                    pred,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            predictions.append(pred)
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(pred * weight for pred, weight in zip(predictions, weights))
        
        return fused


class KeypointFPNv2(nn.Module):
    """
    Enhanced FPN model with custom multi-scale heads and attention
    
    This version provides more control over the architecture
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        encoder_depth: int = 5,
        pyramid_channels: int = 256,
        classes: int = 1,
        use_attention: bool = True,
        use_ppm: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Create encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights
        )
        
        encoder_channels = self.encoder.out_channels[1:]  # Skip first (input) channel
        
        # Pyramid pooling module for context
        if use_ppm:
            self.ppm = PyramidPoolingModule(
                encoder_channels[-1],
                pool_sizes=[1, 2, 3, 6],
                out_channels=pyramid_channels // 4
            )
        else:
            self.ppm = None
        
        # FPN layers
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_ch in encoder_channels:
            # Lateral connection
            lateral = nn.Conv2d(in_ch, pyramid_channels, 1)
            self.lateral_convs.append(lateral)
            
            # Output convolution
            output_conv = nn.Sequential(
                nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1),
                nn.BatchNorm2d(pyramid_channels),
                nn.ReLU(inplace=True)
            )
            self.output_convs.append(output_conv)
        
        # Attention modules
        if use_attention:
            self.attention_modules = nn.ModuleList([
                CBAM(pyramid_channels) for _ in encoder_channels
            ])
        else:
            self.attention_modules = None
        
        # Multi-scale head
        self.multi_scale_head = MultiScaleKeypointHead(
            [pyramid_channels] * len(encoder_channels),
            classes=classes,
            scales=[1.0] * len(encoder_channels)
        )
        
        # Final refinement
        self.final_conv = nn.Sequential(
            nn.Conv2d(classes, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, classes, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output heatmap (B, classes, H, W)
        """
        # Encoder forward
        features = self.encoder(x)[1:]  # Skip first feature
        
        # Apply PPM to highest level feature
        if self.ppm is not None:
            features[-1] = self.ppm(features[-1])
        
        # Build FPN features top-down
        pyramid_features = []
        
        for i in range(len(features) - 1, -1, -1):
            # Lateral connection
            lateral = self.lateral_convs[i](features[i])
            
            # Add top-down feature if not the highest level
            if i < len(features) - 1:
                # Upsample previous pyramid feature
                _, _, H, W = lateral.shape
                top_down = F.interpolate(
                    pyramid_features[-1],
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                lateral = lateral + top_down
            
            # Apply output convolution
            out = self.output_convs[i](lateral)
            
            # Apply attention if enabled
            if self.attention_modules is not None:
                out = self.attention_modules[i](out)
            
            pyramid_features.append(out)
        
        # Reverse to have features from low to high level
        pyramid_features = pyramid_features[::-1]
        
        # Multi-scale prediction and fusion
        output = self.multi_scale_head(pyramid_features)
        
        # Final refinement
        output = self.final_conv(output)
        
        return output


def create_fpn_model(
    model_name: str = "fpn_v1",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create FPN-based keypoint detection models
    
    Args:
        model_name: Model variant ('fpn_v1', 'fpn_v2')
        encoder_name: Encoder backbone name
        encoder_weights: Pretrained weights
        in_channels: Number of input channels
        classes: Number of output classes
        **kwargs: Additional model parameters
    
    Returns:
        Model instance
    """
    models = {
        'fpn_v1': KeypointFPNWithAttention,
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


if __name__ == "__main__":
    # Test FPN models
    
    # Test FPN v1 with attention
    print("Testing FPN v1 with attention...")
    model_v1 = create_fpn_model(
        model_name="fpn_v1",
        encoder_name="resnet34",
        attention_type="both"
    )
    
    x = torch.randn(2, 3, 256, 256)
    y = model_v1(x)
    print(f"FPN v1 - Input: {x.shape}, Output: {y.shape}")
    
    # Test FPN v2
    print("\nTesting FPN v2...")
    model_v2 = create_fpn_model(
        model_name="fpn_v2",
        encoder_name="resnet34",
        use_attention=True,
        use_ppm=True
    )
    
    y = model_v2(x)
    print(f"FPN v2 - Input: {x.shape}, Output: {y.shape}")
    
    # Count parameters
    params_v1 = sum(p.numel() for p in model_v1.parameters() if p.requires_grad)
    params_v2 = sum(p.numel() for p in model_v2.parameters() if p.requires_grad)
    print(f"\nFPN v1 parameters: {params_v1:,}")
    print(f"FPN v2 parameters: {params_v2:,}")