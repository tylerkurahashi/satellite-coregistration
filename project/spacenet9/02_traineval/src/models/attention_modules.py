#!/usr/bin/env python3
"""
Attention modules for SpaceNet9 keypoint detection models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Ensure minimum channel size
        reduced_channels = max(1, in_channels // reduction)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Channel attention weights (B, C, 1, 1)
        """
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = torch.sigmoid(avg_out + max_out)
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    Args:
        kernel_size: Convolution kernel size
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Spatial attention weights (B, 1, H, W)
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        
        return torch.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    Combines channel and spatial attention
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio
        kernel_size: Spatial attention kernel size
    """
    
    def __init__(
        self, 
        in_channels: int, 
        reduction: int = 16, 
        kernel_size: int = 7
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Attention-weighted tensor (B, C, H, W)
        """
        # Apply channel attention
        x = x * self.channel_attention(x)
        
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for SAR-Optical feature interaction
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(
        self, 
        x: torch.Tensor, 
        cross_modal_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (B, C, H, W)
            cross_modal_features: Features from other modality (B, C, H, W)
            
        Returns:
            Attention-weighted features (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # Use cross-modal features if provided, otherwise self-attention
        feat = cross_modal_features if cross_modal_features is not None else x
        
        # Generate Q, K, V
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x HW x C'
        key = self.key_conv(feat).view(B, -1, H * W)  # B x C' x HW
        value = self.value_conv(feat).view(B, -1, H * W)  # B x C x HW
        
        # Attention scores
        attention = torch.bmm(query, key)  # B x HW x HW
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(B, C, H, W)
        
        # Output projection with residual
        out = self.out_conv(out)
        out = self.gamma * out + x
        
        return out


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention module for FPN features
    
    Args:
        in_channels_list: List of channel numbers for each scale
        out_channels: Output channel number
    """
    
    def __init__(
        self, 
        in_channels_list: list, 
        out_channels: int = 256
    ):
        super().__init__()
        
        # Channel alignment for different scales
        self.align_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) 
            for in_ch in in_channels_list
        ])
        
        # Scale-aware attention
        self.scale_attention = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, len(in_channels_list), 1),
            nn.Softmax(dim=1)
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: list) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: List of feature maps at different scales
            
        Returns:
            Fused multi-scale features
        """
        # Align channels and resize to same spatial dimension
        target_size = features[0].shape[-2:]
        aligned_features = []
        
        for feat, align_conv in zip(features, self.align_convs):
            # Align channels
            feat = align_conv(feat)
            
            # Resize if needed
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            aligned_features.append(feat)
        
        # Concatenate all scales
        concat_features = torch.cat(aligned_features, dim=1)
        
        # Generate scale attention weights
        scale_weights = self.scale_attention(concat_features)
        
        # Apply scale-specific weights
        weighted_features = []
        for i, feat in enumerate(aligned_features):
            weight = scale_weights[:, i:i+1, :, :]
            weighted_features.append(feat * weight)
        
        # Sum weighted features
        fused = sum(weighted_features)
        
        # Refine
        output = self.refine(fused)
        
        return output


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module for capturing multi-scale context
    
    Args:
        in_channels: Number of input channels
        pool_sizes: List of pooling sizes
        out_channels: Output channels per pooling branch
    """
    
    def __init__(
        self, 
        in_channels: int, 
        pool_sizes: list = [1, 2, 3, 6],
        out_channels: int = 256
    ):
        super().__init__()
        
        self.pool_sizes = pool_sizes
        
        # Pooling branches
        self.pools = nn.ModuleList()
        for pool_size in pool_sizes:
            self.pools.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_sizes) * out_channels, 
                in_channels, 
                3, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Multi-scale context features (B, C, H, W)
        """
        h, w = x.shape[-2:]
        
        # Apply pyramid pooling
        pool_outs = [x]
        for pool in self.pools:
            pool_out = pool(x)
            # Upsample back to original size
            pool_out = F.interpolate(
                pool_out, 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            )
            pool_outs.append(pool_out)
        
        # Concatenate and fuse
        concat = torch.cat(pool_outs, dim=1)
        output = self.fusion(concat)
        
        return output


if __name__ == "__main__":
    # Test attention modules
    
    # Test CBAM
    cbam = CBAM(in_channels=256)
    x = torch.randn(2, 256, 64, 64)
    out = cbam(x)
    print(f"CBAM - Input: {x.shape}, Output: {out.shape}")
    
    # Test Cross-modal Attention
    cross_attn = CrossModalAttention(in_channels=256)
    x1 = torch.randn(2, 256, 32, 32)
    x2 = torch.randn(2, 256, 32, 32)
    out = cross_attn(x1, x2)
    print(f"CrossModalAttention - Input: {x1.shape}, Output: {out.shape}")
    
    # Test Multi-scale Attention
    ms_attn = MultiScaleAttention([64, 128, 256], out_channels=128)
    features = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16)
    ]
    out = ms_attn(features)
    print(f"MultiScaleAttention - Output: {out.shape}")
    
    # Test Pyramid Pooling
    ppm = PyramidPoolingModule(in_channels=512)
    x = torch.randn(2, 512, 32, 32)
    out = ppm(x)
    print(f"PyramidPooling - Input: {x.shape}, Output: {out.shape}")