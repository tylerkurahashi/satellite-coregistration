#!/usr/bin/env python3
"""
Paper準拠の回帰ベース損失関数
SpaceNet9 位置合わせタスク用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class WeightedMSELoss(nn.Module):
    """
    Paper準拠の重み付きMSE損失
    
    ガウシアンヒートマップの中心部分（キーポイント周辺）により高い重みを付与し、
    位置の正確性を重視した学習を行う。
    
    Args:
        positive_weight: ガウシアン部分への重み（デフォルト: 10.0）
        threshold: 重み付け判定の閾値（デフォルト: 0.1）
        reduction: 損失の削減方法 ('mean', 'sum', 'none')
    """
    
    def __init__(
        self, 
        positive_weight: float = 10.0, 
        threshold: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.positive_weight = positive_weight
        self.threshold = threshold
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向き計算
        
        Args:
            pred: 予測ヒートマップ (B, 1, H, W)
            target: 正解ヒートマップ (B, 1, H, W)
            
        Returns:
            重み付きMSE損失
        """
        # ガウシアン部分（target > threshold）により高い重みを付与
        weights = torch.where(
            target > self.threshold, 
            self.positive_weight, 
            1.0
        )
        
        # MSE計算
        mse = (pred - target) ** 2
        
        # 重み付き損失
        weighted_mse = weights * mse
        
        # 削減方法に応じて処理
        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse


class GaussianRegularization(nn.Module):
    """
    ガウシアン正則化項
    
    予測ヒートマップが滑らかな分布を持つように正則化する。
    急激な変化を抑制し、ガウシアン様の分布を促進する。
    
    Args:
        weight: 正則化項の重み（デフォルト: 0.1）
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        前向き計算
        
        Args:
            pred: 予測ヒートマップ (B, 1, H, W)
            
        Returns:
            ガウシアン正則化損失
        """
        # 勾配計算（隣接ピクセルとの差分）
        grad_x = torch.diff(pred, dim=-1)  # x方向勾配
        grad_y = torch.diff(pred, dim=-2)  # y方向勾配
        
        # 勾配の二乗平均（滑らかさの指標）
        smooth_loss = (grad_x ** 2).mean() + (grad_y ** 2).mean()
        
        return self.weight * smooth_loss


class KeypointRegressionLoss(nn.Module):
    """
    キーポイント回帰用の統合損失関数
    
    Paper準拠の回帰ベース損失で、位置合わせタスクに最適化。
    重み付きMSE + ガウシアン正則化の組み合わせ。
    
    Args:
        positive_weight: ガウシアン部分への重み
        regularization_weight: 正則化項の重み
        threshold: 重み付け判定の閾値
    """
    
    def __init__(
        self,
        positive_weight: float = 10.0,
        regularization_weight: float = 0.1,
        threshold: float = 0.1
    ):
        super().__init__()
        self.mse_loss = WeightedMSELoss(positive_weight, threshold)
        self.regularization = GaussianRegularization(regularization_weight)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        前向き計算
        
        Args:
            pred: 予測ヒートマップ (B, 1, H, W)
            target: 正解ヒートマップ (B, 1, H, W)
            
        Returns:
            損失の辞書（total_loss, mse_loss, reg_loss）
        """
        # 重み付きMSE損失
        mse_loss = self.mse_loss(pred, target)
        
        # ガウシアン正則化
        reg_loss = self.regularization(pred)
        
        # 総損失
        total_loss = mse_loss + reg_loss
        
        return {
            'total': total_loss,
            'mse_loss': mse_loss,
            'reg_loss': reg_loss
        }


class AdaptiveWeightedMSELoss(nn.Module):
    """
    適応的重み付きMSE損失
    
    キーポイント密度に応じて重みを動的に調整する高度版。
    キーポイントが少ない画像でも適切に学習できるように改良。
    
    Args:
        base_positive_weight: 基本的なガウシアン部分への重み
        adaptive_factor: 適応的調整の係数
        threshold: 重み付け判定の閾値
    """
    
    def __init__(
        self,
        base_positive_weight: float = 10.0,
        adaptive_factor: float = 2.0,
        threshold: float = 0.1
    ):
        super().__init__()
        self.base_positive_weight = base_positive_weight
        self.adaptive_factor = adaptive_factor
        self.threshold = threshold
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向き計算
        
        Args:
            pred: 予測ヒートマップ (B, 1, H, W)
            target: 正解ヒートマップ (B, 1, H, W)
            
        Returns:
            適応的重み付きMSE損失
        """
        # キーポイント密度の計算
        positive_ratio = (target > self.threshold).float().mean(dim=(-2, -1), keepdim=True)
        
        # 密度に基づく適応的重み調整
        # キーポイントが少ない場合により高い重みを付与
        adaptive_weight = self.base_positive_weight * (
            1.0 + self.adaptive_factor * (1.0 - positive_ratio)
        )
        
        # 重み付きMSE計算
        weights = torch.where(target > self.threshold, adaptive_weight, 1.0)
        mse = (pred - target) ** 2
        weighted_mse = weights * mse
        
        return weighted_mse.mean()


class FocalMSELoss(nn.Module):
    """
    Focal-style MSE損失
    
    予測が困難なピクセル（高い誤差）により集中して学習する。
    Focal Lossの概念をMSE損失に適用。
    
    Args:
        alpha: バランス調整パラメータ
        gamma: フォーカス調整パラメータ
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向き計算
        
        Args:
            pred: 予測ヒートマップ (B, 1, H, W)
            target: 正解ヒートマップ (B, 1, H, W)
            
        Returns:
            Focal-style MSE損失
        """
        # MSE計算
        mse = (pred - target) ** 2
        
        # 誤差の大きさに基づくフォーカス重み
        # 誤差が大きいほど重みが大きくなる
        focal_weight = (mse + 1e-8) ** (self.gamma / 2)
        
        # Focal MSE
        focal_mse = self.alpha * focal_weight * mse
        
        return focal_mse.mean()


def create_regression_loss(loss_config: dict) -> nn.Module:
    """
    設定に基づいて回帰損失関数を作成
    
    Args:
        loss_config: 損失関数の設定辞書
        
    Returns:
        損失関数インスタンス
    """
    loss_type = loss_config.get('type', 'keypoint_regression')
    
    if loss_type == 'keypoint_regression':
        return KeypointRegressionLoss(
            positive_weight=loss_config.get('positive_weight', 10.0),
            regularization_weight=loss_config.get('regularization_weight', 0.1),
            threshold=loss_config.get('threshold', 0.1)
        )
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(
            positive_weight=loss_config.get('positive_weight', 10.0),
            threshold=loss_config.get('threshold', 0.1)
        )
    elif loss_type == 'adaptive_weighted_mse':
        return AdaptiveWeightedMSELoss(
            base_positive_weight=loss_config.get('base_positive_weight', 10.0),
            adaptive_factor=loss_config.get('adaptive_factor', 2.0),
            threshold=loss_config.get('threshold', 0.1)
        )
    elif loss_type == 'focal_mse':
        return FocalMSELoss(
            alpha=loss_config.get('alpha', 1.0),
            gamma=loss_config.get('gamma', 2.0)
        )
    else:
        raise ValueError(f"Unknown regression loss type: {loss_type}")


if __name__ == "__main__":
    # テスト用コード
    
    # ダミーデータ作成
    batch_size, channels, height, width = 2, 1, 256, 256
    
    # 予測ヒートマップ（ランダム）
    pred = torch.rand(batch_size, channels, height, width)
    
    # 正解ヒートマップ（ガウシアン分布）
    target = torch.zeros(batch_size, channels, height, width)
    
    # 中心にガウシアン分布を配置
    center_x, center_y = width // 2, height // 2
    sigma = 13
    
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    gaussian = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    target[0, 0] = gaussian
    
    # 損失関数のテスト
    print("Testing regression losses...")
    
    # 重み付きMSE
    weighted_mse = WeightedMSELoss(positive_weight=10.0)
    loss1 = weighted_mse(pred, target)
    print(f"Weighted MSE Loss: {loss1.item():.6f}")
    
    # ガウシアン正則化
    gaussian_reg = GaussianRegularization(weight=0.1)
    loss2 = gaussian_reg(pred)
    print(f"Gaussian Regularization: {loss2.item():.6f}")
    
    # 統合損失
    keypoint_loss = KeypointRegressionLoss()
    loss_dict = keypoint_loss(pred, target)
    print(f"Keypoint Regression Loss: {loss_dict['total_loss'].item():.6f}")
    print(f"  - MSE component: {loss_dict['mse_loss'].item():.6f}")
    print(f"  - Regularization component: {loss_dict['reg_loss'].item():.6f}")
    
    # 適応的重み付きMSE
    adaptive_mse = AdaptiveWeightedMSELoss()
    loss3 = adaptive_mse(pred, target)
    print(f"Adaptive Weighted MSE Loss: {loss3.item():.6f}")
    
    # Focal MSE
    focal_mse = FocalMSELoss()
    loss4 = focal_mse(pred, target)
    print(f"Focal MSE Loss: {loss4.item():.6f}")
    
    print("All regression loss tests completed successfully!")