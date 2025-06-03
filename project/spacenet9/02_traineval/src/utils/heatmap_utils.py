#!/usr/bin/env python3
"""
Paper準拠のヒートマップ生成ユーティリティ
SpaceNet9 位置合わせタスク用
"""

import numpy as np
import torch
import cv2
from typing import List, Tuple, Union, Optional
import matplotlib.pyplot as plt


def generate_gaussian_heatmap(
    keypoints: List[Tuple[float, float]], 
    image_size: Tuple[int, int] = (256, 256),
    sigma: float = 13.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Paper準拠のガウシアンヒートマップ生成
    
    標準偏差σ=13ピクセルの2Dガウシアンカーネルを使用してヒートマップを生成。
    複数のキーポイントがある場合は最大値で統合。
    
    Args:
        keypoints: キーポイント座標のリスト [(x, y), ...]
        image_size: 画像サイズ (height, width)
        sigma: ガウシアン分布の標準偏差（Paper準拠: 13.0）
        normalize: 正規化するかどうか（0-1範囲）
        
    Returns:
        ガウシアンヒートマップ (H, W)
    """
    h, w = image_size
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # 各キーポイントに対してガウシアン分布を生成
    for kp in keypoints:
        x, y = float(kp[0]), float(kp[1])
        
        # 画像範囲内のキーポイントのみ処理
        if 0 <= x < w and 0 <= y < h:
            # メッシュグリッド作成
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            
            # 2Dガウシアン分布計算
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            
            # 最大値で統合（複数のキーポイントが重複する場合）
            heatmap = np.maximum(heatmap, gaussian)
    
    # 正規化
    if normalize and heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def generate_gaussian_heatmap_fast(
    keypoints: List[Tuple[float, float]], 
    image_size: Tuple[int, int] = (256, 256),
    sigma: float = 13.0,
    cutoff_radius: float = 3.0
) -> np.ndarray:
    """
    高速版ガウシアンヒートマップ生成
    
    効率的な実装のため、ガウシアン分布を局所的な範囲のみで計算。
    cutoff_radius * sigma の範囲外は無視する。
    
    Args:
        keypoints: キーポイント座標のリスト
        image_size: 画像サイズ (height, width)
        sigma: ガウシアン分布の標準偏差
        cutoff_radius: カットオフ半径（σの倍数）
        
    Returns:
        ガウシアンヒートマップ (H, W)
    """
    h, w = image_size
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # カットオフ範囲の計算
    radius = int(cutoff_radius * sigma)
    
    for kp in keypoints:
        x, y = int(round(kp[0])), int(round(kp[1]))
        
        # 画像範囲内のキーポイントのみ処理
        if 0 <= x < w and 0 <= y < h:
            # 局所的な範囲を定義
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            
            # 局所的なメッシュグリッド
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max), 
                np.arange(y_min, y_max)
            )
            
            # 2Dガウシアン分布計算
            gaussian = np.exp(-((xx - kp[0])**2 + (yy - kp[1])**2) / (2 * sigma**2))
            
            # ヒートマップに統合
            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max], 
                gaussian
            )
    
    return heatmap


def generate_multi_channel_heatmap(
    keypoints_list: List[List[Tuple[float, float]]], 
    image_size: Tuple[int, int] = (256, 256),
    sigma: float = 13.0
) -> np.ndarray:
    """
    マルチチャンネルヒートマップ生成
    
    複数のキーポイントセット（例：異なるクラス）に対して
    チャンネル別のヒートマップを生成。
    
    Args:
        keypoints_list: キーポイントセットのリスト
        image_size: 画像サイズ
        sigma: ガウシアン分布の標準偏差
        
    Returns:
        マルチチャンネルヒートマップ (C, H, W)
    """
    num_channels = len(keypoints_list)
    h, w = image_size
    
    heatmaps = np.zeros((num_channels, h, w), dtype=np.float32)
    
    for i, keypoints in enumerate(keypoints_list):
        heatmaps[i] = generate_gaussian_heatmap_fast(
            keypoints, image_size, sigma
        )
    
    return heatmaps


def extract_keypoints_from_heatmap(
    heatmap: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.1,
    nms_radius: int = 5,
    max_keypoints: int = 100
) -> List[Tuple[float, float]]:
    """
    ヒートマップからキーポイントを抽出
    
    Args:
        heatmap: 入力ヒートマップ (H, W) または (1, H, W)
        threshold: 検出閾値
        nms_radius: NMS（Non-Maximum Suppression）の半径
        max_keypoints: 最大キーポイント数
        
    Returns:
        キーポイント座標のリスト [(x, y), ...]
    """
    # テンソルをnumpy配列に変換
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    
    # 次元調整
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()
    
    # 閾値フィルタリング
    candidates = heatmap > threshold
    
    if not candidates.any():
        return []
    
    # Non-Maximum Suppression
    if nms_radius > 0:
        # 最大値プーリングでNMS
        kernel_size = 2 * nms_radius + 1
        maxpool = cv2.dilate(heatmap, np.ones((kernel_size, kernel_size)))
        peaks = (heatmap == maxpool) & candidates
    else:
        peaks = candidates
    
    # ピーク座標を取得
    peak_coords = np.where(peaks)
    
    if len(peak_coords[0]) == 0:
        return []
    
    # スコア順でソート
    scores = heatmap[peak_coords]
    sorted_indices = np.argsort(scores)[::-1]
    
    # 最大キーポイント数に制限
    if len(sorted_indices) > max_keypoints:
        sorted_indices = sorted_indices[:max_keypoints]
    
    # キーポイントリストを作成 (x, y) 形式
    keypoints = []
    for idx in sorted_indices:
        y, x = peak_coords[0][idx], peak_coords[1][idx]
        keypoints.append((float(x), float(y)))
    
    return keypoints


def visualize_heatmap_with_keypoints(
    image: np.ndarray,
    heatmap: np.ndarray,
    keypoints: List[Tuple[float, float]],
    save_path: Optional[str] = None,
    title: str = "Heatmap with Keypoints"
) -> None:
    """
    ヒートマップとキーポイントの可視化
    
    Args:
        image: 元画像 (H, W, 3) または (H, W)
        heatmap: ヒートマップ (H, W)
        keypoints: キーポイント座標のリスト
        save_path: 保存パス（Noneの場合は表示のみ）
        title: 図のタイトル
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元画像
    if image.ndim == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # ヒートマップ
    im1 = axes[1].imshow(heatmap, cmap='hot', interpolation='nearest')
    axes[1].set_title('Heatmap')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # オーバーレイ
    if image.ndim == 3:
        axes[2].imshow(image)
    else:
        axes[2].imshow(image, cmap='gray')
    
    # ヒートマップをオーバーレイ
    axes[2].imshow(heatmap, cmap='hot', alpha=0.5, interpolation='nearest')
    
    # キーポイントをプロット
    if keypoints:
        kp_x, kp_y = zip(*keypoints)
        axes[2].scatter(kp_x, kp_y, c='cyan', s=50, marker='x', linewidths=2)
    
    axes[2].set_title('Overlay with Keypoints')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_heatmap_quality(
    pred_heatmap: np.ndarray,
    gt_heatmap: np.ndarray,
    sigma: float = 13.0
) -> dict:
    """
    ヒートマップの品質評価
    
    Args:
        pred_heatmap: 予測ヒートマップ
        gt_heatmap: 正解ヒートマップ
        sigma: ガウシアン分布の標準偏差
        
    Returns:
        評価指標の辞書
    """
    # MSE
    mse = np.mean((pred_heatmap - gt_heatmap) ** 2)
    
    # MAE
    mae = np.mean(np.abs(pred_heatmap - gt_heatmap))
    
    # ピーク相関
    pred_max = pred_heatmap.max()
    gt_max = gt_heatmap.max()
    peak_correlation = min(pred_max, gt_max) / max(pred_max, gt_max) if gt_max > 0 else 0
    
    # 構造類似度（簡易版）
    pred_norm = pred_heatmap / (pred_heatmap.max() + 1e-8)
    gt_norm = gt_heatmap / (gt_heatmap.max() + 1e-8)
    
    correlation = np.corrcoef(pred_norm.flatten(), gt_norm.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'peak_correlation': float(peak_correlation),
        'correlation': float(correlation)
    }


if __name__ == "__main__":
    # テスト用コード
    
    print("Testing heatmap generation...")
    
    # テストデータ
    image_size = (256, 256)
    keypoints = [(64, 64), (128, 128), (192, 192)]
    
    # 標準的なヒートマップ生成
    heatmap1 = generate_gaussian_heatmap(keypoints, image_size, sigma=13.0)
    print(f"Standard heatmap shape: {heatmap1.shape}")
    print(f"Heatmap range: [{heatmap1.min():.4f}, {heatmap1.max():.4f}]")
    
    # 高速版ヒートマップ生成
    heatmap2 = generate_gaussian_heatmap_fast(keypoints, image_size, sigma=13.0)
    print(f"Fast heatmap shape: {heatmap2.shape}")
    print(f"Fast heatmap range: [{heatmap2.min():.4f}, {heatmap2.max():.4f}]")
    
    # キーポイント抽出テスト
    extracted_kps = extract_keypoints_from_heatmap(heatmap1, threshold=0.1)
    print(f"Extracted keypoints: {len(extracted_kps)}")
    print(f"Original keypoints: {len(keypoints)}")
    
    # 品質評価テスト
    quality = evaluate_heatmap_quality(heatmap1, heatmap2)
    print(f"Heatmap quality comparison: {quality}")
    
    # マルチチャンネルテスト
    keypoints_list = [keypoints[:2], keypoints[1:]]
    multi_heatmap = generate_multi_channel_heatmap(keypoints_list, image_size)
    print(f"Multi-channel heatmap shape: {multi_heatmap.shape}")
    
    print("All heatmap tests completed successfully!")