#!/usr/bin/env python3
"""
Paper準拠の位置合わせ評価システム
SpaceNet9 回帰タスク用評価指標
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import cv2


def calculate_registration_accuracy(
    pred_keypoints: Union[List[Tuple[float, float]], np.ndarray],
    gt_keypoints: Union[List[Tuple[float, float]], np.ndarray],
    pixel_threshold: float = 10.0,
    max_distance: float = 50.0
) -> Dict[str, float]:
    """
    Paper準拠の位置合わせ精度評価
    
    予測キーポイントと正解キーポイント間の位置誤差を計算し、
    SpaceNet9 paperの評価方式に準拠した指標を提供。
    
    Args:
        pred_keypoints: 予測キーポイント座標
        gt_keypoints: 正解キーポイント座標
        pixel_threshold: 成功判定のピクセル閾値
        max_distance: マッチング可能な最大距離
        
    Returns:
        評価指標の辞書
    """
    # 入力の正規化
    if isinstance(pred_keypoints, list):
        pred_keypoints = np.array(pred_keypoints, dtype=np.float32)
    if isinstance(gt_keypoints, list):
        gt_keypoints = np.array(gt_keypoints, dtype=np.float32)
    
    # 空の場合の処理
    if len(pred_keypoints) == 0 or len(gt_keypoints) == 0:
        return {
            'mean_pixel_error': float('inf'),
            'median_pixel_error': float('inf'),
            'std_pixel_error': 0.0,
            'success_rate': 0.0,
            'detection_rate': 0.0,
            'num_matches': 0,
            'num_predicted': len(pred_keypoints),
            'num_ground_truth': len(gt_keypoints)
        }
    
    # 距離行列計算
    distances = cdist(pred_keypoints, gt_keypoints, metric='euclidean')
    
    # ハンガリアンアルゴリズムで最適マッチング
    pred_indices, gt_indices = linear_sum_assignment(distances)
    
    # マッチングされた距離
    matched_distances = distances[pred_indices, gt_indices]
    
    # 最大距離以内のマッチのみを有効とする
    valid_matches = matched_distances < max_distance
    valid_distances = matched_distances[valid_matches]
    
    # 評価指標計算
    if len(valid_distances) > 0:
        mean_error = float(np.mean(valid_distances))
        median_error = float(np.median(valid_distances))
        std_error = float(np.std(valid_distances))
        success_rate = float(np.mean(valid_distances < pixel_threshold) * 100)
    else:
        mean_error = float('inf')
        median_error = float('inf')
        std_error = 0.0
        success_rate = 0.0
    
    # 検出率（正解キーポイントのうち何個が検出されたか）
    detection_rate = float(len(valid_distances) / len(gt_keypoints) * 100)
    
    return {
        'mean_pixel_error': mean_error,
        'median_pixel_error': median_error,
        'std_pixel_error': std_error,
        'success_rate': success_rate,
        'detection_rate': detection_rate,
        'num_matches': int(len(valid_distances)),
        'num_predicted': int(len(pred_keypoints)),
        'num_ground_truth': int(len(gt_keypoints))
    }


def evaluate_heatmap_regression(
    pred_heatmap: Union[torch.Tensor, np.ndarray],
    gt_heatmap: Union[torch.Tensor, np.ndarray],
    extract_keypoints_func: callable,
    pixel_threshold: float = 10.0
) -> Dict[str, float]:
    """
    ヒートマップベースの回帰評価
    
    予測ヒートマップと正解ヒートマップから評価指標を計算。
    位置誤差とヒートマップ品質の両方を評価。
    
    Args:
        pred_heatmap: 予測ヒートマップ
        gt_heatmap: 正解ヒートマップ
        extract_keypoints_func: キーポイント抽出関数
        pixel_threshold: 成功判定の閾値
        
    Returns:
        総合評価指標
    """
    # テンソルをnumpy配列に変換
    if isinstance(pred_heatmap, torch.Tensor):
        pred_heatmap = pred_heatmap.detach().cpu().numpy()
    if isinstance(gt_heatmap, torch.Tensor):
        gt_heatmap = gt_heatmap.detach().cpu().numpy()
    
    # 次元調整
    if pred_heatmap.ndim > 2:
        pred_heatmap = pred_heatmap.squeeze()
    if gt_heatmap.ndim > 2:
        gt_heatmap = gt_heatmap.squeeze()
    
    # キーポイント抽出
    pred_keypoints = extract_keypoints_func(pred_heatmap)
    gt_keypoints = extract_keypoints_func(gt_heatmap)
    
    # 位置精度評価
    registration_metrics = calculate_registration_accuracy(
        pred_keypoints, gt_keypoints, pixel_threshold
    )
    
    # ヒートマップ品質評価
    heatmap_metrics = calculate_heatmap_similarity(pred_heatmap, gt_heatmap)
    
    # 統合指標
    combined_metrics = {**registration_metrics, **heatmap_metrics}
    
    return combined_metrics


def calculate_heatmap_similarity(
    pred_heatmap: np.ndarray,
    gt_heatmap: np.ndarray
) -> Dict[str, float]:
    """
    ヒートマップ間の類似度評価
    
    Args:
        pred_heatmap: 予測ヒートマップ
        gt_heatmap: 正解ヒートマップ
        
    Returns:
        類似度指標の辞書
    """
    # MSE (Mean Squared Error)
    mse = float(np.mean((pred_heatmap - gt_heatmap) ** 2))
    
    # MAE (Mean Absolute Error)
    mae = float(np.mean(np.abs(pred_heatmap - gt_heatmap)))
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse > 0:
        psnr = float(20 * np.log10(1.0 / np.sqrt(mse)))
    else:
        psnr = float('inf')
    
    # 正規化相互相関
    pred_norm = pred_heatmap / (np.linalg.norm(pred_heatmap) + 1e-8)
    gt_norm = gt_heatmap / (np.linalg.norm(gt_heatmap) + 1e-8)
    ncc = float(np.sum(pred_norm * gt_norm))
    
    # ピアソン相関係数
    if pred_heatmap.size > 1:
        correlation = np.corrcoef(pred_heatmap.flatten(), gt_heatmap.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # 構造類似度指数（簡易版）
    mu1, mu2 = pred_heatmap.mean(), gt_heatmap.mean()
    sigma1, sigma2 = pred_heatmap.std(), gt_heatmap.std()
    sigma12 = np.mean((pred_heatmap - mu1) * (gt_heatmap - mu2))
    
    c1, c2 = 0.01, 0.03
    ssim = float(
        ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) /
        ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    )
    
    return {
        'heatmap_mse': mse,
        'heatmap_mae': mae,
        'heatmap_psnr': psnr,
        'heatmap_ncc': ncc,
        'heatmap_correlation': float(correlation),
        'heatmap_ssim': ssim
    }


def calculate_localization_precision(
    pred_keypoints: List[Tuple[float, float]],
    gt_keypoints: List[Tuple[float, float]],
    distance_thresholds: List[float] = [5.0, 10.0, 15.0, 20.0]
) -> Dict[str, float]:
    """
    多段階の距離閾値での位置特定精度評価
    
    COCO-style evaluation の位置合わせ版。
    異なる距離閾値での成功率を計算。
    
    Args:
        pred_keypoints: 予測キーポイント
        gt_keypoints: 正解キーポイント
        distance_thresholds: 評価距離閾値のリスト
        
    Returns:
        各閾値での評価指標
    """
    if len(pred_keypoints) == 0 or len(gt_keypoints) == 0:
        return {f'precision@{t}px': 0.0 for t in distance_thresholds}
    
    # 距離計算
    pred_array = np.array(pred_keypoints)
    gt_array = np.array(gt_keypoints)
    distances = cdist(pred_array, gt_array)
    
    # 最適マッチング
    pred_indices, gt_indices = linear_sum_assignment(distances)
    matched_distances = distances[pred_indices, gt_indices]
    
    # 各閾値での評価
    precision_at_threshold = {}
    for threshold in distance_thresholds:
        success_count = np.sum(matched_distances < threshold)
        precision = success_count / len(matched_distances)
        precision_at_threshold[f'precision@{threshold}px'] = float(precision)
    
    # 平均精度計算（AP）
    precisions = []
    for threshold in distance_thresholds:
        success_count = np.sum(matched_distances < threshold)
        precision = success_count / len(matched_distances)
        precisions.append(precision)
    
    average_precision = float(np.mean(precisions))
    precision_at_threshold['average_precision'] = average_precision
    
    return precision_at_threshold


def batch_evaluate_registration(
    pred_heatmaps: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    extract_keypoints_func: callable,
    pixel_threshold: float = 10.0
) -> Dict[str, float]:
    """
    バッチでの位置合わせ評価
    
    Args:
        pred_heatmaps: 予測ヒートマップのバッチ (B, 1, H, W)
        gt_heatmaps: 正解ヒートマップのバッチ (B, 1, H, W)
        extract_keypoints_func: キーポイント抽出関数
        pixel_threshold: 成功判定の閾値
        
    Returns:
        バッチ平均の評価指標
    """
    batch_size = pred_heatmaps.shape[0]
    
    # 各指標のリスト
    all_metrics = []
    
    for i in range(batch_size):
        # 単一サンプルの評価
        metrics = evaluate_heatmap_regression(
            pred_heatmaps[i], 
            gt_heatmaps[i],
            extract_keypoints_func,
            pixel_threshold
        )
        all_metrics.append(metrics)
    
    # バッチ平均の計算
    batch_metrics = {}
    
    # 有効なサンプル（無限大でない）のみを使用して平均計算
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isinf(m[key])]
        if values:
            batch_metrics[f'avg_{key}'] = float(np.mean(values))
            batch_metrics[f'std_{key}'] = float(np.std(values))
        else:
            batch_metrics[f'avg_{key}'] = float('inf')
            batch_metrics[f'std_{key}'] = 0.0
    
    # 総合統計
    batch_metrics['total_samples'] = batch_size
    batch_metrics['valid_samples'] = len([m for m in all_metrics if not np.isinf(m['mean_pixel_error'])])
    
    return batch_metrics


class RegistrationMetricsTracker:
    """
    位置合わせ評価指標の追跡クラス
    
    学習中の評価指標を追跡し、履歴を管理する。
    """
    
    def __init__(self, metrics_to_track: Optional[List[str]] = None):
        """
        初期化
        
        Args:
            metrics_to_track: 追跡する指標のリスト
        """
        if metrics_to_track is None:
            metrics_to_track = [
                'mean_pixel_error', 'success_rate', 'detection_rate',
                'heatmap_mse', 'heatmap_mae'
            ]
        
        self.metrics_to_track = metrics_to_track
        self.history = {metric: [] for metric in metrics_to_track}
        self.best_values = {metric: None for metric in metrics_to_track}
        self.best_epochs = {metric: -1 for metric in metrics_to_track}
    
    def update(self, metrics: Dict[str, float], epoch: int):
        """
        指標を更新
        
        Args:
            metrics: 評価指標の辞書
            epoch: エポック番号
        """
        for metric in self.metrics_to_track:
            if metric in metrics:
                value = metrics[metric]
                self.history[metric].append(value)
                
                # ベスト値の更新
                if self.best_values[metric] is None:
                    self.best_values[metric] = value
                    self.best_epochs[metric] = epoch
                else:
                    # 位置誤差系は小さいほうが良い
                    if 'error' in metric or 'mse' in metric or 'mae' in metric:
                        if value < self.best_values[metric]:
                            self.best_values[metric] = value
                            self.best_epochs[metric] = epoch
                    # 成功率系は大きいほうが良い
                    else:
                        if value > self.best_values[metric]:
                            self.best_values[metric] = value
                            self.best_epochs[metric] = epoch
    
    def get_summary(self) -> str:
        """
        評価サマリーを取得
        
        Returns:
            評価サマリーの文字列
        """
        summary = "Registration Metrics Summary:\n"
        summary += "=" * 40 + "\n"
        
        for metric in self.metrics_to_track:
            if metric in self.best_values and self.best_values[metric] is not None:
                best_val = self.best_values[metric]
                best_epoch = self.best_epochs[metric]
                
                if np.isinf(best_val):
                    summary += f"{metric}: inf (epoch {best_epoch})\n"
                else:
                    summary += f"{metric}: {best_val:.4f} (epoch {best_epoch})\n"
        
        return summary
    
    def get_best(self, metric: str) -> Tuple[float, int]:
        """
        特定指標のベスト値を取得
        
        Args:
            metric: 指標名
            
        Returns:
            (ベスト値, エポック番号)
        """
        return self.best_values.get(metric), self.best_epochs.get(metric)


if __name__ == "__main__":
    # テスト用コード
    
    print("Testing registration metrics...")
    
    # テストデータ作成
    np.random.seed(42)
    
    # 正解キーポイント
    gt_keypoints = [(50, 50), (100, 100), (150, 150)]
    
    # 予測キーポイント（少しノイズを加える）
    pred_keypoints = [(52, 48), (98, 102), (151, 148), (200, 200)]  # 追加の誤検出
    
    # 位置合わせ精度評価
    registration_metrics = calculate_registration_accuracy(
        pred_keypoints, gt_keypoints, pixel_threshold=10.0
    )
    print("Registration accuracy:")
    for key, value in registration_metrics.items():
        print(f"  {key}: {value}")
    
    # 多段階精度評価
    precision_metrics = calculate_localization_precision(
        pred_keypoints, gt_keypoints
    )
    print("\nLocalization precision:")
    for key, value in precision_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # ヒートマップ類似度評価
    pred_heatmap = np.random.rand(256, 256) * 0.5
    gt_heatmap = np.random.rand(256, 256) * 0.5
    
    similarity_metrics = calculate_heatmap_similarity(pred_heatmap, gt_heatmap)
    print("\nHeatmap similarity:")
    for key, value in similarity_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # メトリクストラッカーのテスト
    tracker = RegistrationMetricsTracker()
    
    # 模擬エポックでの更新
    for epoch in range(5):
        fake_metrics = {
            'mean_pixel_error': 10.0 - epoch,
            'success_rate': 50.0 + epoch * 10,
            'detection_rate': 60.0 + epoch * 5
        }
        tracker.update(fake_metrics, epoch)
    
    print("\nMetrics tracker summary:")
    print(tracker.get_summary())
    
    print("All registration metrics tests completed successfully!")