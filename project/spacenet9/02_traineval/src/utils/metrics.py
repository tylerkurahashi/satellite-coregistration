#!/usr/bin/env python3
"""
Metrics for keypoint detection evaluation
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, precision_recall_curve


def calculate_pixel_error(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    max_dist: float = 10.0
) -> Tuple[float, float, int]:
    """
    Calculate pixel error between predicted and ground truth keypoints
    
    Args:
        pred_keypoints: Predicted keypoints (N, 2)
        gt_keypoints: Ground truth keypoints (M, 2)
        max_dist: Maximum distance for matching
    
    Returns:
        mean_error: Mean pixel error
        success_rate: Percentage of matches within max_dist
        num_matches: Number of successful matches
    """
    if len(pred_keypoints) == 0 or len(gt_keypoints) == 0:
        return float('inf'), 0.0, 0
    
    # Compute pairwise distances
    pred_keypoints = pred_keypoints[:, None, :]  # (N, 1, 2)
    gt_keypoints = gt_keypoints[None, :, :]      # (1, M, 2)
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=2)  # (N, M)
    
    # Find optimal assignment using Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(distances)
    
    # Calculate metrics
    matched_distances = distances[pred_indices, gt_indices]
    valid_matches = matched_distances < max_dist
    
    if np.any(valid_matches):
        mean_error = np.mean(matched_distances[valid_matches])
        success_rate = np.mean(valid_matches) * 100
        num_matches = np.sum(valid_matches)
    else:
        mean_error = float('inf')
        success_rate = 0.0
        num_matches = 0
    
    return mean_error, success_rate, num_matches


def calculate_repeatability(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    threshold: float = 5.0,
    overlap_region: Optional[Tuple[int, int, int, int]] = None
) -> float:
    """
    Calculate repeatability score between two sets of keypoints
    
    Args:
        keypoints1: First set of keypoints (N, 2)
        keypoints2: Second set of keypoints (M, 2)
        threshold: Distance threshold for matching
        overlap_region: (x_min, y_min, x_max, y_max) for overlap region
    
    Returns:
        Repeatability score (percentage)
    """
    # Filter keypoints in overlap region if specified
    if overlap_region is not None:
        x_min, y_min, x_max, y_max = overlap_region
        
        mask1 = (keypoints1[:, 0] >= x_min) & (keypoints1[:, 0] <= x_max) & \
                (keypoints1[:, 1] >= y_min) & (keypoints1[:, 1] <= y_max)
        keypoints1 = keypoints1[mask1]
        
        mask2 = (keypoints2[:, 0] >= x_min) & (keypoints2[:, 0] <= x_max) & \
                (keypoints2[:, 1] >= y_min) & (keypoints2[:, 1] <= y_max)
        keypoints2 = keypoints2[mask2]
    
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return 0.0
    
    # Compute pairwise distances
    distances = np.linalg.norm(keypoints1[:, None] - keypoints2[None, :], axis=2)
    
    # Count matches
    matches1 = np.any(distances < threshold, axis=1)
    matches2 = np.any(distances < threshold, axis=0)
    
    # Calculate repeatability
    repeatability = (np.sum(matches1) + np.sum(matches2)) / (len(keypoints1) + len(keypoints2)) * 100
    
    return repeatability


def calculate_localization_error(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate localization error metrics from heatmaps
    
    Args:
        pred_heatmap: Predicted heatmap (B, 1, H, W)
        gt_heatmap: Ground truth heatmap (B, 1, H, W)
        threshold: Threshold for peak detection
    
    Returns:
        Dictionary of metrics
    """
    batch_size = pred_heatmap.shape[0]
    metrics = {
        'mae': 0.0,
        'mse': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }
    
    for b in range(batch_size):
        # Get predicted and ground truth peaks
        pred_mask = pred_heatmap[b, 0] > threshold
        gt_mask = gt_heatmap[b, 0] > threshold
        
        # Calculate pixel-wise metrics
        mae = torch.abs(pred_heatmap[b, 0] - gt_heatmap[b, 0]).mean()
        mse = ((pred_heatmap[b, 0] - gt_heatmap[b, 0]) ** 2).mean()
        
        # Calculate detection metrics
        tp = (pred_mask & gt_mask).sum().float()
        fp = (pred_mask & ~gt_mask).sum().float()
        fn = (~pred_mask & gt_mask).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Accumulate metrics
        metrics['mae'] += mae.item()
        metrics['mse'] += mse.item()
        metrics['precision'] += precision.item()
        metrics['recall'] += recall.item()
        metrics['f1'] += f1.item()
    
    # Average over batch
    for key in metrics:
        metrics[key] /= batch_size
    
    return metrics


def calculate_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    post_processor: Optional[object] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for model evaluation
    
    Args:
        outputs: Model outputs (B, 1, H, W)
        targets: Target heatmaps (B, 1, H, W)
        post_processor: Optional post-processor for keypoint extraction
    
    Returns:
        Dictionary of metrics
    """
    # Ensure outputs are in [0, 1] range
    if outputs.min() < 0 or outputs.max() > 1:
        outputs = torch.sigmoid(outputs)
    
    # Calculate heatmap metrics
    heatmap_metrics = calculate_localization_error(outputs, targets)
    
    # If post-processor is provided, calculate keypoint metrics
    if post_processor is not None:
        pred_keypoints_batch = post_processor.extract_keypoints(outputs)
        gt_keypoints_batch = post_processor.extract_keypoints(targets)
        
        pixel_errors = []
        success_rates = []
        
        for pred_kps, gt_kps in zip(pred_keypoints_batch, gt_keypoints_batch):
            if len(pred_kps) > 0 and len(gt_kps) > 0:
                mean_error, success_rate, _ = calculate_pixel_error(
                    pred_kps.cpu().numpy(),
                    gt_kps.cpu().numpy()
                )
                if mean_error < float('inf'):
                    pixel_errors.append(mean_error)
                    success_rates.append(success_rate)
        
        if pixel_errors:
            heatmap_metrics['pixel_error'] = np.mean(pixel_errors)
            heatmap_metrics['success_rate'] = np.mean(success_rates)
        else:
            heatmap_metrics['pixel_error'] = float('inf')
            heatmap_metrics['success_rate'] = 0.0
    
    return heatmap_metrics


class MetricTracker:
    """
    Track metrics over epochs
    """
    
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}
        self.best = {metric: None for metric in metrics}
        self.best_epoch = {metric: -1 for metric in metrics}
    
    def update(self, metrics_dict: Dict[str, float], epoch: int):
        """Update metrics for current epoch"""
        for metric in self.metrics:
            if metric in metrics_dict:
                value = metrics_dict[metric]
                self.history[metric].append(value)
                
                # Update best
                if self.best[metric] is None:
                    self.best[metric] = value
                    self.best_epoch[metric] = epoch
                else:
                    # Assume lower is better for loss/error metrics
                    if 'loss' in metric or 'error' in metric:
                        if value < self.best[metric]:
                            self.best[metric] = value
                            self.best_epoch[metric] = epoch
                    else:
                        # Higher is better for accuracy/rate metrics
                        if value > self.best[metric]:
                            self.best[metric] = value
                            self.best_epoch[metric] = epoch
    
    def get_best(self, metric: str) -> Tuple[float, int]:
        """Get best value and epoch for a metric"""
        return self.best[metric], self.best_epoch[metric]
    
    def get_history(self, metric: str) -> List[float]:
        """Get history for a metric"""
        return self.history[metric]
    
    def summary(self) -> str:
        """Get summary of best metrics"""
        summary = "Best metrics:\n"
        for metric in self.metrics:
            if self.best[metric] is not None:
                summary += f"  {metric}: {self.best[metric]:.4f} (epoch {self.best_epoch[metric]})\n"
        return summary


def calculate_average_precision(
    pred_heatmap: np.ndarray,
    gt_heatmap: np.ndarray,
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    num_thresholds: int = 10
) -> Dict[str, float]:
    """
    Calculate Average Precision (AP) for keypoint detection
    
    Args:
        pred_heatmap: Predicted heatmap (H, W)
        gt_heatmap: Ground truth heatmap (H, W)
        threshold_range: Range of thresholds to evaluate
        num_thresholds: Number of thresholds to test
        
    Returns:
        Dictionary with AP metrics
    """
    # Flatten heatmaps
    pred_flat = pred_heatmap.flatten()
    gt_flat = (gt_heatmap > 0.5).astype(int).flatten()
    
    # Calculate AP using sklearn
    if np.sum(gt_flat) == 0:
        return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    ap = average_precision_score(gt_flat, pred_flat)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(gt_flat, pred_flat)
    
    return {
        'ap': ap,
        'precision': precision.mean(),
        'recall': recall.mean(),
        'max_precision': precision.max(),
        'max_recall': recall.max()
    }


def calculate_detection_metrics(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    distance_thresholds: List[float] = [5.0, 10.0, 15.0, 20.0]
) -> Dict[str, float]:
    """
    Calculate detection metrics at multiple distance thresholds (like COCO AP)
    
    Args:
        pred_keypoints: Predicted keypoints (N, 2)
        gt_keypoints: Ground truth keypoints (M, 2)
        distance_thresholds: List of distance thresholds
        
    Returns:
        Dictionary with detection metrics
    """
    metrics = {}
    
    if len(pred_keypoints) == 0 or len(gt_keypoints) == 0:
        for thresh in distance_thresholds:
            metrics[f'recall@{thresh}'] = 0.0
            metrics[f'precision@{thresh}'] = 0.0
        metrics['ap'] = 0.0
        return metrics
    
    # Calculate distances between all pairs
    from scipy.spatial.distance import cdist
    distances = cdist(pred_keypoints, gt_keypoints, metric='euclidean')
    
    ap_values = []
    
    for thresh in distance_thresholds:
        # Find matches within threshold
        row_indices, col_indices = linear_sum_assignment(distances)
        
        tp = 0
        for r, c in zip(row_indices, col_indices):
            if distances[r, c] <= thresh:
                tp += 1
        
        fp = len(pred_keypoints) - tp
        fn = len(gt_keypoints) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics[f'recall@{thresh}'] = recall
        metrics[f'precision@{thresh}'] = precision
        
        # Calculate AP at this threshold
        if tp > 0:
            ap_values.append(precision)
        else:
            ap_values.append(0.0)
    
    # Average AP across thresholds
    metrics['ap'] = np.mean(ap_values)
    
    return metrics


def calculate_comprehensive_metrics(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    pred_keypoints: Optional[np.ndarray] = None,
    gt_keypoints: Optional[np.ndarray] = None,
    distance_thresholds: List[float] = [5.0, 10.0, 15.0, 20.0]
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics including AP
    
    Args:
        pred_heatmap: Predicted heatmap tensor
        gt_heatmap: Ground truth heatmap tensor
        pred_keypoints: Predicted keypoints (optional)
        gt_keypoints: Ground truth keypoints (optional)
        distance_thresholds: Distance thresholds for detection metrics
        
    Returns:
        Dictionary with all metrics
    """
    # Convert tensors to numpy
    pred_np = pred_heatmap.detach().cpu().numpy()
    gt_np = gt_heatmap.detach().cpu().numpy()
    
    # Squeeze if needed
    if pred_np.ndim > 2:
        pred_np = pred_np.squeeze()
    if gt_np.ndim > 2:
        gt_np = gt_np.squeeze()
    
    metrics = {}
    
    # Basic heatmap metrics
    basic_metrics = calculate_metrics(pred_heatmap, gt_heatmap)
    metrics.update(basic_metrics)
    
    # Average Precision for heatmap
    ap_metrics = calculate_average_precision(pred_np, gt_np)
    for key, value in ap_metrics.items():
        metrics[f'heatmap_{key}'] = value
    
    # Keypoint detection metrics if keypoints are provided
    if pred_keypoints is not None and gt_keypoints is not None:
        detection_metrics = calculate_detection_metrics(
            pred_keypoints, gt_keypoints, distance_thresholds
        )
        for key, value in detection_metrics.items():
            metrics[f'keypoint_{key}'] = value
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    
    # Test pixel error calculation
    pred_kps = np.array([[10, 20], [30, 40], [50, 60]])
    gt_kps = np.array([[11, 21], [32, 41], [55, 65]])
    
    mean_error, success_rate, num_matches = calculate_pixel_error(pred_kps, gt_kps)
    print(f"Mean pixel error: {mean_error:.2f}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Number of matches: {num_matches}")
    
    # Test heatmap metrics
    pred_heatmap = torch.rand(2, 1, 256, 256)
    gt_heatmap = torch.rand(2, 1, 256, 256)
    
    metrics = calculate_localization_error(pred_heatmap, gt_heatmap)
    print("\nHeatmap metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")