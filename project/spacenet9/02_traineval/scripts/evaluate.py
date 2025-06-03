#!/usr/bin/env python3
"""
Evaluation script for SpaceNet9 keypoint detection
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import create_single_dataloader
from src.models import create_model, HeatmapPostProcessor
from src.utils import (
    calculate_pixel_error,
    calculate_metrics,
    MetricTracker,
    load_model_only,
    visualize_predictions,
    visualize_matching,
    save_prediction_grid
)


class Evaluator:
    """
    Evaluator class for SpaceNet9 models
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_evaluation()
        
    def _setup_model(self):
        """Setup model"""
        # Create model
        self.model = create_model(
            model_name=self.config['model']['name'],
            encoder_name=self.config['model']['encoder'],
            encoder_weights=None,  # Don't load pretrained weights
            in_channels=self.config['model'].get('in_channels', 3),
            classes=self.config['model'].get('classes', 1)
        )
        
        # Load checkpoint
        checkpoint_path = Path(self.config['checkpoint_path'])
        load_model_only(self.model, checkpoint_path, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        
    def _setup_data(self):
        """Setup data loader"""
        self.dataloader = create_single_dataloader(
            data_root=self.config['data']['data_root'],
            regions=self.config['data']['regions'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            task=self.config['data']['task'],
            mode='val',
            shuffle=False
        )
        
    def _setup_evaluation(self):
        """Setup evaluation components"""
        # Heatmap post-processor
        pp_config = self.config['evaluation']['post_process']
        self.post_processor = HeatmapPostProcessor(
            threshold=pp_config['threshold'],
            nms_radius=pp_config['nms_radius'],
            max_keypoints=pp_config['max_keypoints']
        )
        
        # Metric tracker
        self.metric_tracker = MetricTracker([
            'mae', 'mse', 'precision', 'recall', 'f1',
            'pixel_error', 'success_rate'
        ])
        
    def extract_keypoints_from_heatmap(
        self, 
        heatmap: torch.Tensor
    ) -> List[np.ndarray]:
        """Extract keypoints from heatmap"""
        # Apply sigmoid if needed
        if heatmap.min() < 0 or heatmap.max() > 1:
            heatmap = torch.sigmoid(heatmap)
        
        # Extract keypoints
        keypoints_batch = self.post_processor.extract_keypoints(heatmap)
        
        # Refine keypoints to sub-pixel accuracy
        refined_keypoints = []
        for i, kps in enumerate(keypoints_batch):
            if len(kps) > 0:
                refined = self.post_processor.refine_keypoints(
                    heatmap[i:i+1], kps
                )
                refined_keypoints.append(refined.cpu().numpy())
            else:
                refined_keypoints.append(np.array([]))
        
        return refined_keypoints
    
    def match_keypoints(
        self,
        kp1: np.ndarray,
        kp2: np.ndarray,
        max_dist: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match keypoints between two sets
        
        Args:
            kp1: First set of keypoints (N, 2)
            kp2: Second set of keypoints (M, 2)
            max_dist: Maximum distance for matching
        
        Returns:
            matches: Matched indices (K, 2)
            kp1_matched: Matched keypoints from set 1
            kp2_matched: Matched keypoints from set 2
        """
        if len(kp1) == 0 or len(kp2) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Compute pairwise distances
        distances = np.linalg.norm(kp1[:, None] - kp2[None, :], axis=2)
        
        # Find matches within threshold
        matches = []
        used_kp2 = set()
        
        for i in range(len(kp1)):
            # Find nearest neighbor
            min_idx = np.argmin(distances[i])
            min_dist = distances[i, min_idx]
            
            if min_dist < max_dist and min_idx not in used_kp2:
                matches.append([i, min_idx])
                used_kp2.add(min_idx)
        
        if matches:
            matches = np.array(matches)
            kp1_matched = kp1[matches[:, 0]]
            kp2_matched = kp2[matches[:, 1]]
        else:
            matches = np.array([])
            kp1_matched = np.array([])
            kp2_matched = np.array([])
        
        return matches, kp1_matched, kp2_matched
    
    def estimate_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        transform_type: str = 'affine'
    ) -> Optional[np.ndarray]:
        """
        Estimate transformation matrix between point sets
        
        Args:
            src_points: Source points (N, 2)
            dst_points: Destination points (N, 2)
            transform_type: Type of transform ('affine', 'homography', 'similarity')
        
        Returns:
            Transformation matrix or None
        """
        if len(src_points) < 3 or len(dst_points) < 3:
            return None
        
        if transform_type == 'affine':
            # Need at least 3 points
            transform, inliers = cv2.estimateAffine2D(
                src_points, dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if transform is not None:
                # Convert to 3x3 matrix
                transform_3x3 = np.eye(3)
                transform_3x3[:2, :] = transform
                return transform_3x3
                
        elif transform_type == 'homography':
            # Need at least 4 points
            if len(src_points) < 4:
                return None
                
            transform, inliers = cv2.findHomography(
                src_points, dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            return transform
            
        elif transform_type == 'similarity':
            # Estimate similarity transform (rotation, scale, translation)
            transform, inliers = cv2.estimateAffinePartial2D(
                src_points, dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if transform is not None:
                # Convert to 3x3 matrix
                transform_3x3 = np.eye(3)
                transform_3x3[:2, :] = transform
                return transform_3x3
        
        return None
    
    def evaluate_batch(self, batch: Dict) -> Dict[str, float]:
        """Evaluate a single batch"""
        # Move to device
        inputs = batch['input'].to(self.device)
        targets = batch['target'].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Calculate heatmap metrics
        metrics = calculate_metrics(outputs, targets, self.post_processor)
        
        # Extract keypoints for matching evaluation
        if self.config['evaluation'].get('evaluate_matching', True):
            pred_keypoints = self.extract_keypoints_from_heatmap(outputs)
            gt_keypoints = self.extract_keypoints_from_heatmap(targets)
            
            # Evaluate matching for each sample
            pixel_errors = []
            success_rates = []
            num_matches_list = []
            
            for pred_kps, gt_kps in zip(pred_keypoints, gt_keypoints):
                if len(pred_kps) > 0 and len(gt_kps) > 0:
                    # Direct keypoint evaluation
                    mean_error, success_rate, num_matches = calculate_pixel_error(
                        pred_kps, gt_kps,
                        max_dist=self.config['evaluation']['pixel_error_threshold']
                    )
                    
                    if mean_error < float('inf'):
                        pixel_errors.append(mean_error)
                        success_rates.append(success_rate)
                        num_matches_list.append(num_matches)
            
            if pixel_errors:
                metrics['matching_pixel_error'] = np.mean(pixel_errors)
                metrics['matching_success_rate'] = np.mean(success_rates)
                metrics['avg_matches'] = np.mean(num_matches_list)
        
        return metrics, outputs, targets
    
    def evaluate(self):
        """Run full evaluation"""
        print("Starting evaluation...")
        
        all_metrics = []
        visualizations = []
        
        with torch.no_grad():
            pbar = tqdm(self.dataloader, desc='Evaluating')
            
            for batch_idx, batch in enumerate(pbar):
                # Evaluate batch
                metrics, outputs, targets = self.evaluate_batch(batch)
                all_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'mae': f"{metrics['mae']:.4f}",
                    'pixel_error': f"{metrics.get('pixel_error', 0):.2f}"
                })
                
                # Save visualizations for first few batches
                if batch_idx < self.config['evaluation'].get('num_vis_batches', 5):
                    vis_path = self.output_dir / f'vis_batch_{batch_idx}.png'
                    visualize_predictions(
                        batch['input'][:4],  # First 4 samples
                        outputs[:4],
                        targets[:4],
                        save_path=vis_path,
                        task=self.config['data']['task']
                    )
                    visualizations.append(vis_path)
        
        # Aggregate metrics
        final_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values and not any(np.isinf(v) for v in values):
                final_metrics[key] = np.mean(values)
        
        # Update metric tracker
        self.metric_tracker.update(final_metrics, epoch=0)
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 50)
        for key, value in final_metrics.items():
            print(f"{key:20s}: {value:.4f}")
        print("-" * 50)
        
        # Save results
        # Convert numpy float32 to Python float for JSON serialization
        json_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        
        results = {
            'config': self.config,
            'metrics': json_metrics,
            'visualizations': [str(p) for p in visualizations]
        }
        
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        
        return final_metrics
    
    def evaluate_cross_modal_matching(self):
        """Evaluate cross-modal matching performance"""
        print("\nEvaluating cross-modal matching...")
        
        # This would require loading both optical and SAR keypoint models
        # and evaluating the matching between their predictions
        # Implementation depends on specific requirements
        
        pass


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate SpaceNet9 model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to evaluation configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--regions',
        nargs='+',
        help='Regions to evaluate on'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['checkpoint_path'] = args.checkpoint
    config['output_dir'] = args.output_dir
    config['device'] = args.device
    
    if args.regions:
        config['data']['regions'] = args.regions
    
    # Create evaluator and run evaluation
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == '__main__':
    main()