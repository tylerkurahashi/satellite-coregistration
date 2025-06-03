#!/usr/bin/env python3
"""
Visualization utilities for keypoint detection
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional, Union
import cv2


def visualize_predictions(
    images: torch.Tensor,
    pred_heatmaps: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    save_path: Optional[Path] = None,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (16, 12),
    task: str = 'opt_keypoint_detection'
) -> None:
    """
    Visualize predictions vs ground truth heatmaps
    
    Args:
        images: Input images (B, C, H, W)
        pred_heatmaps: Predicted heatmaps (B, 1, H, W)
        gt_heatmaps: Ground truth heatmaps (B, 1, H, W)
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
        figsize: Figure size
        task: Task type ('opt_keypoint_detection' or 'sar_keypoint_detection')
    """
    batch_size = min(images.shape[0], num_samples)
    
    fig, axes = plt.subplots(3, batch_size, figsize=figsize)
    if batch_size == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(batch_size):
        # Convert image to numpy and denormalize
        img = images[i].cpu()
        if img.shape[0] == 3:
            if task == 'opt_keypoint_detection':
                # Input is SAR image with SAR normalization (0-1)
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                
                # Apply contrast enhancement for visualization if image is too dark
                if img.max() < 0.1:
                    # Stretch to full range
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                    # Apply gamma correction for better visibility
                    img = np.power(img, 0.5)
            else:
                # Input is optical image with ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                
                # Apply contrast enhancement for visualization if image is too dark
                if img.max() < 0.1:
                    # Stretch to full range
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                    # Apply gamma correction for better visibility
                    img = np.power(img, 0.7)
        else:
            img = img[0].numpy()
        
        # Get heatmaps
        pred_hm = pred_heatmaps[i, 0].cpu().numpy()
        gt_hm = gt_heatmaps[i, 0].cpu().numpy()
        
        # Input image
        axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        
        # Predicted heatmap
        im1 = axes[1, i].imshow(pred_hm, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title('Predicted')
        axes[1, i].axis('off')
        
        # Ground truth heatmap
        im2 = axes[2, i].imshow(gt_hm, cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title('Ground Truth')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    save_path: Optional[Path] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = -1
) -> np.ndarray:
    """
    Visualize keypoints on image
    
    Args:
        image: Input image (H, W, C) or (H, W)
        keypoints: Keypoint coordinates (N, 2)
        save_path: Optional path to save image
        color: Color for keypoints (B, G, R)
        radius: Radius of keypoint circles
        thickness: Circle thickness (-1 for filled)
    
    Returns:
        Image with keypoints drawn
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert grayscale to RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Draw keypoints
    vis_image = image.copy()
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(vis_image, (x, y), radius, color, thickness)
    
    if save_path:
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image


def visualize_matching(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    matches: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize keypoint matching between two images
    
    Args:
        img1: First image
        img2: Second image
        kp1: Keypoints in first image (N, 2)
        kp2: Keypoints in second image (M, 2)
        matches: Match indices (K, 2) or None for all-to-all
        save_path: Path to save visualization
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Display images
    ax1.imshow(img1, cmap='gray' if img1.ndim == 2 else None)
    ax2.imshow(img2, cmap='gray' if img2.ndim == 2 else None)
    
    # Plot keypoints
    ax1.scatter(kp1[:, 0], kp1[:, 1], c='red', s=20, marker='+')
    ax2.scatter(kp2[:, 0], kp2[:, 1], c='red', s=20, marker='+')
    
    # Draw matches
    if matches is not None:
        # Create connection lines
        con_list = []
        for i, j in matches:
            con = patches.ConnectionPatch(
                kp1[i], kp2[j], "data", "data",
                axesA=ax1, axesB=ax2,
                color='green', linewidth=1, alpha=0.5
            )
            con_list.append(con)
            ax2.add_artist(con)
    
    ax1.set_title('Image 1')
    ax1.axis('off')
    ax2.set_title('Image 2')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        save_path: Path to save plot
        figsize: Figure size
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=figsize)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'hot'
) -> np.ndarray:
    """
    Create overlay of heatmap on image
    
    Args:
        image: Base image (H, W, C) or (H, W)
        heatmap: Heatmap to overlay (H, W)
        alpha: Transparency of overlay
        colormap: Matplotlib colormap name
    
    Returns:
        Overlaid image
    """
    # Ensure image is RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Normalize image to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Apply colormap to heatmap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Create overlay
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)


def save_prediction_grid(
    images: List[np.ndarray],
    predictions: List[np.ndarray],
    labels: List[str],
    save_path: Path,
    grid_size: Tuple[int, int] = (2, 4),
    figsize: Tuple[int, int] = (16, 8)
) -> None:
    """
    Save grid of predictions
    
    Args:
        images: List of images
        predictions: List of predictions (same length as images)
        labels: List of labels for each image
        save_path: Path to save grid
        grid_size: Grid dimensions (rows, cols)
        figsize: Figure size
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (img, pred, label) in enumerate(zip(images, predictions, labels)):
        if i >= rows * cols:
            break
        
        # Create overlay
        if pred is not None:
            vis = create_heatmap_overlay(img, pred)
        else:
            vis = img
        
        axes[i].imshow(vis)
        axes[i].set_title(label)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), rows * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test visualizations
    import torch
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256)
    pred_heatmaps = torch.sigmoid(torch.randn(batch_size, 1, 256, 256))
    gt_heatmaps = torch.zeros(batch_size, 1, 256, 256)
    
    # Add some peaks to ground truth
    for i in range(batch_size):
        x, y = np.random.randint(50, 200, 2)
        gt_heatmaps[i, 0, y-5:y+5, x-5:x+5] = 1.0
    
    # Test prediction visualization
    visualize_predictions(images, pred_heatmaps, gt_heatmaps)
    
    # Test keypoint visualization
    img = np.random.rand(256, 256, 3)
    keypoints = np.random.rand(10, 2) * 256
    vis_img = visualize_keypoints(img, keypoints)
    
    print("Visualization tests completed!")