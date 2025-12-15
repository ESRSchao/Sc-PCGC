"""Metrics for point cloud compression evaluation."""

import numpy as np
import torch
from typing import Dict
from scipy.spatial.distance import cdist


def chamfer_distance(
    points1: np.ndarray,
    points2: np.ndarray,
) -> float:
    """Compute Chamfer distance between two point clouds.
    
    Args:
        points1: First point cloud [N, 3]
        points2: Second point cloud [M, 3]
        
    Returns:
        Chamfer distance
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # Compute pairwise distances
    dist_matrix = cdist(points1, points2)
    
    # Forward: points1 to points2
    forward = np.mean(np.min(dist_matrix, axis=1))
    
    # Backward: points2 to points1
    backward = np.mean(np.min(dist_matrix, axis=0))
    
    return (forward + backward) / 2


def hausdorff_distance(
    points1: np.ndarray,
    points2: np.ndarray,
) -> float:
    """Compute Hausdorff distance between two point clouds.
    
    Args:
        points1: First point cloud [N, 3]
        points2: Second point cloud [M, 3]
        
    Returns:
        Hausdorff distance
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    dist_matrix = cdist(points1, points2)
    
    forward = np.max(np.min(dist_matrix, axis=1))
    backward = np.max(np.min(dist_matrix, axis=0))
    
    return max(forward, backward)


def point_to_point_psnr(
    points1: np.ndarray,
    points2: np.ndarray,
    peak: float = 1.0,
) -> float:
    """Compute point-to-point PSNR.
    
    Args:
        points1: Original point cloud [N, 3]
        points2: Reconstructed point cloud [M, 3]
        peak: Peak value for PSNR calculation
        
    Returns:
        PSNR value in dB
    """
    mse = chamfer_distance(points1, points2) ** 2
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10(peak ** 2 / mse)
    return psnr


def compute_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    bitrate: float = None,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        original: Original point cloud [N, 3]
        reconstructed: Reconstructed point cloud [M, 3]
        bitrate: Compression bitrate in bits (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Geometric metrics
    metrics['chamfer_distance'] = chamfer_distance(original, reconstructed)
    metrics['hausdorff_distance'] = hausdorff_distance(original, reconstructed)
    metrics['psnr'] = point_to_point_psnr(original, reconstructed)
    
    # Compression metrics
    metrics['num_original_points'] = len(original)
    metrics['num_reconstructed_points'] = len(reconstructed)
    metrics['compression_ratio'] = len(original) / max(len(reconstructed), 1)
    
    if bitrate is not None:
        metrics['bitrate'] = bitrate
        metrics['bpp'] = bitrate / len(original) if len(original) > 0 else 0
    
    return metrics


def compute_batch_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    encoded: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute metrics for a batch of point clouds.
    
    Args:
        original: Original point clouds [B, N, 3]
        reconstructed: Reconstructed point clouds [B, M, 3]
        encoded: Encoded representations
        
    Returns:
        Dictionary of averaged metrics
    """
    batch_size = original.shape[0]
    all_metrics = []
    
    for b in range(batch_size):
        orig = original[b].cpu().numpy()
        recon = reconstructed[b].cpu().numpy()
        
        # Filter out padding
        orig = orig[np.sum(orig, axis=-1) != 0]
        recon = recon[np.sum(recon, axis=-1) != 0]
        
        if len(orig) > 0 and len(recon) > 0:
            metrics = compute_metrics(orig, recon)
            all_metrics.append(metrics)
    
    # Average metrics
    if len(all_metrics) == 0:
        return {}
    
    averaged_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m and not np.isinf(m[key])]
        if values:
            averaged_metrics[key] = np.mean(values)
    
    return averaged_metrics
