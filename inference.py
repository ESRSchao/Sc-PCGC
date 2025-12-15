#!/usr/bin/env python3
"""Inference script for Sc-PCGC."""

import os
import argparse
import numpy as np
import torch

from sc_pcgc.config import Config, load_config
from sc_pcgc.compression import PointCloudCompressor
from sc_pcgc.utils import compute_metrics, visualize_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with Sc-PCGC')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input point cloud (.bin or .npy)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    return parser.parse_args()


def load_point_cloud(filepath: str) -> np.ndarray:
    """Load point cloud from file.
    
    Args:
        filepath: Path to point cloud file
        
    Returns:
        Point cloud array [N, 3] or [N, 4]
    """
    ext = os.path.splitext(filepath)[1]
    
    if ext == '.bin':
        # SemanticKITTI format
        points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]  # Return only XYZ
    elif ext == '.npy':
        points = np.load(filepath)
        if points.shape[1] > 3:
            return points[:, :3]
        return points
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_point_cloud(points: np.ndarray, filepath: str):
    """Save point cloud to file.
    
    Args:
        points: Point cloud array [N, 3]
        filepath: Output file path
    """
    ext = os.path.splitext(filepath)[1]
    
    if ext == '.bin':
        # Save as binary (SemanticKITTI format with dummy intensity)
        intensity = np.zeros((len(points), 1), dtype=np.float32)
        data = np.hstack([points, intensity]).astype(np.float32)
        data.tofile(filepath)
    elif ext == '.npy':
        np.save(filepath, points)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config(config_dict)
    else:
        config = Config()
    
    # Create model
    print("Creating model...")
    model = PointCloudCompressor(
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        max_depth=config.octree.max_depth,
        voxel_size=config.octree.voxel_size,
        quantization_bits=config.compression.quantization_bits,
        use_entropy_coding=config.compression.use_entropy_coding,
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load input point cloud
    print(f"Loading point cloud from {args.input}")
    points = load_point_cloud(args.input)
    print(f"Loaded {len(points)} points")
    
    # Compress
    print("Compressing...")
    compressed = model.compress(points, return_metrics=True)
    
    print(f"Original points: {len(points)}")
    print(f"Processed points: {compressed['num_points']}")
    print(f"Bitrate: {compressed['bitrate']:.2f} bits")
    print(f"Bits per point: {compressed['bpp']:.4f}")
    
    # Decompress
    print("Decompressing...")
    reconstructed = model.decompress(compressed['encoded'])
    print(f"Reconstructed {len(reconstructed)} points")
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(points, reconstructed, compressed['bitrate'])
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    output_basename = os.path.splitext(os.path.basename(args.input))[0]
    
    # Save reconstructed point cloud
    output_path = os.path.join(args.output, f"{output_basename}_reconstructed.npy")
    save_point_cloud(reconstructed, output_path)
    print(f"\nSaved reconstructed point cloud to {output_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output, f"{output_basename}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Compression Metrics\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Saved metrics to {metrics_path}")
    
    # Visualize if requested
    if args.visualize:
        print("Generating visualization...")
        vis_path = os.path.join(args.output, f"{output_basename}_comparison.png")
        visualize_comparison(points, reconstructed, save_path=vis_path)
        print(f"Saved visualization to {vis_path}")
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()
