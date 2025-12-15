#!/usr/bin/env python3
"""Demo script showing basic usage of Sc-PCGC."""

import numpy as np
import torch

from sc_pcgc import PointCloudCompressor
from sc_pcgc.utils import compute_metrics, visualize_comparison


def create_synthetic_point_cloud(num_points=1000):
    """Create a synthetic point cloud for demonstration.
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        Point cloud array [N, 3]
    """
    # Create a simple cube with some noise
    points = np.random.uniform(-1, 1, (num_points, 3)).astype(np.float32)
    
    # Add some structure (a denser region representing a "car")
    car_points = np.random.normal(0, 0.2, (num_points // 4, 3)).astype(np.float32)
    car_points[:, 2] += 0.5  # Shift up
    
    points = np.vstack([points, car_points])
    
    return points


def main():
    """Main demo function."""
    print("=" * 60)
    print("Sc-PCGC Demo: Point Cloud Compression with Oct-Attention")
    print("=" * 60)
    
    # Create synthetic point cloud
    print("\n1. Creating synthetic point cloud...")
    points = create_synthetic_point_cloud(num_points=1000)
    print(f"   Created point cloud with {len(points)} points")
    
    # Create compressor
    print("\n2. Initializing compressor...")
    compressor = PointCloudCompressor(
        d_model=128,        # Smaller model for demo
        n_heads=4,
        n_layers=3,
        d_ff=512,
        dropout=0.1,
        max_depth=8,        # Shallower tree for demo
        voxel_size=0.1,
    )
    print("   Compressor initialized")
    
    # Compress
    print("\n3. Compressing point cloud...")
    compressed = compressor.compress(points, return_metrics=True)
    print(f"   Original points: {len(points)}")
    print(f"   Processed points: {compressed['num_points']}")
    print(f"   Bitrate: {compressed['bitrate']:.2f} bits")
    print(f"   Bits per point: {compressed['bpp']:.4f}")
    
    # Decompress
    print("\n4. Decompressing...")
    reconstructed = compressor.decompress(compressed['encoded'])
    print(f"   Reconstructed {len(reconstructed)} points")
    
    # Compute metrics
    print("\n5. Computing quality metrics...")
    metrics = compute_metrics(points, reconstructed, compressed['bitrate'])
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key:30s}: {value:.4f}")
    
    # Visualize
    print("\n6. Generating visualization...")
    visualize_comparison(
        points,
        reconstructed,
        save_path="demo_comparison.png"
    )
    print("   Saved visualization to: demo_comparison.png")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
