"""Visualization utilities for point clouds."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional


def visualize_point_cloud(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Point Cloud",
    point_size: int = 1,
    figsize: tuple = (10, 10),
):
    """Visualize point cloud.
    
    Args:
        points: Point cloud coordinates [N, 3]
        labels: Optional semantic labels [N]
        save_path: Path to save figure (optional)
        title: Figure title
        point_size: Size of points
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        # Color by labels
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=labels,
            s=point_size,
            cmap='tab20',
        )
        plt.colorbar(scatter, ax=ax, label='Semantic Label')
    else:
        # Color by height (z-coordinate)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            s=point_size,
            cmap='viridis',
        )
        plt.colorbar(scatter, ax=ax, label='Height')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 8),
):
    """Visualize original and reconstructed point clouds side by side.
    
    Args:
        original: Original point cloud [N, 3]
        reconstructed: Reconstructed point cloud [M, 3]
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(
        original[:, 0],
        original[:, 1],
        original[:, 2],
        c=original[:, 2],
        s=1,
        cmap='viridis',
    )
    ax1.set_title(f'Original ({len(original)} points)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Reconstructed
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        reconstructed[:, 0],
        reconstructed[:, 1],
        reconstructed[:, 2],
        c=reconstructed[:, 2],
        s=1,
        cmap='viridis',
    )
    ax2.set_title(f'Reconstructed ({len(reconstructed)} points)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Match view angles
    ax2.view_init(elev=ax1.elev, azim=ax1.azim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
