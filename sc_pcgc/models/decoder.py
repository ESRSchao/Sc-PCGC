"""Priority Octree Decoder for decompression."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from .octree import Octree, OctreeNode


class PriorityOctreeDecoder(nn.Module):
    """Priority-based octree decoder for point cloud decompression."""
    
    def __init__(
        self,
        d_model: int = 256,
        max_depth: int = 10,
    ):
        """Initialize Priority Octree Decoder.
        
        Args:
            d_model: Model dimension
            max_depth: Maximum octree depth
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_depth = max_depth
        
        # Context decoder
        self.context_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8),  # Decode to occupancy pattern
        )
    
    def decode_octree(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Octree:
        """Decode octree from encoded representation.
        
        Args:
            encoded: Encoded representation containing:
                - occupancy: Occupancy patterns
                - contexts: Context features (optional)
                - depths: Node depths
                
        Returns:
            Reconstructed octree
        """
        occupancy = encoded['occupancy']
        depths = encoded['depths']
        
        # Handle batched input
        if isinstance(occupancy, list):
            occupancy = occupancy[0]
            depths = depths[0]
        
        if occupancy.dim() > 1:
            occupancy = occupancy.squeeze(0)
            depths = depths.squeeze(0)
        
        # Create empty octree
        octree = Octree(max_depth=self.max_depth)
        
        # Initialize root with default bounds
        center = np.array([0.0, 0.0, 0.0])
        size = 100.0  # Default size
        octree.root = OctreeNode(center=center, size=size, depth=0)
        octree.nodes = [octree.root]
        
        # Reconstruct tree structure from occupancy patterns
        self._reconstruct_tree(octree.root, occupancy, depths, 0)
        
        return octree
    
    def _reconstruct_tree(
        self,
        node: OctreeNode,
        occupancy: torch.Tensor,
        depths: torch.Tensor,
        idx: int,
    ) -> int:
        """Recursively reconstruct octree structure.
        
        Args:
            node: Current node
            occupancy: Occupancy patterns
            depths: Node depths
            idx: Current index in encoded sequence
            
        Returns:
            Next index in sequence
        """
        if idx >= len(occupancy) or node.depth >= self.max_depth:
            return idx
        
        # Get occupancy pattern for current node
        occ = int(occupancy[idx].item())
        
        if occ == 0:
            # Empty node
            return idx + 1
        
        # Subdivide if needed
        node.subdivide()
        idx += 1
        
        # Recursively process children based on occupancy
        for i in range(8):
            if occ & (1 << i):
                child = node.children[i]
                if child is not None:
                    idx = self._reconstruct_tree(child, occupancy, depths, idx)
        
        return idx
    
    def decode_to_points(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Decode to point cloud.
        
        Args:
            encoded: Encoded representation
            
        Returns:
            Reconstructed point cloud [N, 3]
        """
        # Decode octree
        octree = self.decode_octree(encoded)
        
        # Extract points from leaf nodes
        points = []
        for node in octree.get_leaf_nodes():
            if len(node.points) > 0 or node.occupancy > 0:
                # Use node center as reconstructed point
                points.append(node.center)
        
        if len(points) == 0:
            return torch.zeros(0, 3)
        
        points = np.array(points)
        return torch.from_numpy(points).float()
    
    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass: decode to point cloud.
        
        Args:
            encoded: Encoded representation
            
        Returns:
            Reconstructed point cloud [B, N, 3]
        """
        # Handle batched input
        if isinstance(encoded['occupancy'], list):
            results = []
            for i in range(len(encoded['occupancy'])):
                batch_encoded = {
                    key: value[i] if isinstance(value, list) else value[i:i+1]
                    for key, value in encoded.items()
                }
                points = self.decode_to_points(batch_encoded)
                results.append(points)
            
            # Pad to same length for batching
            max_len = max(len(pts) for pts in results)
            batched = torch.zeros(len(results), max_len, 3)
            for i, pts in enumerate(results):
                batched[i, :len(pts)] = pts
            
            return batched
        else:
            points = self.decode_to_points(encoded)
            return points.unsqueeze(0)


class OctreeReconstructor:
    """Utility class for reconstructing point clouds from octrees."""
    
    @staticmethod
    def octree_to_points(
        octree: Octree,
        sampling_method: str = "center",
    ) -> np.ndarray:
        """Convert octree to point cloud.
        
        Args:
            octree: Input octree
            sampling_method: Method for sampling points ("center" or "uniform")
            
        Returns:
            Point cloud array [N, 3]
        """
        points = []
        
        for node in octree.get_leaf_nodes():
            if len(node.points) > 0 or node.occupancy > 0:
                if sampling_method == "center":
                    # Use node center
                    points.append(node.center)
                elif sampling_method == "uniform":
                    # Sample uniformly within node bounds
                    half_size = node.size / 2
                    offset = np.random.uniform(-half_size, half_size, 3)
                    points.append(node.center + offset)
        
        if len(points) == 0:
            return np.zeros((0, 3))
        
        return np.array(points)
    
    @staticmethod
    def compute_reconstruction_error(
        original: np.ndarray,
        reconstructed: np.ndarray,
        metric: str = "chamfer",
    ) -> float:
        """Compute reconstruction error.
        
        Args:
            original: Original point cloud [N, 3]
            reconstructed: Reconstructed point cloud [M, 3]
            metric: Distance metric ("chamfer" or "hausdorff")
            
        Returns:
            Reconstruction error
        """
        if len(original) == 0 or len(reconstructed) == 0:
            return float('inf')
        
        if metric == "chamfer":
            # Chamfer distance (simplified)
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(original, reconstructed)
            forward = np.mean(np.min(dist_matrix, axis=1))
            backward = np.mean(np.min(dist_matrix, axis=0))
            return (forward + backward) / 2
        elif metric == "hausdorff":
            # Hausdorff distance
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(original, reconstructed)
            forward = np.max(np.min(dist_matrix, axis=1))
            backward = np.max(np.min(dist_matrix, axis=0))
            return max(forward, backward)
        else:
            raise ValueError(f"Unknown metric: {metric}")
