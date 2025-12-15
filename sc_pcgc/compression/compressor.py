"""Point Cloud Compressor with Oct-Attention."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from ..models.encoder import PriorityOctreeEncoder
from ..models.decoder import PriorityOctreeDecoder
from ..data.preprocessing import (
    voxelize_point_cloud,
    compute_semantic_priorities,
)


class PointCloudCompressor(nn.Module):
    """Complete point cloud compression system with Oct-Attention.
    
    This system implements the full pipeline:
    1. Semantic segmentation (if needed)
    2. Voxelization
    3. Priority mapping
    4. Priority octree encoding with Oct-Attention
    5. Entropy coding (placeholder)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_depth: int = 10,
        voxel_size: float = 0.1,
        quantization_bits: int = 10,
        use_entropy_coding: bool = True,
    ):
        """Initialize Point Cloud Compressor.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of attention layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_depth: Maximum octree depth
            voxel_size: Voxel size for downsampling
            quantization_bits: Number of bits for quantization
            use_entropy_coding: Whether to use entropy coding
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        self.quantization_bits = quantization_bits
        self.use_entropy_coding = use_entropy_coding
        
        # Encoder and decoder
        self.encoder = PriorityOctreeEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_depth=max_depth,
            quantization_bits=quantization_bits,
        )
        
        self.decoder = PriorityOctreeDecoder(
            d_model=d_model,
            max_depth=max_depth,
        )
    
    def preprocess(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess point cloud.
        
        Pipeline: Voxelization -> Priority Mapping
        
        Args:
            points: Point cloud coordinates [N, 3]
            labels: Semantic labels [N] (optional)
            
        Returns:
            Tuple of:
                - processed_points: Voxelized points
                - priorities: Priority values
        """
        # Compute priorities from semantic labels
        if labels is not None:
            priorities = compute_semantic_priorities(labels)
        else:
            priorities = np.ones(len(points))
        
        # Voxelize point cloud
        voxelized_points, _, voxelized_priorities, _ = voxelize_point_cloud(
            points=points,
            voxel_size=self.voxel_size,
            labels=labels,
            priorities=priorities,
        )
        
        return voxelized_points, voxelized_priorities
    
    def compress(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        return_metrics: bool = False,
    ) -> Dict[str, any]:
        """Compress point cloud.
        
        Args:
            points: Point cloud coordinates [N, 3]
            labels: Semantic labels [N] (optional)
            return_metrics: Whether to return compression metrics
            
        Returns:
            Dictionary containing:
                - encoded: Encoded representation
                - bitrate: Compression bitrate (if return_metrics=True)
                - num_points: Number of points after preprocessing
        """
        # Preprocess
        processed_points, priorities = self.preprocess(points, labels)
        
        # Convert to tensors
        points_tensor = torch.from_numpy(processed_points).float().unsqueeze(0)
        priorities_tensor = torch.from_numpy(priorities).float().unsqueeze(0)
        
        # Encode
        with torch.no_grad():
            encoded = self.encoder(points_tensor, priorities_tensor)
        
        result = {
            'encoded': encoded,
            'num_points': len(processed_points),
        }
        
        if return_metrics:
            bitrate = self.encoder.compute_bitrate(encoded)
            result['bitrate'] = bitrate.item()
            result['bpp'] = bitrate.item() / len(processed_points) if len(processed_points) > 0 else 0
        
        return result
    
    def decompress(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        """Decompress point cloud.
        
        Args:
            encoded: Encoded representation
            
        Returns:
            Reconstructed point cloud [N, 3]
        """
        with torch.no_grad():
            reconstructed = self.decoder(encoded)
        
        # Convert to numpy
        if reconstructed.dim() > 2:
            reconstructed = reconstructed.squeeze(0)
        
        return reconstructed.cpu().numpy()
    
    def forward(
        self,
        points: torch.Tensor,
        priorities: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for training.
        
        Args:
            points: Point cloud coordinates [B, N, 3]
            priorities: Priority values [B, N] (optional)
            
        Returns:
            Tuple of:
                - reconstructed: Reconstructed point cloud [B, M, 3]
                - encoded: Encoded representation
        """
        # Encode
        encoded = self.encoder(points, priorities)
        
        # Decode
        reconstructed = self.decoder(encoded)
        
        return reconstructed, encoded
    
    def compute_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        encoded: Dict[str, torch.Tensor],
        lambda_rate: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute compression loss.
        
        Loss = Distortion + lambda * Rate
        
        Args:
            original: Original point cloud [B, N, 3]
            reconstructed: Reconstructed point cloud [B, M, 3]
            encoded: Encoded representation
            lambda_rate: Rate-distortion trade-off parameter
            
        Returns:
            Tuple of:
                - loss: Total loss
                - metrics: Dictionary of loss components
        """
        # Compute distortion (Chamfer distance approximation)
        # For simplicity, use MSE on nearest neighbors
        distortion = self._compute_distortion(original, reconstructed)
        
        # Compute rate
        rate = self.encoder.compute_bitrate(encoded)
        
        # Total loss
        loss = distortion + lambda_rate * rate
        
        metrics = {
            'loss': loss,
            'distortion': distortion,
            'rate': rate,
        }
        
        return loss, metrics
    
    def _compute_distortion(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distortion between point clouds.
        
        Args:
            original: Original point cloud [B, N, 3]
            reconstructed: Reconstructed point cloud [B, M, 3]
            
        Returns:
            Distortion value
        """
        # Simple MSE for now
        # In practice, use Chamfer distance or Earth Mover's Distance
        batch_size = original.shape[0]
        total_dist = 0.0
        
        for b in range(batch_size):
            orig = original[b]
            recon = reconstructed[b]
            
            # Filter out padding (zeros)
            orig = orig[orig.sum(dim=-1) != 0]
            recon = recon[recon.sum(dim=-1) != 0]
            
            if len(orig) == 0 or len(recon) == 0:
                continue
            
            # Compute pairwise distances
            dist_matrix = torch.cdist(orig.unsqueeze(0), recon.unsqueeze(0)).squeeze(0)
            
            # Chamfer distance (simplified)
            forward = torch.mean(torch.min(dist_matrix, dim=1)[0])
            backward = torch.mean(torch.min(dist_matrix, dim=0)[0])
            total_dist += (forward + backward) / 2
        
        return total_dist / batch_size if batch_size > 0 else torch.tensor(0.0)
