"""Priority Octree Encoder with Oct-Attention."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from .octree import Octree, OctreeNode
from .oct_attention import OctAttentionEncoder


class PriorityOctreeEncoder(nn.Module):
    """Priority-based octree encoder using Oct-Attention.
    
    This encoder processes octree nodes in priority order based on semantic
    importance, using transformer-based attention to capture context.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_depth: int = 10,
        quantization_bits: int = 10,
    ):
        """Initialize Priority Octree Encoder.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of attention layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_depth: Maximum octree depth
            quantization_bits: Number of bits for coordinate quantization
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_depth = max_depth
        self.quantization_bits = quantization_bits
        
        # Oct-Attention encoder
        self.oct_attention = OctAttentionEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_depth=max_depth,
        )
        
        # Context model for entropy coding
        self.context_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # Output: probability parameters
        )
        
        # Occupancy prediction head
        self.occupancy_head = nn.Linear(256, 8)  # Predict 8-bit occupancy
    
    def encode_octree(
        self,
        octree: Octree,
        return_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode octree structure.
        
        Args:
            octree: Input octree
            return_context: Whether to return context features
            
        Returns:
            Dictionary containing:
                - occupancy: Encoded occupancy patterns
                - contexts: Context features (if requested)
                - depths: Node depths
                - priorities: Node priorities
        """
        # Get nodes in priority order
        nodes = octree.traverse_priority_order()
        
        if len(nodes) == 0:
            return {
                'occupancy': torch.zeros(1, 0, dtype=torch.long),
                'contexts': torch.zeros(1, 0, self.d_model) if return_context else None,
                'depths': torch.zeros(1, 0, dtype=torch.long),
                'priorities': torch.zeros(1, 0),
            }
        
        # Extract node features (convert to numpy first for efficiency)
        occupancy_np = np.array([node.compute_occupancy() for node in nodes], dtype=np.float32)
        occupancy = torch.from_numpy(occupancy_np).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        
        positions_np = np.array([node.center for node in nodes], dtype=np.float32)
        positions = torch.from_numpy(positions_np).unsqueeze(0)  # [1, N, 3]
        
        depths_np = np.array([node.depth for node in nodes], dtype=np.int64)
        depths = torch.from_numpy(depths_np).unsqueeze(0)  # [1, N]
        
        priorities_np = np.array([node.priority for node in nodes], dtype=np.float32)
        priorities = torch.from_numpy(priorities_np).unsqueeze(0)  # [1, N]
        
        # Encode using Oct-Attention
        context_features = self.oct_attention(occupancy, positions, depths)
        
        # Get context for entropy coding
        contexts = self.context_model(context_features)
        
        result = {
            'occupancy': (occupancy.squeeze(-1) > 0).long(),  # [1, N]
            'depths': depths,
            'priorities': priorities,
            'positions': positions,  # Include positions for reconstruction
        }
        
        if return_context:
            result['contexts'] = contexts
        
        return result
    
    def forward(
        self,
        points: torch.Tensor,
        priorities: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: encode point cloud.
        
        Args:
            points: Point cloud coordinates [B, N, 3]
            priorities: Priority values [B, N] (optional)
            
        Returns:
            Dictionary containing encoded representations
        """
        batch_size = points.shape[0]
        results = []
        
        for b in range(batch_size):
            # Convert to numpy for octree construction
            pts = points[b].cpu().numpy()
            prio = priorities[b].cpu().numpy() if priorities is not None else None
            
            # Build octree
            octree = Octree(max_depth=self.max_depth)
            octree.build(pts, prio)
            
            # Encode octree
            encoded = self.encode_octree(octree, return_context=True)
            results.append(encoded)
        
        # Batch results
        batched_results = {
            'occupancy': [r['occupancy'] for r in results],
            'contexts': [r['contexts'] for r in results],
            'depths': [r['depths'] for r in results],
            'priorities': [r['priorities'] for r in results],
            'positions': [r['positions'] for r in results],
        }
        
        return batched_results
    
    def compute_bitrate(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute bitrate of encoded representation.
        
        Args:
            encoded: Encoded representation
            
        Returns:
            Bitrate in bits
        """
        # Simple bitrate estimation: count occupied nodes
        occupancy = encoded['occupancy']
        if isinstance(occupancy, list):
            total_bits = sum(occ.sum().item() * 8 for occ in occupancy)  # 8 bits per occupancy
        else:
            total_bits = occupancy.sum().item() * 8
        
        return torch.tensor(total_bits, dtype=torch.float32)


class EntropyModel(nn.Module):
    """Entropy model for learned compression.
    
    This model estimates probability distributions for entropy coding.
    """
    
    def __init__(self, context_dim: int = 256):
        """Initialize entropy model.
        
        Args:
            context_dim: Dimension of context features
        """
        super().__init__()
        
        self.context_dim = context_dim
        
        # Probability estimation network
        self.prob_net = nn.Sequential(
            nn.Linear(context_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # Output: probability parameters
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Estimate probability parameters.
        
        Args:
            context: Context features [B, N, context_dim]
            
        Returns:
            Probability parameters [B, N, 256]
        """
        return self.prob_net(context)
    
    def compute_rate(
        self,
        symbols: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rate (negative log-likelihood).
        
        Args:
            symbols: Symbols to encode [B, N]
            context: Context features [B, N, context_dim]
            
        Returns:
            Rate in bits [B, N]
        """
        # Get probability parameters
        prob_params = self.forward(context)
        
        # Simple uniform distribution for now
        # In practice, use learned distributions
        prob = torch.sigmoid(prob_params.mean(dim=-1))
        
        # Compute negative log-likelihood
        rate = -torch.log2(prob + 1e-10)
        
        return rate
