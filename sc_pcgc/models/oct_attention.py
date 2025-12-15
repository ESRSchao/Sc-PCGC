"""Oct-Attention: Transformer-based attention mechanism for octree nodes."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for octree nodes."""
    
    def __init__(self, d_model: int, max_depth: int = 10):
        """Initialize 3D positional encoding.
        
        Args:
            d_model: Model dimension
            max_depth: Maximum octree depth
        """
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth
        
        # Learnable depth embeddings
        self.depth_embedding = nn.Embedding(max_depth + 1, d_model)
        
        # Learnable octant embeddings (8 octants)
        self.octant_embedding = nn.Embedding(8, d_model)
        
        # Spatial position encoding using sinusoidal functions
        self.register_buffer('freq', self._get_frequencies(d_model))
    
    def _get_frequencies(self, d_model: int) -> torch.Tensor:
        """Get frequency values for sinusoidal encoding.
        
        Args:
            d_model: Model dimension
            
        Returns:
            Frequency tensor
        """
        freq = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        return freq
    
    def forward(
        self,
        positions: torch.Tensor,
        depths: torch.Tensor,
        octants: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute positional encoding.
        
        Args:
            positions: Node center positions [B, N, 3]
            depths: Node depth levels [B, N]
            octants: Node octant indices [B, N] (optional)
            
        Returns:
            Positional encodings [B, N, d_model]
        """
        batch_size, num_nodes, _ = positions.shape
        device = positions.device
        
        # Depth encoding
        depth_enc = self.depth_embedding(depths)  # [B, N, d_model]
        
        # Spatial position encoding using sinusoidal functions
        # Use standard sinusoidal encoding across all dimensions
        pos_enc = torch.zeros(batch_size, num_nodes, self.d_model, device=device)
        
        # Flatten positions for encoding
        positions_flat = positions.reshape(batch_size, num_nodes, 3)
        
        # Apply sinusoidal encoding
        for dim_idx in range(self.d_model // 2):
            freq = self.freq[dim_idx]
            for coord_idx in range(3):
                pos_coord = positions_flat[:, :, coord_idx]
                angles = pos_coord * freq
                pos_enc[:, :, dim_idx * 2] += torch.sin(angles) / 3.0
                pos_enc[:, :, dim_idx * 2 + 1] += torch.cos(angles) / 3.0
        
        # Combine encodings
        encoding = depth_enc + pos_enc
        
        # Add octant encoding if provided
        if octants is not None:
            octant_enc = self.octant_embedding(octants)
            encoding = encoding + octant_enc
        
        return encoding


class OctAttention(nn.Module):
    """Oct-Attention mechanism for octree nodes.
    
    This implements attention between octree nodes, considering their
    spatial relationships and hierarchical structure.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize Oct-Attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Relative position bias
        self.relative_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Oct-Attention.
        
        Args:
            x: Input features [B, N, d_model]
            mask: Attention mask [B, N, N] (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
                - output: Output features [B, N, d_model]
                - attention_weights: Attention weights [B, n_heads, N, N] (if requested)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(x).view(batch_size, num_nodes, self.n_heads, self.d_head)
        K = self.W_k(x).view(batch_size, num_nodes, self.n_heads, self.d_head)
        V = self.W_v(x).view(batch_size, num_nodes, self.n_heads, self.d_head)
        
        # Transpose for attention computation: [B, n_heads, N, d_head]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, N, N]
        
        # Add relative position bias
        scores = scores + self.relative_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [B, n_heads, N, d_head]
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous()  # [B, N, n_heads, d_head]
        context = context.view(batch_size, num_nodes, self.d_model)
        
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


class MultiHeadOctAttention(nn.Module):
    """Multi-Head Oct-Attention with feed-forward network."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        """Initialize Multi-Head Oct-Attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Oct-Attention
        self.attention = OctAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features [B, N, d_model]
            mask: Attention mask [B, N, N] (optional)
            
        Returns:
            Output features [B, N, d_model]
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class OctAttentionEncoder(nn.Module):
    """Octree encoder using stacked Oct-Attention layers."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_depth: int = 10,
    ):
        """Initialize Oct-Attention Encoder.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of attention layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_depth: Maximum octree depth
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(d_model, max_depth)
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)  # Project occupancy to d_model
        
        # Stacked Oct-Attention layers
        self.layers = nn.ModuleList([
            MultiHeadOctAttention(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 256)  # Context features for entropy coding
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        occupancy: torch.Tensor,
        positions: torch.Tensor,
        depths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            occupancy: Node occupancy patterns [B, N, 1]
            positions: Node positions [B, N, 3]
            depths: Node depths [B, N]
            mask: Attention mask [B, N, N] (optional)
            
        Returns:
            Context features [B, N, 256]
        """
        # Project occupancy to model dimension
        x = self.input_proj(occupancy)  # [B, N, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoding(positions, depths)
        x = x + pos_enc
        x = self.dropout(x)
        
        # Apply stacked Oct-Attention layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Project to context features
        context = self.output_proj(x)
        
        return context
