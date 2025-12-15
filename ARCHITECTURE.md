# Sc-PCGC Architecture

This document provides detailed information about the architecture and implementation of the Scalable Point Cloud Geometry Compression (Sc-PCGC) system.

## System Overview

The Sc-PCGC system implements a deep learning-based point cloud compression pipeline using Oct-Attention, a transformer-based attention mechanism designed specifically for octree structures.

## Core Components

### 1. Data Pipeline (`sc_pcgc/data/`)

#### SemanticKITTI Dataset Loader
- **File**: `dataset.py`
- **Purpose**: Loads point clouds and semantic labels from the SemanticKITTI dataset
- **Features**:
  - Supports train/val/test splits
  - Loads `.bin` format point clouds
  - Extracts semantic labels from `.label` files
  - Provides 20-class semantic segmentation labels

#### Preprocessing Module
- **File**: `preprocessing.py`
- **Key Functions**:
  - `voxelize_point_cloud()`: Downsamples point cloud using voxel grid
  - `compute_semantic_priorities()`: Maps semantic labels to importance weights
  - `SemanticSegmentation`: Interface for segmentation models (placeholder for RangeNet++, SalsaNext, etc.)

**Voxelization Process**:
1. Compute voxel indices for each point
2. Group points by voxel
3. Aggregate properties (average position, majority label, max priority)
4. Output downsampled representation

**Priority Mapping**:
- High priority (1.0): Dynamic objects (cars, pedestrians, cyclists)
- Medium priority (0.4-0.5): Static structures (buildings, roads)
- Low priority (0.2-0.3): Background (vegetation, terrain)

### 2. Octree Structure (`sc_pcgc/models/octree.py`)

#### OctreeNode
- Represents a cubic region in 3D space
- Properties:
  - `center`: 3D coordinates of node center
  - `size`: Edge length of the cube
  - `depth`: Level in the tree hierarchy
  - `children`: 8 child nodes (octants)
  - `priority`: Semantic importance value
  - `occupancy`: 8-bit pattern indicating occupied children
  - `features`: Neural network features

#### Octree
- Hierarchical spatial data structure
- Build process:
  1. Compute bounding box of point cloud
  2. Create root node
  3. Recursively subdivide based on point count and depth
  4. Assign priorities to nodes
- Traversal methods:
  - Breadth-first: Level-by-level traversal
  - Priority-order: High-to-low importance

### 3. Oct-Attention (`sc_pcgc/models/oct_attention.py`)

#### PositionalEncoding3D
Encodes spatial and hierarchical information:
- **Depth encoding**: Learned embedding for tree level
- **Spatial encoding**: Sinusoidal encoding of 3D position
- **Octant encoding**: Learned embedding for octant index

#### OctAttention
Core attention mechanism:
- Multi-head self-attention over octree nodes
- Query, Key, Value projections
- Relative position bias
- Attention scores: `Attention(Q,K,V) = softmax(QK^T/√d_k + bias)V`

#### MultiHeadOctAttention
Complete transformer block:
- Oct-Attention layer with residual connection
- Layer normalization
- Feed-forward network (FFN) with residual
- Structure: `x → Attention → +x → Norm → FFN → +x → Norm`

#### OctAttentionEncoder
Stacked transformer encoder:
- Input: Node occupancy patterns, positions, depths
- Processing: Multiple attention layers
- Output: Context features for entropy coding

### 4. Encoder (`sc_pcgc/models/encoder.py`)

#### PriorityOctreeEncoder
Compression encoder with enhancement layer:

**Encoding Pipeline**:
1. Build priority octree from input points
2. Extract node features (occupancy, position, depth)
3. Process through Oct-Attention encoder
4. Generate context features for entropy coding

**Output**:
- Occupancy patterns (8 bits per node)
- Context features (256-D vectors)
- Node depths and priorities
- Node positions

#### EntropyModel
Estimates probability distributions for compression:
- Takes context features as input
- Outputs probability parameters
- Used for entropy coding (arithmetic coding, range coding, etc.)

### 5. Decoder (`sc_pcgc/models/decoder.py`)

#### PriorityOctreeDecoder
Decompression decoder:

**Decoding Pipeline**:
1. Receive encoded representation
2. Reconstruct octree structure from occupancy
3. Extract node centers as point coordinates
4. Output reconstructed point cloud

**Reconstruction Strategy**:
- Use encoded positions directly (most accurate)
- Fallback: Reconstruct tree and extract leaf centers

### 6. Compression System (`sc_pcgc/compression/compressor.py`)

#### PointCloudCompressor
Complete end-to-end system:

**Compression Flow**:
```
Input → Voxelize → Priority Mapping → Octree → Encode → Compress
```

**Decompression Flow**:
```
Compressed → Decode → Octree → Extract Points → Output
```

**Loss Function**:
```
L = D(P, P̂) + λR
```
where:
- D: Distortion (Chamfer distance)
- P: Original point cloud
- P̂: Reconstructed point cloud
- R: Bitrate
- λ: Rate-distortion trade-off parameter

## Model Parameters

### Default Configuration

```yaml
Model:
  d_model: 256        # Transformer dimension
  n_heads: 8          # Attention heads
  n_layers: 6         # Transformer layers
  d_ff: 1024         # FFN dimension
  dropout: 0.1

Octree:
  max_depth: 10       # Tree depth (affects resolution)
  voxel_size: 0.1     # Downsampling resolution (meters)
  min_points: 1       # Minimum points per leaf

Compression:
  quantization_bits: 10   # Coordinate quantization
  use_entropy_coding: true
```

## Training Process

1. **Data Loading**: Load point clouds with semantic labels
2. **Preprocessing**: Voxelize and compute priorities
3. **Forward Pass**: Encode → Decode
4. **Loss Computation**: Distortion + Rate
5. **Backpropagation**: Update encoder/decoder weights
6. **Validation**: Evaluate on held-out data

## Evaluation Metrics

### Geometric Quality
- **Chamfer Distance**: Average nearest-neighbor distance
- **Hausdorff Distance**: Maximum nearest-neighbor distance
- **PSNR**: Peak Signal-to-Noise Ratio

### Compression Efficiency
- **Bitrate**: Total bits used
- **BPP (Bits Per Point)**: Average bits per point
- **Compression Ratio**: Original size / Compressed size

## Key Features

1. **Semantic-Aware**: Prioritizes important regions (vehicles, pedestrians)
2. **Hierarchical**: Multi-scale representation via octree
3. **Learned Compression**: Neural network-based context modeling
4. **Scalable**: Adjustable quality via depth and voxel size

## Implementation Notes

### Tensor Shapes
Throughout the codebase:
- Batch: `B`
- Number of points: `N`
- Model dimension: `d_model`
- Point coordinates: 3 (x, y, z)

Common shapes:
- Points: `[B, N, 3]`
- Features: `[B, N, d_model]`
- Occupancy: `[B, num_nodes]`

### Performance Considerations

1. **Memory**: Proportional to number of octree nodes
2. **Computation**: O(N²) attention over nodes
3. **Voxelization**: Reduces point count for efficiency
4. **Batch Processing**: Small batch sizes (typically 1-4) due to variable-length octrees

## Future Enhancements

1. **Better Entropy Coding**: Integrate arithmetic/ANS coding
2. **Learned Quantization**: Differentiable quantization
3. **Multi-Scale**: Encode different octree levels separately
4. **Conditional**: Use semantic labels as conditions
5. **Real-time**: Optimize for inference speed

## References

- Oct-Attention: Transformer-based attention for octrees
- SemanticKITTI: Large-scale outdoor LiDAR dataset
- Point Cloud Compression: G-PCC, V-PCC standards
- Octree: Hierarchical spatial data structure
