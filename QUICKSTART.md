# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/ESRSchao/Sc-PCGC.git
cd Sc-PCGC

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Try the Demo (No Dataset Required)

```bash
# Run demo with synthetic data
python examples/demo.py
```

This will:
1. Generate a synthetic point cloud (1250 points)
2. Compress it using Oct-Attention
3. Decompress and compute metrics
4. Save visualization to `demo_comparison.png`

Expected output:
```
Original points:         1250
Processed points:        ~1100
Reconstructed points:    ~260
Compression ratio:       ~4.7x
Bits per point:          ~1.7 bpp
Chamfer distance:        ~0.11
PSNR:                    ~19 dB
```

## Use with Your Data

### Python API

```python
from sc_pcgc import PointCloudCompressor
import numpy as np

# Load your point cloud [N, 3]
points = np.load('your_pointcloud.npy')

# Create compressor
compressor = PointCloudCompressor(
    d_model=256,      # Model dimension
    n_heads=8,        # Attention heads
    n_layers=6,       # Transformer layers
    max_depth=10,     # Octree depth
    voxel_size=0.1,   # Voxelization size (meters)
)

# Compress
result = compressor.compress(points, return_metrics=True)
print(f"Bitrate: {result['bitrate']:.2f} bits")
print(f"BPP: {result['bpp']:.4f}")

# Decompress
reconstructed = compressor.decompress(result['encoded'])
print(f"Reconstructed {len(reconstructed)} points")
```

### Command Line

```bash
# Compress a point cloud
python inference.py \
    --checkpoint checkpoints/model.pth \
    --input data/pointcloud.bin \
    --output results/ \
    --visualize
```

## Train on SemanticKITTI

### 1. Download Dataset

Download SemanticKITTI from http://semantic-kitti.org/dataset.html

Organize as:
```
data/SemanticKITTI/
  sequences/
    00/
      velodyne/
        000000.bin
        ...
      labels/
        000000.label
        ...
    01/
    ...
```

### 2. Configure

Edit `example_config.yaml`:
```yaml
data:
  data_root: ./data/SemanticKITTI

model:
  d_model: 256
  n_heads: 8
  n_layers: 6

octree:
  max_depth: 10
  voxel_size: 0.1

training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 100
```

### 3. Train

```bash
python train.py \
    --config example_config.yaml \
    --data-root ./data/SemanticKITTI \
    --output-dir ./checkpoints \
    --log-dir ./logs
```

Monitor training:
```bash
tensorboard --logdir ./logs
```

## Key Parameters

### Model Size
- **Small**: d_model=128, n_heads=4, n_layers=3
- **Medium**: d_model=256, n_heads=8, n_layers=6 (default)
- **Large**: d_model=512, n_heads=8, n_layers=12

### Compression Quality
- **High Quality**: voxel_size=0.05, max_depth=12
- **Balanced**: voxel_size=0.1, max_depth=10 (default)
- **High Compression**: voxel_size=0.2, max_depth=8

### Semantic Priorities
Edit in config to prioritize different objects:
```yaml
priority:
  semantic_weights:
    car: 1.0          # High priority
    pedestrian: 1.0   # High priority
    vegetation: 0.3   # Low priority
```

## Testing

Run unit tests:
```bash
python -m unittest discover tests -v
```

All tests should pass:
```
test_octree_creation ... ok
test_octree_build ... ok
test_voxelization ... ok
test_priority_computation ... ok
test_oct_attention ... ok
test_encoder ... ok
test_decoder ... ok
test_compressor_creation ... ok
test_compression_pipeline ... ok
----------------------------------------------------------------------
Ran 10 tests ... OK
```

## Troubleshooting

### Out of Memory
- Reduce batch_size (default: 4 → 1)
- Reduce d_model (256 → 128)
- Increase voxel_size (0.1 → 0.2)

### Poor Quality
- Decrease voxel_size (0.1 → 0.05)
- Increase max_depth (10 → 12)
- Adjust lambda_rate in training

### Slow Training
- Reduce n_layers (6 → 3)
- Reduce max_depth (10 → 8)
- Use GPU (--device cuda)

## Next Steps

1. Train on full SemanticKITTI dataset
2. Evaluate on test sequences
3. Compare with baselines (G-PCC, V-PCC)
4. Tune hyperparameters for your use case
5. Integrate entropy coding for actual compression

## Support

- Documentation: See README.md and ARCHITECTURE.md
- Issues: GitHub Issues
- Examples: examples/ directory

## License

MIT License - See LICENSE file
