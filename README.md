# Sc-PCGC: Scalable Point Cloud Geometry Compression

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning-based point cloud compression system using **Oct-Attention** (Octree-based Transformer Attention) for semantic-aware geometry compression. This implementation focuses on the SemanticKITTI dataset and prioritizes important semantic regions for better rate-distortion performance.

## 🌟 Features

- **Oct-Attention Mechanism**: Transformer-based attention for octree nodes
- **Priority-Based Encoding**: Semantic importance-driven compression
- **SemanticKITTI Support**: Native support for autonomous driving datasets
- **End-to-End Pipeline**: Complete workflow from preprocessing to compression
- **Flexible Architecture**: Configurable model parameters and compression settings

## 📋 Overview

### Architecture

The system implements a complete point cloud compression pipeline:

```
Input Point Cloud (SemanticKITTI)
    ↓
Semantic Segmentation (labels from dataset)
    ↓
Voxelization (downsampling)
    ↓
Priority Mapping (semantic importance)
    ↓
Octree Construction (spatial hierarchy)
    ↓
Oct-Attention Encoder (Transformer-based)
    ↓
Context Model (for entropy coding)
    ↓
Compressed Bitstream
    ↓
Octree Decoder
    ↓
Reconstructed Point Cloud
```

### Key Components

1. **Preprocessing Module** (`sc_pcgc/data/`)
   - SemanticKITTI dataset loader
   - Voxelization for downsampling
   - Priority mapping based on semantic labels

2. **Octree Structure** (`sc_pcgc/models/octree.py`)
   - Priority-based octree construction
   - Spatial hierarchy representation
   - Efficient node traversal

3. **Oct-Attention** (`sc_pcgc/models/oct_attention.py`)
   - Multi-head transformer attention for octree nodes
   - 3D positional encoding (depth, spatial position, octant)
   - Context-aware feature learning

4. **Encoder/Decoder** (`sc_pcgc/models/`)
   - Priority octree encoder with enhancement layer
   - Context model for entropy coding
   - Octree decoder for reconstruction

5. **Compression System** (`sc_pcgc/compression/`)
   - End-to-end compression pipeline
   - Rate-distortion optimization
   - Configurable compression parameters

## 🚀 Installation

### Requirements

- Python 3.7+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ESRSchao/Sc-PCGC.git
cd Sc-PCGC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## 📊 Dataset Preparation

### SemanticKITTI

1. Download the SemanticKITTI dataset from [official website](http://semantic-kitti.org/dataset.html)

2. Extract and organize the data:
```
data/SemanticKITTI/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/
│   │   │   ├── 000000.bin
│   │   │   └── ...
│   │   └── labels/
│   │       ├── 000000.label
│   │       └── ...
│   ├── 01/
│   └── ...
```

3. Update the data path in `example_config.yaml` or pass it via command line.

## 🎯 Usage

### Quick Demo

Run a quick demo with synthetic data:

```bash
python examples/demo.py
```

This will:
- Create a synthetic point cloud
- Compress it using Oct-Attention
- Decompress and compute metrics
- Generate comparison visualization

### Training

Train the model on SemanticKITTI:

```bash
python train.py \
    --config example_config.yaml \
    --data-root /path/to/SemanticKITTI \
    --output-dir ./checkpoints \
    --log-dir ./logs \
    --device cuda
```

**Training Options:**
- `--config`: Path to configuration file
- `--data-root`: Path to SemanticKITTI dataset
- `--checkpoint`: Resume from checkpoint
- `--batch-size`: Override batch size
- `--epochs`: Override number of epochs
- `--lr`: Override learning rate

### Inference

Compress a point cloud using a trained model:

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pth \
    --input /path/to/point_cloud.bin \
    --output ./output \
    --visualize
```

**Inference Options:**
- `--checkpoint`: Path to trained model
- `--input`: Input point cloud (.bin or .npy)
- `--output`: Output directory
- `--visualize`: Generate comparison plots

### Python API

```python
import numpy as np
from sc_pcgc import PointCloudCompressor
from sc_pcgc.utils import compute_metrics

# Create compressor
compressor = PointCloudCompressor(
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_depth=10,
    voxel_size=0.1,
)

# Load point cloud
points = np.load('point_cloud.npy')

# Compress
compressed = compressor.compress(points, return_metrics=True)
print(f"Bitrate: {compressed['bitrate']:.2f} bits")
print(f"BPP: {compressed['bpp']:.4f}")

# Decompress
reconstructed = compressor.decompress(compressed['encoded'])

# Evaluate
metrics = compute_metrics(points, reconstructed, compressed['bitrate'])
print(f"Chamfer Distance: {metrics['chamfer_distance']:.4f}")
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

## ⚙️ Configuration

Configure the model and compression settings in `example_config.yaml`:

```yaml
model:
  d_model: 256          # Transformer dimension
  n_heads: 8            # Number of attention heads
  n_layers: 6           # Number of layers
  d_ff: 1024           # Feed-forward dimension
  dropout: 0.1

octree:
  max_depth: 10         # Maximum tree depth
  voxel_size: 0.1       # Voxel size (meters)

priority:
  semantic_weights:     # Importance weights by class
    car: 1.0
    pedestrian: 1.0
    cyclist: 0.9
    building: 0.5
    vegetation: 0.3
    road: 0.4
    # ...

compression:
  use_entropy_coding: true
  quantization_bits: 10

training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 100
```

## 📈 Evaluation Metrics

The system computes several metrics:

- **Geometric Quality**:
  - Chamfer Distance: Average nearest-neighbor distance
  - Hausdorff Distance: Maximum nearest-neighbor distance
  - PSNR: Peak Signal-to-Noise Ratio

- **Compression Efficiency**:
  - Bitrate: Total bits used for compression
  - BPP (Bits Per Point): Average bits per point
  - Compression Ratio: Original/Compressed size

## 🏗️ Architecture Details

### Oct-Attention Mechanism

The Oct-Attention mechanism extends standard transformer attention to octree structures:

1. **3D Positional Encoding**: Encodes spatial position, depth, and octant information
2. **Multi-Head Attention**: Captures relationships between octree nodes
3. **Context Modeling**: Learns contextual features for entropy coding

### Priority-Based Encoding

Semantic labels are mapped to priority values:
- **High Priority** (1.0): Cars, pedestrians, cyclists
- **Medium Priority** (0.4-0.5): Buildings, roads, signs
- **Low Priority** (0.2-0.3): Vegetation, terrain

Nodes with higher priority are encoded with more detail.

### Rate-Distortion Optimization

The loss function balances compression rate and reconstruction quality:

```
Loss = Distortion + λ × Rate
```

where λ controls the rate-distortion trade-off.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{sc-pcgc,
  title = {Sc-PCGC: Scalable Point Cloud Geometry Compression},
  author = {ESRSchao},
  year = {2024},
  url = {https://github.com/ESRSchao/Sc-PCGC}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the Oct-Attention paper and octree-based compression techniques
- SemanticKITTI dataset for autonomous driving point clouds
- PyTorch and Open3D communities

## 📧 Contact

For questions and discussions, please open an issue on GitHub.

---

**Note**: This is a research implementation. For production use, consider optimizing the entropy coding and adding actual codec integration.
