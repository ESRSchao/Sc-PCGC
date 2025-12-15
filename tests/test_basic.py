"""Basic tests for Sc-PCGC components."""

import unittest
import numpy as np
import torch

from sc_pcgc.models.octree import Octree, OctreeNode
from sc_pcgc.data.preprocessing import (
    voxelize_point_cloud,
    compute_semantic_priorities,
)
from sc_pcgc.compression import PointCloudCompressor


class TestOctree(unittest.TestCase):
    """Test octree functionality."""
    
    def test_octree_creation(self):
        """Test basic octree creation."""
        octree = Octree(max_depth=5)
        self.assertEqual(octree.max_depth, 5)
        self.assertIsNone(octree.root)
    
    def test_octree_build(self):
        """Test building octree from points."""
        points = np.random.rand(100, 3).astype(np.float32)
        octree = Octree(max_depth=5)
        octree.build(points)
        
        self.assertIsNotNone(octree.root)
        self.assertGreater(len(octree.nodes), 0)
    
    def test_octree_with_priorities(self):
        """Test octree with priority values."""
        points = np.random.rand(50, 3).astype(np.float32)
        priorities = np.random.rand(50).astype(np.float32)
        
        octree = Octree(max_depth=5)
        octree.build(points, priorities)
        
        self.assertIsNotNone(octree.root)
        self.assertGreater(octree.root.priority, 0)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    def test_voxelization(self):
        """Test point cloud voxelization."""
        points = np.random.rand(100, 3).astype(np.float32)
        voxel_size = 0.1
        
        voxelized, _, _, _ = voxelize_point_cloud(points, voxel_size)
        
        self.assertIsNotNone(voxelized)
        self.assertLessEqual(len(voxelized), len(points))
    
    def test_priority_computation(self):
        """Test semantic priority computation."""
        labels = np.array([1, 6, 9, 15])  # car, person, road, vegetation
        priorities = compute_semantic_priorities(labels)
        
        self.assertEqual(len(priorities), len(labels))
        self.assertEqual(priorities[0], 1.0)  # car
        self.assertEqual(priorities[1], 1.0)  # person
        self.assertLess(priorities[3], priorities[0])  # vegetation < car


class TestCompressor(unittest.TestCase):
    """Test point cloud compressor."""
    
    def test_compressor_creation(self):
        """Test compressor initialization."""
        compressor = PointCloudCompressor(
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_depth=5,
        )
        self.assertIsNotNone(compressor)
    
    def test_compression_pipeline(self):
        """Test full compression and decompression pipeline."""
        # Create small synthetic point cloud
        points = np.random.rand(50, 3).astype(np.float32)
        
        # Create compressor with small model
        compressor = PointCloudCompressor(
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_depth=5,
            voxel_size=0.1,
        )
        
        # Compress
        compressed = compressor.compress(points, return_metrics=True)
        
        self.assertIn('encoded', compressed)
        self.assertIn('bitrate', compressed)
        self.assertGreater(compressed['bitrate'], 0)
        
        # Decompress
        reconstructed = compressor.decompress(compressed['encoded'])
        
        self.assertIsNotNone(reconstructed)
        self.assertEqual(reconstructed.shape[1], 3)


class TestModels(unittest.TestCase):
    """Test neural network models."""
    
    def test_oct_attention(self):
        """Test Oct-Attention module."""
        from sc_pcgc.models.oct_attention import OctAttention
        
        d_model = 64
        n_heads = 4
        batch_size = 2
        num_nodes = 10
        
        attention = OctAttention(d_model, n_heads)
        x = torch.randn(batch_size, num_nodes, d_model)
        
        output, _ = attention(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_encoder(self):
        """Test priority octree encoder."""
        from sc_pcgc.models.encoder import PriorityOctreeEncoder
        
        encoder = PriorityOctreeEncoder(
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_depth=5,
        )
        
        points = torch.randn(1, 50, 3)
        priorities = torch.rand(1, 50)
        
        encoded = encoder(points, priorities)
        
        self.assertIn('occupancy', encoded)
        self.assertIn('depths', encoded)
    
    def test_decoder(self):
        """Test priority octree decoder."""
        from sc_pcgc.models.decoder import PriorityOctreeDecoder
        
        decoder = PriorityOctreeDecoder(d_model=64, max_depth=5)
        
        # Create dummy encoded data
        encoded = {
            'occupancy': torch.randint(0, 2, (1, 10)),
            'depths': torch.randint(0, 5, (1, 10)),
        }
        
        reconstructed = decoder(encoded)
        
        self.assertIsNotNone(reconstructed)
        self.assertEqual(reconstructed.shape[2], 3)


if __name__ == '__main__':
    unittest.main()
