"""Octree data structure for point cloud representation."""

import numpy as np
from typing import List, Optional, Tuple
import torch


class OctreeNode:
    """Octree node for spatial subdivision of point clouds."""
    
    def __init__(
        self,
        center: np.ndarray,
        size: float,
        depth: int = 0,
        parent: Optional['OctreeNode'] = None,
    ):
        """Initialize octree node.
        
        Args:
            center: Center coordinates of the node [3]
            size: Size of the node (cube side length)
            depth: Depth level in the tree
            parent: Parent node reference
        """
        self.center = center
        self.size = size
        self.depth = depth
        self.parent = parent
        
        # Children nodes (8 octants)
        self.children: List[Optional[OctreeNode]] = [None] * 8
        
        # Node properties
        self.points: List[int] = []  # Indices of points in this node
        self.is_leaf = True
        self.priority = 0.0  # Priority for encoding
        self.occupancy = 0  # Binary occupancy pattern (8 bits)
        
        # Features for neural network
        self.features: Optional[torch.Tensor] = None
    
    def get_child_center(self, octant: int) -> np.ndarray:
        """Get center coordinates of child octant.
        
        Args:
            octant: Octant index (0-7)
            
        Returns:
            Center coordinates of child node
        """
        offset = self.size / 4
        offsets = [
            [-offset, -offset, -offset],  # 0: (-, -, -)
            [offset, -offset, -offset],   # 1: (+, -, -)
            [-offset, offset, -offset],   # 2: (-, +, -)
            [offset, offset, -offset],    # 3: (+, +, -)
            [-offset, -offset, offset],   # 4: (-, -, +)
            [offset, -offset, offset],    # 5: (+, -, +)
            [-offset, offset, offset],    # 6: (-, +, +)
            [offset, offset, offset],     # 7: (+, +, +)
        ]
        return self.center + np.array(offsets[octant])
    
    def get_octant(self, point: np.ndarray) -> int:
        """Determine which octant a point belongs to.
        
        Args:
            point: Point coordinates [3]
            
        Returns:
            Octant index (0-7)
        """
        # Compare point position relative to node center
        diff = point - self.center
        octant = 0
        if diff[0] >= 0:
            octant |= 1
        if diff[1] >= 0:
            octant |= 2
        if diff[2] >= 0:
            octant |= 4
        return octant
    
    def subdivide(self):
        """Subdivide node into 8 children."""
        if not self.is_leaf:
            return
        
        self.is_leaf = False
        child_size = self.size / 2
        
        for i in range(8):
            child_center = self.get_child_center(i)
            self.children[i] = OctreeNode(
                center=child_center,
                size=child_size,
                depth=self.depth + 1,
                parent=self,
            )
    
    def compute_occupancy(self) -> int:
        """Compute occupancy pattern from children.
        
        Returns:
            8-bit occupancy pattern
        """
        occupancy = 0
        for i, child in enumerate(self.children):
            if child is not None and (not child.is_leaf or len(child.points) > 0):
                occupancy |= (1 << i)
        self.occupancy = occupancy
        return occupancy
    
    def __repr__(self) -> str:
        return (
            f"OctreeNode(depth={self.depth}, "
            f"center={self.center}, "
            f"size={self.size:.3f}, "
            f"points={len(self.points)}, "
            f"priority={self.priority:.3f})"
        )


class Octree:
    """Octree structure for point cloud compression."""
    
    def __init__(
        self,
        max_depth: int = 10,
        min_points: int = 1,
    ):
        """Initialize octree.
        
        Args:
            max_depth: Maximum depth of the tree
            min_points: Minimum points per leaf node
        """
        self.max_depth = max_depth
        self.min_points = min_points
        self.root: Optional[OctreeNode] = None
        self.nodes: List[OctreeNode] = []
        self.points: Optional[np.ndarray] = None
        self.priorities: Optional[np.ndarray] = None
    
    def build(
        self,
        points: np.ndarray,
        priorities: Optional[np.ndarray] = None,
    ):
        """Build octree from point cloud.
        
        Args:
            points: Point cloud coordinates [N, 3]
            priorities: Optional priority values [N]
        """
        if len(points) == 0:
            return
        
        # Store points for later reference
        self.points = points
        self.priorities = priorities if priorities is not None else np.ones(len(points))
        
        # Compute bounding box
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        center = (min_bound + max_bound) / 2
        size = np.max(max_bound - min_bound) * 1.1  # Add margin
        
        # Create root node
        self.root = OctreeNode(center=center, size=size, depth=0)
        self.nodes = [self.root]
        
        # Insert points
        for i, point in enumerate(points):
            priority = self.priorities[i]
            self._insert_point(self.root, point, i, priority)
    
    def _insert_point(
        self,
        node: OctreeNode,
        point: np.ndarray,
        point_idx: int,
        priority: float,
    ):
        """Recursively insert point into octree.
        
        Args:
            node: Current node
            point: Point coordinates [3]
            point_idx: Index of the point
            priority: Priority value
        """
        node.points.append(point_idx)
        node.priority = max(node.priority, priority)
        
        # Check if we should subdivide
        if (
            node.is_leaf
            and len(node.points) > self.min_points
            and node.depth < self.max_depth
        ):
            node.subdivide()
            
            # Redistribute points to children
            points_to_redistribute = node.points[:]
            node.points = []
            
            for idx in points_to_redistribute:
                # Get the actual point coordinates for this index
                point_coords = self.points[idx]
                point_priority = self.priorities[idx]
                octant = node.get_octant(point_coords)
                child = node.children[octant]
                if child is not None:
                    self._insert_point(child, point_coords, idx, point_priority)
                    if child not in self.nodes:
                        self.nodes.append(child)
        
        # If not leaf, insert into appropriate child
        elif not node.is_leaf:
            octant = node.get_octant(point)
            child = node.children[octant]
            if child is not None:
                self._insert_point(child, point, point_idx, priority)
    
    def get_leaf_nodes(self) -> List[OctreeNode]:
        """Get all leaf nodes.
        
        Returns:
            List of leaf nodes
        """
        return [node for node in self.nodes if node.is_leaf]
    
    def get_nodes_at_depth(self, depth: int) -> List[OctreeNode]:
        """Get all nodes at specific depth.
        
        Args:
            depth: Target depth level
            
        Returns:
            List of nodes at specified depth
        """
        return [node for node in self.nodes if node.depth == depth]
    
    def traverse_breadth_first(self) -> List[OctreeNode]:
        """Traverse octree in breadth-first order.
        
        Returns:
            List of nodes in breadth-first order
        """
        if self.root is None:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            if not node.is_leaf:
                for child in node.children:
                    if child is not None:
                        queue.append(child)
        
        return result
    
    def traverse_priority_order(self) -> List[OctreeNode]:
        """Traverse octree in priority order (high to low).
        
        Returns:
            List of nodes sorted by priority
        """
        return sorted(self.nodes, key=lambda x: x.priority, reverse=True)
    
    def __len__(self) -> int:
        """Get number of nodes in the tree."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        return (
            f"Octree(nodes={len(self.nodes)}, "
            f"max_depth={self.max_depth}, "
            f"min_points={self.min_points})"
        )
