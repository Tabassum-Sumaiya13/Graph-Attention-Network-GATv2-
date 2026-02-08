"""
Graph Construction utilities for the GNN Survival Pipeline
"""

import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors, KDTree
from typing import Tuple


class GraphBuilder:
    """
    Build graphs from cell spatial data.
    
    Supports multiple graph construction methods:
    - k-NN: k-nearest neighbor graph
    - Radius: Radius-based graph
    - Delaunay: Delaunay triangulation
    """
    
    def __init__(self, config):
        self.config = config
        self.k = config.K_NEIGHBORS
        self.radius = config.RADIUS
        self.graph_type = config.GRAPH_TYPE
    
    def build_graph(self, coords: np.ndarray) -> np.ndarray:
        """
        Build graph based on configured type.
        
        Args:
            coords: Cell coordinates (N, 2)
            
        Returns:
            edge_index: Edge list (2, E)
        """
        if self.graph_type == 'knn':
            return self.build_knn_graph(coords)
        elif self.graph_type == 'radius':
            return self.build_radius_graph(coords)
        elif self.graph_type == 'delaunay':
            return self.build_delaunay_graph(coords)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
    
    def build_knn_graph(self, coords: np.ndarray) -> np.ndarray:
        """
        Build k-nearest neighbor graph.
        
        Creates bidirectional edges between each cell and its k nearest neighbors.
        """
        n_cells = len(coords)
        k = min(self.k, n_cells - 1)
        
        if k < 1:
            return np.array([[], []], dtype=np.int64)
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        # Build bidirectional edge list
        sources = []
        targets = []
        
        for i in range(n_cells):
            for j in indices[i, 1:]:  # Skip self (index 0)
                sources.extend([i, j])
                targets.extend([j, i])
        
        # Convert to array and remove duplicates
        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_set = set(zip(edge_index[0], edge_index[1]))
        edge_index = np.array(list(edge_set), dtype=np.int64).T
        
        return edge_index
    
    def build_radius_graph(self, coords: np.ndarray) -> np.ndarray:
        """
        Build radius-based graph.
        
        Creates edges between cells within a specified distance.
        """
        # Use KDTree for efficient radius search
        tree = KDTree(coords)
        pairs = tree.query_pairs(self.radius)
        
        if not pairs:
            return np.array([[], []], dtype=np.int64)
        
        # Build bidirectional edges
        sources = []
        targets = []
        for i, j in pairs:
            sources.extend([i, j])
            targets.extend([j, i])
        
        return np.array([sources, targets], dtype=np.int64)
    
    def build_delaunay_graph(self, coords: np.ndarray) -> np.ndarray:
        """
        Build Delaunay triangulation graph.
        
        Creates edges based on Delaunay triangulation of cell positions.
        """
        if len(coords) < 4:
            return np.array([[], []], dtype=np.int64)
        
        try:
            tri = Delaunay(coords)
        except:
            # Fall back to k-NN if Delaunay fails
            return self.build_knn_graph(coords)
        
        # Extract edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add((simplex[i], simplex[j]))
                    edges.add((simplex[j], simplex[i]))
        
        edge_index = np.array(list(edges), dtype=np.int64).T
        return edge_index
    
    def compute_graph_statistics(self, edge_index: np.ndarray, 
                                 n_nodes: int) -> dict:
        """
        Compute graph statistics.
        
        Returns:
            Dictionary with:
            - n_edges: Number of edges
            - avg_degree: Average node degree
            - max_degree: Maximum node degree
            - min_degree: Minimum node degree
            - density: Graph density
        """
        n_edges = edge_index.shape[1]
        
        # Compute degree for each node
        degrees = np.zeros(n_nodes)
        for i in range(n_edges):
            degrees[edge_index[0, i]] += 1
        
        stats = {
            'n_edges': n_edges,
            'avg_degree': degrees.mean(),
            'max_degree': degrees.max(),
            'min_degree': degrees.min(),
            'density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        }
        
        return stats
