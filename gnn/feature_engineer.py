"""
Enhanced Feature Engineering for the GNN Survival Pipeline

Features computed:
1. Node features (per cell): 69+ dimensions
   - Biomarker expression (38)
   - Cell type one-hot (16)
   - Local density (5)
   - Spatial context (10)
   
2. Edge features (per edge): 3 dimensions
   - Distance
   - Biomarker similarity
   - Same cell type
   
3. Graph-level features (per sample): 370+ dimensions
   - Cell type proportions
   - Neighborhood matrix
   - Spatial statistics
   - Biomarker statistics
   - Diversity indices
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from typing import Dict, List, Tuple, Optional


class FeatureEngineer:
    """
    Comprehensive feature engineering from spatial cell data.
    """
    
    def __init__(self, marker_names: List[str], num_clusters: int, config):
        self.marker_names = marker_names
        self.num_clusters = num_clusters
        self.config = config
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.graph_scaler = StandardScaler()
        self._fitted = False
    
    def get_node_feature_dim(self) -> int:
        """Return expected node feature dimension"""
        dim = 0
        if self.config.USE_BIOMARKERS:
            dim += len(self.marker_names)
        if self.config.USE_CELLTYPE_ONEHOT:
            dim += self.num_clusters
        if self.config.USE_DENSITY_FEATURES:
            dim += 5
        if self.config.USE_SPATIAL_CONTEXT:
            dim += 10
        return dim
    
    def compute_node_features(self, expr_df: pd.DataFrame, coords: np.ndarray, 
                             clusters: np.ndarray) -> np.ndarray:
        """
        Compute node features for each cell.
        
        Args:
            expr_df: Expression data for this sample
            coords: Cell coordinates (N, 2)
            clusters: Cell cluster assignments (N,)
            
        Returns:
            Node feature matrix (N, D)
        """
        n_cells = len(expr_df)
        features_list = []
        
        # 1. Biomarker expression
        if self.config.USE_BIOMARKERS:
            bio_cols = [c for c in expr_df.columns if c in self.marker_names]
            biomarker_features = expr_df[bio_cols].values
            features_list.append(biomarker_features)
        
        # 2. Cell type one-hot
        if self.config.USE_CELLTYPE_ONEHOT:
            celltype_onehot = np.zeros((n_cells, self.num_clusters))
            for i, c in enumerate(clusters):
                if 0 <= c < self.num_clusters:
                    celltype_onehot[i, c] = 1
            features_list.append(celltype_onehot)
        
        # 3. Local density features
        if self.config.USE_DENSITY_FEATURES:
            density_features = self._compute_density_features(coords)
            features_list.append(density_features)
        
        # 4. Spatial context features
        if self.config.USE_SPATIAL_CONTEXT:
            spatial_features = self._compute_spatial_context(coords, clusters)
            features_list.append(spatial_features)
        
        # Concatenate all features
        node_features = np.hstack(features_list)
        
        return node_features
    
    def _compute_density_features(self, coords: np.ndarray) -> np.ndarray:
        """Compute local density features for each cell"""
        n_cells = len(coords)
        k = self.config.DENSITY_K
        features = np.zeros((n_cells, 5))
        
        if n_cells < k + 1:
            return features
        
        # k-NN for density estimation
        nbrs = NearestNeighbors(n_neighbors=min(k+1, n_cells)).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        
        # Features
        features[:, 0] = distances[:, 1:].mean(axis=1)      # Mean distance
        features[:, 1] = distances[:, 1]                     # Min distance (NN)
        features[:, 2] = distances[:, -1]                    # Max distance (k-th)
        features[:, 3] = distances[:, 1:].std(axis=1)       # Distance std
        features[:, 4] = 1 / (features[:, 0] + 1e-6)        # Local density
        
        return features
    
    def _compute_spatial_context(self, coords: np.ndarray, 
                                 clusters: np.ndarray) -> np.ndarray:
        """Compute spatial context features for each cell"""
        n_cells = len(coords)
        k = self.config.CONTEXT_K
        features = np.zeros((n_cells, 10))
        
        if n_cells < k + 1:
            return features
        
        # k-NN
        nbrs = NearestNeighbors(n_neighbors=min(k+1, n_cells)).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        for i in range(n_cells):
            neighbor_idx = indices[i, 1:]
            neighbor_types = clusters[neighbor_idx]
            
            # Cell type composition
            type_counts = Counter(neighbor_types)
            top_types = type_counts.most_common(5)
            for j, (_, count) in enumerate(top_types):
                if j < 5:
                    features[i, j] = count / len(neighbor_idx)
            
            # Diversity
            type_probs = np.array([c / len(neighbor_idx) for _, c in type_counts.items()])
            features[i, 5] = entropy(type_probs + 1e-10)
            
            # Same-type fraction
            features[i, 6] = (neighbor_types == clusters[i]).mean()
            
            # Unique types normalized
            features[i, 7] = len(type_counts) / self.num_clusters
            
            # Distance from neighborhood centroid
            neighbor_centroid = coords[neighbor_idx].mean(axis=0)
            features[i, 8] = np.linalg.norm(coords[i] - neighbor_centroid)
            
            # Distance from global centroid
            features[i, 9] = np.linalg.norm(coords[i] - coords.mean(axis=0))
        
        return features
    
    def compute_edge_features(self, coords: np.ndarray, biomarkers: np.ndarray,
                             clusters: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """
        Compute edge features for each connection.
        
        Args:
            coords: Cell coordinates (N, 2)
            biomarkers: Biomarker expression (N, M)
            clusters: Cell cluster assignments (N,)
            edge_index: Edge connections (2, E)
            
        Returns:
            Edge feature matrix (E, 3)
        """
        n_edges = edge_index.shape[1]
        edge_features = np.zeros((n_edges, 3))
        
        for e in range(n_edges):
            i, j = edge_index[0, e], edge_index[1, e]
            
            # Distance
            if self.config.USE_EDGE_DISTANCE:
                edge_features[e, 0] = np.linalg.norm(coords[i] - coords[j])
            
            # Biomarker similarity
            if self.config.USE_EDGE_SIMILARITY:
                bio_i, bio_j = biomarkers[i], biomarkers[j]
                norm_i, norm_j = np.linalg.norm(bio_i), np.linalg.norm(bio_j)
                if norm_i > 0 and norm_j > 0:
                    edge_features[e, 1] = np.dot(bio_i, bio_j) / (norm_i * norm_j)
            
            # Same cell type
            if self.config.USE_EDGE_SAMETYPE:
                edge_features[e, 2] = 1.0 if clusters[i] == clusters[j] else 0.0
        
        return edge_features
    
    def compute_graph_features(self, expr_df: pd.DataFrame, coords: np.ndarray,
                              clusters: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute graph-level features for a sample.
        
        Returns dictionary with:
            - celltype_prop: Cell type proportions (16,)
            - biomarker_mean: Mean expression per marker (38,)
            - biomarker_std: Std expression per marker (38,)
            - spatial_stats: Spatial distribution stats (20,)
            - neighbor_matrix: Cell-cell adjacency (256,)
            - diversity: Diversity indices (10,)
        """
        features = {}
        bio_cols = [c for c in expr_df.columns if c in self.marker_names]
        biomarkers = expr_df[bio_cols].values
        
        # 1. Cell type proportions
        type_counts = Counter(clusters)
        props = np.zeros(self.num_clusters)
        for t, c in type_counts.items():
            if 0 <= t < self.num_clusters:
                props[t] = c / len(clusters)
        features['celltype_prop'] = props
        
        # 2. Biomarker statistics
        features['biomarker_mean'] = biomarkers.mean(axis=0)
        features['biomarker_std'] = biomarkers.std(axis=0)
        
        # 3. Spatial statistics
        features['spatial_stats'] = self._compute_spatial_statistics(coords, clusters)
        
        # 4. Neighborhood matrix
        features['neighbor_matrix'] = self._compute_neighborhood_matrix(coords, clusters).flatten()
        
        # 5. Diversity indices
        features['diversity'] = self._compute_diversity_indices(clusters, biomarkers)
        
        return features
    
    def _compute_spatial_statistics(self, coords: np.ndarray, 
                                    clusters: np.ndarray) -> np.ndarray:
        """Compute spatial distribution statistics"""
        stats = np.zeros(20)
        
        # Spatial extent
        stats[0] = coords[:, 0].ptp()  # X range
        stats[1] = coords[:, 1].ptp()  # Y range
        stats[2] = len(coords)         # Cell count
        
        # Centroid
        centroid = coords.mean(axis=0)
        stats[3], stats[4] = centroid
        
        # Spread
        stats[5], stats[6] = coords.std(axis=0)
        
        # Distances from centroid
        dists = np.linalg.norm(coords - centroid, axis=1)
        stats[7] = dists.mean()
        stats[8] = dists.std()
        stats[9] = dists.max()
        
        # Per-celltype centroids (first 5 types)
        for t in range(min(5, self.num_clusters)):
            mask = clusters == t
            if mask.sum() > 0:
                type_centroid = coords[mask].mean(axis=0)
                stats[10 + t*2] = type_centroid[0]
                stats[11 + t*2] = type_centroid[1]
        
        return stats
    
    def _compute_neighborhood_matrix(self, coords: np.ndarray, 
                                     clusters: np.ndarray) -> np.ndarray:
        """Compute cell-cell type adjacency matrix"""
        k = self.config.K_NEIGHBORS
        n_cells = len(coords)
        neighbor_mat = np.zeros((self.num_clusters, self.num_clusters))
        
        if n_cells < k + 1:
            return neighbor_mat
        
        nbrs = NearestNeighbors(n_neighbors=min(k+1, n_cells)).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        for i in range(n_cells):
            center_type = clusters[i]
            if not (0 <= center_type < self.num_clusters):
                continue
            for j in indices[i, 1:]:
                neighbor_type = clusters[j]
                if 0 <= neighbor_type < self.num_clusters:
                    neighbor_mat[center_type, neighbor_type] += 1
        
        # Normalize rows
        row_sums = neighbor_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        neighbor_mat = neighbor_mat / row_sums
        
        return neighbor_mat
    
    def _compute_diversity_indices(self, clusters: np.ndarray, 
                                   biomarkers: np.ndarray) -> np.ndarray:
        """Compute diversity and heterogeneity indices"""
        indices = np.zeros(10)
        
        # Cell type diversity
        type_counts = Counter(clusters)
        type_probs = np.array([c / len(clusters) for c in type_counts.values()])
        
        indices[0] = entropy(type_probs + 1e-10)            # Shannon
        indices[1] = 1 - np.sum(type_probs ** 2)            # Simpson's
        indices[2] = len(type_counts)                        # Richness
        
        # Evenness
        if len(type_counts) > 1:
            indices[3] = indices[0] / np.log(len(type_counts))
        
        # Biomarker heterogeneity
        indices[4] = biomarkers.std(axis=0).mean()
        indices[5] = biomarkers.std(axis=0).std()
        
        # PCA variance
        if len(biomarkers) > 3 and biomarkers.shape[1] > 1:
            try:
                pca = PCA(n_components=min(3, len(biomarkers), biomarkers.shape[1]))
                pca.fit(biomarkers)
                indices[6] = pca.explained_variance_ratio_[0]
                indices[7] = pca.explained_variance_ratio_.sum()
            except:
                pass
        
        # Higher-order statistics
        try:
            indices[8] = np.mean([skew(biomarkers[:, i]) for i in range(biomarkers.shape[1])])
            indices[9] = np.mean([kurtosis(biomarkers[:, i]) for i in range(biomarkers.shape[1])])
        except:
            pass
        
        return indices
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using StandardScaler"""
        if fit:
            return self.node_scaler.fit_transform(features)
        else:
            if not self._fitted:
                return self.node_scaler.fit_transform(features)
            return self.node_scaler.transform(features)
