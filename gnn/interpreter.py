"""
Model interpretation and visualization utilities
"""

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from typing import List, Dict, Tuple, Optional
from collections import Counter


class GNNInterpreter:
    """
    Interpretation utilities for Graph Attention Networks.
    
    Provides:
    - Attention weight analysis
    - Node importance scores
    - Cell type interaction analysis
    - Feature importance estimation
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def get_attention_weights(self, graph: Data) -> List[Tuple]:
        """
        Get attention weights from all GAT layers.
        
        Returns:
            List of (edge_index, attention_weights) tuples
        """
        self.model.eval()
        
        with torch.no_grad():
            batch = Batch.from_data_list([graph]).to(self.config.DEVICE)
            _, attention_weights = self.model(batch, return_attention=True)
        
        return attention_weights
    
    def analyze_attention_by_celltype(self, graph: Data) -> Dict:
        """
        Analyze attention patterns between cell types.
        
        Returns:
            Dictionary with:
            - interaction_matrix: Cell type x Cell type attention matrix
            - top_interactions: Top cell type pairs by attention
        """
        attention_weights = self.get_attention_weights(graph)
        
        if not attention_weights:
            return {}
        
        # Use last layer attention
        edge_index, attn = attention_weights[-1]
        edge_index = edge_index.cpu().numpy()
        attn = attn.cpu().numpy()
        
        # Average over heads if multi-head
        if len(attn.shape) > 1:
            attn = attn.mean(axis=1)
        
        clusters = graph.clusters.cpu().numpy()
        n_types = self.config.num_clusters if hasattr(self.config, 'num_clusters') else 16
        
        # Build interaction matrix
        interaction_mat = np.zeros((n_types, n_types))
        interaction_count = np.zeros((n_types, n_types))
        
        for e in range(edge_index.shape[1]):
            src, tgt = edge_index[0, e], edge_index[1, e]
            if src < len(clusters) and tgt < len(clusters):
                src_type = clusters[src]
                tgt_type = clusters[tgt]
                if 0 <= src_type < n_types and 0 <= tgt_type < n_types:
                    interaction_mat[src_type, tgt_type] += attn[e]
                    interaction_count[src_type, tgt_type] += 1
        
        # Normalize
        interaction_count[interaction_count == 0] = 1
        interaction_mat = interaction_mat / interaction_count
        
        # Find top interactions
        flat_idx = np.argsort(interaction_mat.flatten())[::-1][:20]
        top_interactions = []
        for idx in flat_idx:
            i, j = idx // n_types, idx % n_types
            top_interactions.append({
                'source_type': i,
                'target_type': j,
                'attention': interaction_mat[i, j]
            })
        
        return {
            'interaction_matrix': interaction_mat,
            'top_interactions': top_interactions
        }
    
    def node_importance(self, graph: Data, method: str = 'attention') -> np.ndarray:
        """
        Compute importance score for each node.
        
        Args:
            graph: Input graph
            method: 'attention' or 'gradient'
            
        Returns:
            Importance scores (N,)
        """
        if method == 'attention':
            return self._attention_importance(graph)
        elif method == 'gradient':
            return self._gradient_importance(graph)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _attention_importance(self, graph: Data) -> np.ndarray:
        """Node importance based on received attention"""
        attention_weights = self.get_attention_weights(graph)
        
        if not attention_weights:
            return np.zeros(graph.x.shape[0])
        
        edge_index, attn = attention_weights[-1]
        edge_index = edge_index.cpu().numpy()
        attn = attn.cpu().numpy()
        
        if len(attn.shape) > 1:
            attn = attn.mean(axis=1)
        
        n_nodes = graph.x.shape[0]
        importance = np.zeros(n_nodes)
        
        # Sum of incoming attention
        for e in range(edge_index.shape[1]):
            tgt = edge_index[1, e]
            if tgt < n_nodes:
                importance[tgt] += attn[e]
        
        # Normalize
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return importance
    
    def _gradient_importance(self, graph: Data) -> np.ndarray:
        """Node importance based on gradient magnitude"""
        self.model.eval()
        
        batch = Batch.from_data_list([graph]).to(self.config.DEVICE)
        batch.x.requires_grad_(True)
        
        risk = self.model(batch)
        risk.backward()
        
        grad = batch.x.grad.cpu().numpy()
        importance = np.linalg.norm(grad, axis=1)
        
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return importance
    
    def feature_importance(self, graphs: List[Data], n_samples: int = 100) -> np.ndarray:
        """
        Estimate feature importance using permutation importance.
        
        Args:
            graphs: List of test graphs
            n_samples: Number of permutation samples
            
        Returns:
            Feature importance scores (D,)
        """
        self.model.eval()
        
        # Baseline predictions
        with torch.no_grad():
            batch = Batch.from_data_list(graphs).to(self.config.DEVICE)
            baseline_risk = self.model(batch).cpu().numpy().flatten()
        
        n_features = graphs[0].x.shape[1]
        importance = np.zeros(n_features)
        
        for f in range(n_features):
            diffs = []
            
            for _ in range(n_samples):
                # Permute feature f
                perturbed_graphs = []
                for g in graphs:
                    g_new = g.clone()
                    perm = torch.randperm(g_new.x.shape[0])
                    g_new.x[:, f] = g_new.x[perm, f]
                    perturbed_graphs.append(g_new)
                
                # Predict with perturbation
                with torch.no_grad():
                    batch = Batch.from_data_list(perturbed_graphs).to(self.config.DEVICE)
                    perturbed_risk = self.model(batch).cpu().numpy().flatten()
                
                # Difference
                diffs.append(np.abs(baseline_risk - perturbed_risk).mean())
            
            importance[f] = np.mean(diffs)
        
        # Normalize
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return importance
    
    def identify_critical_cells(self, graph: Data, top_k: int = 20) -> Dict:
        """
        Identify the most critical cells for survival prediction.
        
        Returns:
            Dictionary with top cells and their properties
        """
        importance = self.node_importance(graph, method='attention')
        
        top_indices = np.argsort(importance)[-top_k:][::-1]
        
        clusters = graph.clusters.cpu().numpy()
        coords = graph.coords.cpu().numpy()
        
        critical_cells = []
        for idx in top_indices:
            critical_cells.append({
                'index': int(idx),
                'importance': float(importance[idx]),
                'cell_type': int(clusters[idx]),
                'x': float(coords[idx, 0]),
                'y': float(coords[idx, 1])
            })
        
        # Aggregate by cell type
        type_importance = {}
        for cell in critical_cells:
            t = cell['cell_type']
            if t not in type_importance:
                type_importance[t] = []
            type_importance[t].append(cell['importance'])
        
        type_summary = {
            t: {'mean': np.mean(imps), 'count': len(imps)}
            for t, imps in type_importance.items()
        }
        
        return {
            'critical_cells': critical_cells,
            'type_summary': type_summary
        }
