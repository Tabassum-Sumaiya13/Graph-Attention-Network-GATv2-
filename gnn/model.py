"""
Graph Attention Network Models for Survival Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class SurvivalGAT(nn.Module):
    """
    Graph Attention Network for Survival Prediction
    
    Architecture:
    - Input projection layer
    - Multiple GAT layers with multi-head attention
    - Global pooling (configurable: mean/max/combined)
    - MLP head for risk prediction
    
    The model outputs a scalar risk score where higher values
    indicate higher risk (shorter survival).
    """
    
    def __init__(self, in_channels: int, config):
        super(SurvivalGAT, self).__init__()
        
        self.config = config
        hidden = config.HIDDEN_DIM
        heads = config.NUM_HEADS
        layers = config.NUM_LAYERS
        dropout = config.DROPOUT
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if config.MODEL_TYPE == 'GATv2':
            self.gat_layers.append(
                GATv2Conv(hidden, hidden, heads=heads, dropout=dropout, concat=True)
            )
        else:
            self.gat_layers.append(
                GATConv(hidden, hidden, heads=heads, dropout=dropout, concat=True)
            )
        self.batch_norms.append(nn.BatchNorm1d(hidden * heads))
        
        # Middle layers
        for _ in range(layers - 2):
            if config.MODEL_TYPE == 'GATv2':
                self.gat_layers.append(
                    GATv2Conv(hidden * heads, hidden, heads=heads, dropout=dropout, concat=True)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden * heads, hidden, heads=heads, dropout=dropout, concat=True)
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden * heads))
        
        # Final layer (single head)
        if config.MODEL_TYPE == 'GATv2':
            self.gat_layers.append(
                GATv2Conv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
            )
        else:
            self.gat_layers.append(
                GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
            )
        self.batch_norms.append(nn.BatchNorm1d(hidden))
        
        # Attention pooling layer
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.Tanh(),
            nn.Linear(hidden // 4, 1)
        )
        
        # Output dimension based on pooling type
        if config.POOLING == 'combined':
            mlp_input = hidden * 3  # mean + max + attention
        elif config.POOLING == 'attention':
            mlp_input = hidden
        else:
            mlp_input = hidden
        
        # MLP head for risk prediction
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        
        # Store attention weights for visualization
        self.attention_weights = None
        self.node_embeddings = None
        
    def forward(self, data, return_attention=False):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric batch
            return_attention: If True, return attention weights
            
        Returns:
            risk: Predicted risk scores (B,)
            attention_weights: (optional) List of attention weights
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # GAT layers
        attention_weights = []
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            if return_attention:
                x, attn = gat(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_index, attn))
            else:
                x = gat(x, edge_index)
            
            x = bn(x)
            
            # Activation (except last layer)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.config.DROPOUT, training=self.training)
        
        # Store for visualization
        self.node_embeddings = x.detach()
        if return_attention:
            self.attention_weights = attention_weights
        
        # Global pooling
        if self.config.POOLING == 'mean':
            x_global = global_mean_pool(x, batch)
        elif self.config.POOLING == 'max':
            x_global = global_max_pool(x, batch)
        elif self.config.POOLING == 'add':
            x_global = global_add_pool(x, batch)
        elif self.config.POOLING == 'attention':
            attn_scores = torch.softmax(self.attention_pool(x), dim=0)
            x_global = global_add_pool(x * attn_scores, batch)
        elif self.config.POOLING == 'combined':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            attn_scores = torch.softmax(self.attention_pool(x), dim=0)
            x_attn = global_add_pool(x * attn_scores, batch)
            x_global = torch.cat([x_mean, x_max, x_attn], dim=-1)
        else:
            x_global = global_mean_pool(x, batch)
        
        # Risk prediction
        risk = self.mlp(x_global)
        
        if return_attention:
            return risk, attention_weights
        return risk


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards negative log partial likelihood loss.
    
    This loss function is designed for survival analysis where we want
    to learn a risk score that correctly orders patients by their
    survival outcomes.
    """
    
    def forward(self, risk_scores: torch.Tensor, survival_time: torch.Tensor, 
                event: torch.Tensor) -> torch.Tensor:
        """
        Compute Cox PH loss.
        
        Args:
            risk_scores: Predicted risk scores (B, 1)
            survival_time: Observed survival times (B,)
            event: Event indicators (B,) - 1 if event, 0 if censored
            
        Returns:
            loss: Negative log partial likelihood
        """
        # Ensure proper shapes
        risk_scores = risk_scores.squeeze()
        
        # Handle edge cases
        if event.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
        # Sort by survival time (descending)
        sorted_indices = torch.argsort(survival_time, descending=True)
        sorted_risk = risk_scores[sorted_indices]
        sorted_event = event[sorted_indices]
        
        # Log-sum-exp of risk scores for risk sets
        log_cumsum_h = torch.logcumsumexp(sorted_risk.flip(0), dim=0).flip(0)
        
        # Partial likelihood (only for events)
        log_lik = sorted_risk - log_cumsum_h
        
        # Mask for events
        event_mask = sorted_event.bool()
        
        # Negative mean log likelihood
        loss = -log_lik[event_mask].mean()
        
        return loss


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss for survival prediction.
    
    Encourages correct ordering: patients who died earlier should
    have higher risk scores than those who survived longer.
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, risk_scores: torch.Tensor, survival_time: torch.Tensor,
                event: torch.Tensor) -> torch.Tensor:
        """Compute ranking loss"""
        risk_scores = risk_scores.squeeze()
        n = len(risk_scores)
        
        if n < 2 or event.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
        loss = 0.0
        count = 0
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                # i should have higher risk if i died earlier and j survived longer
                if event[i] == 1 and survival_time[i] < survival_time[j]:
                    # Max margin loss
                    loss += F.relu(self.margin - (risk_scores[i] - risk_scores[j]))
                    count += 1
        
        if count == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
        return loss / count


class CombinedLoss(nn.Module):
    """Combined Cox + Ranking loss"""
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.cox_loss = CoxPHLoss()
        self.ranking_loss = RankingLoss()
        self.alpha = alpha
    
    def forward(self, risk_scores, survival_time, event):
        cox = self.cox_loss(risk_scores, survival_time, event)
        rank = self.ranking_loss(risk_scores, survival_time, event)
        return self.alpha * cox + (1 - self.alpha) * rank
