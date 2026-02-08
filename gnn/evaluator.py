"""
Evaluation metrics and model interpretation for the GNN Survival Pipeline
"""

import numpy as np
import torch
from torch_geometric.data import Batch
from typing import List, Dict, Tuple
from collections import Counter


class SurvivalEvaluator:
    """
    Comprehensive evaluation for survival prediction models.
    
    Metrics:
    - Concordance Index (C-index)
    - Time-dependent AUC
    - Integrated Brier Score
    - Calibration
    """
    
    def __init__(self):
        pass
    
    def concordance_index(self, risk: np.ndarray, time: np.ndarray,
                         event: np.ndarray) -> float:
        """Compute Harrell's concordance index"""
        n = len(risk)
        concordant = 0
        permissible = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if event[i] == 0 and event[j] == 0:
                    continue
                
                if time[i] < time[j] and event[i] == 1:
                    permissible += 1
                    if risk[i] > risk[j]:
                        concordant += 1
                    elif risk[i] == risk[j]:
                        concordant += 0.5
                        
                elif time[j] < time[i] and event[j] == 1:
                    permissible += 1
                    if risk[j] > risk[i]:
                        concordant += 1
                    elif risk[i] == risk[j]:
                        concordant += 0.5
                        
                elif time[i] == time[j]:
                    if event[i] == 1 and event[j] == 0:
                        permissible += 1
                        if risk[i] > risk[j]:
                            concordant += 1
                        elif risk[i] == risk[j]:
                            concordant += 0.5
                    elif event[j] == 1 and event[i] == 0:
                        permissible += 1
                        if risk[j] > risk[i]:
                            concordant += 1
                        elif risk[i] == risk[j]:
                            concordant += 0.5
        
        if permissible == 0:
            return 0.5
        
        return concordant / permissible
    
    def time_dependent_auc(self, risk: np.ndarray, time: np.ndarray,
                          event: np.ndarray, t: float) -> float:
        """
        Compute time-dependent AUC at time t.
        
        AUC(t) = P(risk_i > risk_j | T_i <= t, T_j > t)
        """
        # Cases: died by t
        cases_mask = (time <= t) & (event == 1)
        # Controls: still at risk at t
        controls_mask = time > t
        
        if cases_mask.sum() == 0 or controls_mask.sum() == 0:
            return 0.5
        
        case_risks = risk[cases_mask]
        control_risks = risk[controls_mask]
        
        # Count concordant pairs
        concordant = 0
        total = len(case_risks) * len(control_risks)
        
        for case_risk in case_risks:
            concordant += (case_risk > control_risks).sum()
            concordant += 0.5 * (case_risk == control_risks).sum()
        
        return concordant / total
    
    def stratify_by_risk(self, risk: np.ndarray, time: np.ndarray,
                        event: np.ndarray, n_groups: int = 3) -> Dict:
        """
        Stratify patients by risk and compute survival statistics per group.
        
        Returns:
            Dictionary with group statistics
        """
        # Sort by risk
        sorted_idx = np.argsort(risk)
        group_size = len(risk) // n_groups
        
        results = {}
        
        for g in range(n_groups):
            start = g * group_size
            if g == n_groups - 1:
                end = len(risk)
            else:
                end = (g + 1) * group_size
            
            idx = sorted_idx[start:end]
            
            results[f'group_{g+1}'] = {
                'n_patients': len(idx),
                'mean_risk': risk[idx].mean(),
                'median_survival': np.median(time[idx]),
                'event_rate': event[idx].mean(),
                'mean_survival': time[idx].mean()
            }
        
        return results
    
    def compute_all_metrics(self, risk: np.ndarray, time: np.ndarray,
                           event: np.ndarray) -> Dict:
        """Compute all evaluation metrics"""
        metrics = {}
        
        # C-index
        metrics['c_index'] = self.concordance_index(risk, time, event)
        
        # Time-dependent AUC at different time points
        time_points = [365, 730, 1095]  # 1, 2, 3 years
        for t in time_points:
            if t < time.max():
                metrics[f'auc_{t//365}y'] = self.time_dependent_auc(risk, time, event, t)
        
        # Risk stratification
        metrics['stratification'] = self.stratify_by_risk(risk, time, event)
        
        return metrics
