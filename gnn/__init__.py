"""
GNN Pipeline for Spatial Survival Analysis

A modular Graph Attention Network pipeline for predicting survival outcomes
from spatial cell data.

Components:
- config.py: Configuration settings
- data_loader.py: Data loading utilities
- feature_engineer.py: Feature engineering
- graph_builder.py: Graph construction
- model.py: GNN models and losses
- trainer.py: Training utilities
- evaluator.py: Evaluation metrics
- interpreter.py: Model interpretation
- run_pipeline.py: Main pipeline runner

Usage:
    from gnn_pipeline import run_pipeline, Config
    
    config = Config()
    config.MAX_SAMPLES = 100
    config.EPOCHS = 50
    
    results = run_pipeline(config)
"""

from .config import Config, QuickTestConfig, FullRunConfig
from .run_pipeline import run_pipeline

__all__ = ['Config', 'QuickTestConfig', 'FullRunConfig', 'run_pipeline']
