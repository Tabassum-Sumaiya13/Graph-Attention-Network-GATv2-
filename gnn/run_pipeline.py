"""
Main GNN Pipeline Runner

Complete pipeline for:
1. Data loading and preprocessing
2. Graph construction with multiple methods
3. Enhanced feature engineering
4. Graph Attention Network training
5. Cross-validation evaluation
6. Visualization and interpretation
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.data import Data, Batch
from pathlib import Path
import os
import time
import warnings

warnings.filterwarnings('ignore')

# Local imports
from config import Config, QuickTestConfig
from data_loader import load_all_data, get_sample_data
from feature_engineer import FeatureEngineer
from graph_builder import GraphBuilder
from model import SurvivalGAT, CoxPHLoss, CombinedLoss
from trainer import GNNTrainer
from evaluator import SurvivalEvaluator
from interpreter import GNNInterpreter


def create_graphs(sample_df, expr, cell_locs, marker_names, num_clusters, config):
    """Create PyTorch Geometric graphs for all samples"""
    print("\n" + "="*70)
    print("CREATING GRAPHS")
    print("="*70)
    
    feature_engineer = FeatureEngineer(marker_names, num_clusters, config)
    graph_builder = GraphBuilder(config)
    
    graphs = []
    skipped = 0
    
    for idx, row in sample_df.iterrows():
        sample_id = row['acquisition_id']
        
        # Get sample data
        sample_expr, sample_locs = get_sample_data(
            sample_id, expr, cell_locs, config
        )
        
        if sample_expr is None:
            skipped += 1
            continue
        
        # Coordinates and clusters
        coords = sample_locs[['X', 'Y']].values
        clusters = sample_expr.cluster.values
        
        # Node features
        node_features = feature_engineer.compute_node_features(
            sample_expr, coords, clusters
        )
        
        # Build graph
        edge_index = graph_builder.build_graph(coords)
        
        if edge_index.shape[1] == 0:
            skipped += 1
            continue
        
        # Edge features
        bio_cols = [c for c in sample_expr.columns if c in marker_names]
        biomarkers = sample_expr[bio_cols].values
        edge_features = feature_engineer.compute_edge_features(
            coords, biomarkers, clusters, edge_index
        )
        
        # Create PyTorch Geometric Data
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),
            survival_time=torch.tensor(row['survival_day'], dtype=torch.float32),
            event=torch.tensor(row['survival_status'], dtype=torch.float32),
            patient_id=row['patient_id'],
            sample_id=sample_id,
            coords=torch.tensor(coords, dtype=torch.float32),
            clusters=torch.tensor(clusters, dtype=torch.long)
        )
        
        graphs.append(data)
        
        if len(graphs) % 25 == 0:
            print(f"  Processed {len(graphs)} samples...")
    
    print(f"\n  ✓ Created {len(graphs)} graphs (skipped {skipped})")
    print(f"  ✓ Node features: {graphs[0].x.shape[1]} dimensions")
    print(f"  ✓ Avg edges: {np.mean([g.edge_index.shape[1] for g in graphs]):.0f}")
    print(f"  ✓ Avg cells: {np.mean([g.x.shape[0] for g in graphs]):.0f}")
    
    return graphs, feature_engineer


def normalize_graphs(graphs, feature_engineer):
    """Normalize node features across all graphs"""
    # Collect all features
    all_features = np.vstack([g.x.numpy() for g in graphs])
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Transform
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float32)
    
    return graphs


def visualize_pipeline_steps(graphs, model, config, output_dir):
    """Create comprehensive visualizations"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Sample graph for visualization
    sample_graph = graphs[0]
    coords = sample_graph.coords.numpy()
    clusters = sample_graph.clusters.numpy()
    edge_index = sample_graph.edge_index.numpy()
    
    # ===== Figure 1: Data and Graph Construction =====
    fig1 = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.25)
    
    cell_colors = {i: plt.cm.tab20(i % 20) for i in range(16)}
    
    # Panel A: Cell positions
    ax1 = fig1.add_subplot(gs[0, 0])
    for c in np.unique(clusters):
        mask = clusters == c
        ax1.scatter(coords[mask, 0], coords[mask, 1], 
                   c=[cell_colors.get(c, 'gray')], s=15, alpha=0.7)
    ax1.set_title('A) Cell Spatial Distribution', fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_aspect('equal')
    
    # Panel B: Graph edges
    ax2 = fig1.add_subplot(gs[0, 1])
    for i in range(min(edge_index.shape[1], 3000)):
        src, tgt = edge_index[0, i], edge_index[1, i]
        ax2.plot([coords[src, 0], coords[tgt, 0]], 
                [coords[src, 1], coords[tgt, 1]], 
                'gray', alpha=0.05, linewidth=0.5)
    for c in np.unique(clusters):
        mask = clusters == c
        ax2.scatter(coords[mask, 0], coords[mask, 1], 
                   c=[cell_colors.get(c, 'gray')], s=15, alpha=0.8)
    ax2.set_title(f'B) K-NN Graph (k={config.K_NEIGHBORS})', fontweight='bold')
    ax2.set_xlabel('X Position')
    ax2.set_aspect('equal')
    
    # Panel C: Node features PCA
    ax3 = fig1.add_subplot(gs[0, 2])
    node_features = sample_graph.x.numpy()
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(node_features)
    scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=clusters, cmap='tab20', s=15, alpha=0.7)
    ax3.set_title(f'C) Node Features in PCA Space\n({node_features.shape[1]}D → 2D)', 
                  fontweight='bold')
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, ax=ax3)
    
    # Panel D: Feature composition
    ax4 = fig1.add_subplot(gs[1, 0])
    feature_dims = {'Biomarkers': 38, 'Cell Type': 16, 
                    'Density': 5, 'Spatial': 10}
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = ax4.bar(range(len(feature_dims)), list(feature_dims.values()), color=colors)
    ax4.set_xticks(range(len(feature_dims)))
    ax4.set_xticklabels(list(feature_dims.keys()))
    ax4.set_ylabel('Number of Features')
    ax4.set_title('D) Node Feature Breakdown', fontweight='bold')
    for bar, val in zip(bars, feature_dims.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(val), ha='center', fontweight='bold')
    
    # Panel E: Edge distance distribution
    ax5 = fig1.add_subplot(gs[1, 1])
    edge_distances = sample_graph.edge_attr[:, 0].numpy()
    ax5.hist(edge_distances, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax5.axvline(edge_distances.mean(), color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Edge Distance')
    ax5.set_ylabel('Count')
    ax5.set_title('E) Edge Distance Distribution', fontweight='bold')
    
    # Panel F: Summary stats
    ax6 = fig1.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
    GRAPH STATISTICS
    ════════════════════════════
    
    Sample: {sample_graph.sample_id}
    
    Nodes:      {sample_graph.x.shape[0]}
    Features:   {sample_graph.x.shape[1]}
    
    Edges:      {sample_graph.edge_index.shape[1]}
    Avg Degree: {sample_graph.edge_index.shape[1] / sample_graph.x.shape[0]:.1f}
    
    Survival:   {sample_graph.survival_time.item():.0f} days
    Event:      {'Yes' if sample_graph.event.item() == 1 else 'No'}
    
    Cell Types: {len(np.unique(clusters))}
    
    ════════════════════════════
    """
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Step 1: Data Loading & Graph Construction', fontsize=14, fontweight='bold')
    plt.savefig(output_dir / "step1_graph_construction.png", dpi=config.DPI, 
                bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: step1_graph_construction.png")
    plt.close()
    
    # ===== Figure 2: GAT Architecture =====
    fig2 = plt.figure(figsize=(16, 8))
    ax = fig2.add_subplot(111)
    ax.axis('off')
    
    # Draw architecture boxes
    layers = [
        ('Input\n(69D)', 0.08, '#3498db'),
        ('Linear\nProjection', 0.22, '#2ecc71'),
        (f'GAT×{config.NUM_LAYERS}\n{config.NUM_HEADS} heads', 0.40, '#e74c3c'),
        ('Global\nPooling', 0.58, '#9b59b6'),
        ('MLP\nHead', 0.76, '#f39c12'),
        ('Risk\nScore', 0.90, '#1abc9c'),
    ]
    
    for name, x, color in layers:
        rect = mpatches.FancyBboxPatch((x-0.06, 0.35), 0.12, 0.3,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, alpha=0.4,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.5, name, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    for i in range(len(layers) - 1):
        x1 = layers[i][1] + 0.06
        x2 = layers[i+1][1] - 0.06
        ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Architecture details
    ax.text(0.40, 0.15, 
            'Multi-Head Attention\n'
            f'• Hidden dim: {config.HIDDEN_DIM}\n'
            f'• Heads: {config.NUM_HEADS}\n'
            f'• Layers: {config.NUM_LAYERS}',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.text(0.76, 0.15,
            'Risk Prediction\n'
            f'• Pooling: {config.POOLING}\n'
            f'• Dropout: {config.DROPOUT}\n'
            f'• Output: scalar',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.suptitle('Step 2: Graph Attention Network Architecture', fontsize=14, fontweight='bold')
    plt.savefig(output_dir / "step2_gat_architecture.png", dpi=config.DPI,
                bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: step2_gat_architecture.png")
    plt.close()


def visualize_attention(model, sample_graph, config, output_dir):
    """Visualize attention weights"""
    model.eval()
    
    fig = plt.figure(figsize=(18, 6))
    
    with torch.no_grad():
        batch = Batch.from_data_list([sample_graph.to(config.DEVICE)])
        risk, attention_weights = model(batch, return_attention=True)
    
    coords = sample_graph.coords.cpu().numpy()
    clusters = sample_graph.clusters.cpu().numpy()
    
    if attention_weights:
        edge_idx, attn = attention_weights[-1]
        edge_idx = edge_idx.cpu().numpy()
        attn = attn.cpu().numpy()
        
        if len(attn.shape) > 1:
            attn = attn.mean(axis=1)
        
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-10)
        
        # Panel 1: Attention edges
        ax1 = fig.add_subplot(131)
        
        for i in range(min(len(attn), 5000)):
            src, tgt = edge_idx[0, i], edge_idx[1, i]
            if src < len(coords) and tgt < len(coords):
                alpha = 0.05 + 0.9 * attn_norm[i]
                ax1.plot([coords[src, 0], coords[tgt, 0]],
                        [coords[src, 1], coords[tgt, 1]],
                        color=plt.cm.Reds(attn_norm[i]), alpha=alpha, linewidth=0.8)
        
        cell_colors = {i: plt.cm.tab20(i % 20) for i in range(16)}
        for c in np.unique(clusters):
            mask = clusters == c
            ax1.scatter(coords[mask, 0], coords[mask, 1], 
                       c=[cell_colors.get(c, 'gray')], s=20, alpha=0.9,
                       edgecolor='white', linewidth=0.3)
        
        ax1.set_title('A) Attention-Weighted Edges', fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_aspect('equal')
        
        # Panel 2: Attention distribution
        ax2 = fig.add_subplot(132)
        ax2.hist(attn, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.axvline(attn.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {attn.mean():.4f}')
        ax2.set_xlabel('Attention Weight')
        ax2.set_ylabel('Count')
        ax2.set_title('B) Attention Distribution', fontweight='bold')
        ax2.legend()
        
        # Panel 3: Cell type interactions
        ax3 = fig.add_subplot(133)
        
        # Count high-attention cell type pairs
        top_k = 100
        top_indices = np.argsort(attn)[-top_k:]
        
        from collections import Counter
        pair_counts = Counter()
        for idx in top_indices:
            src, tgt = edge_idx[0, idx], edge_idx[1, idx]
            if src < len(clusters) and tgt < len(clusters):
                t1, t2 = min(clusters[src], clusters[tgt]), max(clusters[src], clusters[tgt])
                pair_counts[(t1, t2)] += 1
        
        top_pairs = pair_counts.most_common(10)
        if top_pairs:
            pair_labels = [f'T{p[0]}-T{p[1]}' for p, _ in top_pairs]
            pair_values = [c for _, c in top_pairs]
            
            ax3.barh(range(len(top_pairs)), pair_values, color='#9b59b6', alpha=0.8)
            ax3.set_yticks(range(len(top_pairs)))
            ax3.set_yticklabels(pair_labels)
            ax3.set_xlabel('Count in Top-100 Edges')
            ax3.set_title('C) High-Attention Cell Type Pairs', fontweight='bold')
    
    plt.suptitle('Step 3: Attention Mechanism Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "step3_attention_analysis.png", dpi=config.DPI,
                bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: step3_attention_analysis.png")
    plt.close()


def visualize_results(cv_results, config, output_dir):
    """Visualize training results"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    c_indices = cv_results['c_indices']
    
    # Panel A: C-index by fold
    ax1 = fig.add_subplot(gs[0, 0])
    folds = range(1, len(c_indices) + 1)
    colors = ['#27ae60' if c > 0.5 else '#e74c3c' for c in c_indices]
    bars = ax1.bar(folds, c_indices, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(0.5, color='gray', linestyle='--', label='Random')
    ax1.axhline(np.mean(c_indices), color='blue', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(c_indices):.3f}')
    ax1.fill_between([0.5, len(c_indices)+0.5], 
                     np.mean(c_indices) - np.std(c_indices),
                     np.mean(c_indices) + np.std(c_indices),
                     color='blue', alpha=0.2)
    ax1.set_xlabel('CV Fold')
    ax1.set_ylabel('Concordance Index')
    ax1.set_title('A) Cross-Validation Performance', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0.3, 1.0)
    
    # Panel B: Training curves
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_losses' in cv_results and cv_results['train_losses']:
        # Pad arrays to same length
        max_len = max(len(losses) for losses in cv_results['train_losses'])
        padded = []
        for losses in cv_results['train_losses']:
            if len(losses) < max_len:
                padded.append(losses + [losses[-1]] * (max_len - len(losses)))
            else:
                padded.append(losses)
        
        mean_train = np.mean(padded, axis=0)
        std_train = np.std(padded, axis=0)
        epochs = range(1, len(mean_train) + 1)
        
        ax2.plot(epochs, mean_train, 'b-', linewidth=2, label='Mean Loss')
        ax2.fill_between(epochs, mean_train - std_train, mean_train + std_train,
                        alpha=0.3, color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cox PH Loss')
    ax2.set_title('B) Training Curve', fontweight='bold')
    ax2.legend()
    
    # Panel C: Comparison with baselines
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Try to load precomputed results
    methods = ['GAT (Ours)']
    scores = [np.mean(c_indices)]
    
    try:
        precomputed = {
            'Cell Type Prop': 'celltype_prop_concordance.npy',
            'Neighbor Mat': 'neighbor_mat_k10_concordance.npy',
            'Biomarker': 'avg_biomarker_cell_concordance.npy'
        }
        for name, fname in precomputed.items():
            path = config.PRECOMPUTED_DIR / fname
            if path.exists():
                methods.append(name)
                scores.append(np.load(path).mean())
    except:
        pass
    
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71'][:len(methods)]
    bars = ax3.bar(range(len(methods)), scores, color=colors, alpha=0.8, edgecolor='black')
    ax3.axhline(0.5, color='gray', linestyle='--')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=15)
    ax3.set_ylabel('Concordance Index')
    ax3.set_title('C) Method Comparison', fontweight='bold')
    ax3.set_ylim(0.4, 0.9)
    
    for bar, val in zip(bars, scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontweight='bold')
    
    # Panel D: Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary = f"""
    ╔═══════════════════════════════════════╗
    ║     GAT PIPELINE RESULTS SUMMARY      ║
    ╠═══════════════════════════════════════╣
    ║                                       ║
    ║  Model Configuration:                 ║
    ║  • Hidden dim: {config.HIDDEN_DIM:<20} ║
    ║  • Attention heads: {config.NUM_HEADS:<15} ║
    ║  • GAT layers: {config.NUM_LAYERS:<18} ║
    ║  • Dropout: {config.DROPOUT:<21} ║
    ║                                       ║
    ║  Cross-Validation Results:            ║
    ║  • Mean C-index: {np.mean(c_indices):.4f}             ║
    ║  • Std: ±{np.std(c_indices):.4f}                     ║
    ║  • Best: {max(c_indices):.4f}                    ║
    ║  • Worst: {min(c_indices):.4f}                   ║
    ║                                       ║
    ║  Training:                            ║
    ║  • Epochs: {config.EPOCHS:<22} ║
    ║  • Batch size: {config.BATCH_SIZE:<19} ║
    ║  • Learning rate: {config.LEARNING_RATE:<15} ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
    """
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Step 4: Model Results & Comparison', fontsize=14, fontweight='bold')
    plt.savefig(output_dir / "step4_results_summary.png", dpi=config.DPI,
                bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: step4_results_summary.png")
    plt.close()


def run_pipeline(config=None):
    """
    Run the complete GNN survival pipeline.
    
    Args:
        config: Configuration object (defaults to Config())
    """
    if config is None:
        config = Config()
    
    print("\n" + "="*70)
    print("█▀▀ █▀█ ▄▀█ █▀█ █ █   ▄▀█ ▀█▀ ▀█▀ █▀▀ █▄ █ ▀█▀ █ █▀█ █▄ █")
    print("█▄█ █▀▄ █▀█ █▀▀ █▀█   █▀█  █   █  ██▄ █ ▀█  █  █ █▄█ █ ▀█")
    print("="*70)
    print("Graph Attention Network for Spatial Survival Analysis")
    print("="*70)
    
    # Set seeds
    np.random.seed(config.RANDOM_STATE)
    torch.manual_seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_STATE)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    total_start = time.time()
    
    # Step 1: Load data
    sample_df, expr, marker_names, cluster_names, num_clusters, cell_locs = load_all_data(config)
    
    # Step 2: Create graphs
    graphs, feature_engineer = create_graphs(
        sample_df, expr, cell_locs, marker_names, num_clusters, config
    )
    
    if len(graphs) < 10:
        print("ERROR: Not enough graphs. Check data.")
        return None
    
    # Normalize features
    graphs = normalize_graphs(graphs, feature_engineer)
    
    # Create model
    in_channels = graphs[0].x.shape[1]
    model = SurvivalGAT(in_channels, config).to(config.DEVICE)
    
    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Visualize pipeline
    visualize_pipeline_steps(graphs, model, config, config.OUTPUT_DIR)
    
    # Step 3: Cross-validation
    print("\n" + "="*70)
    print("CROSS-VALIDATION TRAINING")
    print("="*70)
    
    patient_ids = [g.patient_id for g in graphs]
    unique_patients = list(set(patient_ids))
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    groups = [patient_to_idx[p] for p in patient_ids]
    
    kf = GroupKFold(n_splits=config.N_SPLITS)
    
    cv_results = {
        'c_indices': [],
        'train_losses': [],
        'predictions': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(graphs, groups=groups)):
        print(f"\n  Fold {fold + 1}/{config.N_SPLITS}")
        print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]
        
        # Reinitialize model
        model = SurvivalGAT(in_channels, config).to(config.DEVICE)
        
        # Create trainer
        criterion = CoxPHLoss()
        trainer = GNNTrainer(model, config, criterion)
        
        # Train
        history = trainer.train(train_graphs, test_graphs, verbose=True)
        
        # Final evaluation
        c_index, _ = trainer.evaluate(test_graphs)
        
        cv_results['c_indices'].append(c_index)
        cv_results['train_losses'].append(history['train_loss'])
        
        print(f"    Final C-index: {c_index:.4f}")
    
    print(f"\n  {'='*40}")
    print(f"  Mean C-index: {np.mean(cv_results['c_indices']):.4f} ± {np.std(cv_results['c_indices']):.4f}")
    print(f"  {'='*40}")
    
    # Visualize attention (using last model)
    visualize_attention(model, graphs[0], config, config.OUTPUT_DIR)
    
    # Visualize results
    visualize_results(cv_results, config, config.OUTPUT_DIR)
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n  Total time: {total_time:.1f} seconds ({total_time/60:.1f} min)")
    print(f"\n  Output: {config.OUTPUT_DIR}")
    print(f"  ├── step1_graph_construction.png")
    print(f"  ├── step2_gat_architecture.png")
    print(f"  ├── step3_attention_analysis.png")
    print(f"  └── step4_results_summary.png")
    
    return cv_results


if __name__ == "__main__":
    # Use default config
    config = Config()
    
    # Or use quick test config for faster iteration
    # config = QuickTestConfig()
    
    results = run_pipeline(config)
