"""
Create ML Pipeline Visualization for PPT
Shows the complete survival prediction workflow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
color_input = '#3498db'      # Blue
color_features = '#e74c3c'   # Red
color_rsf = '#2ecc71'        # Green
color_gat = '#9b59b6'        # Purple
color_output = '#f39c12'     # Orange

# Title
ax.text(5, 9.5, 'Survival Prediction Pipeline', 
        fontsize=24, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

# ============================================================================
# LEVEL 1: INPUT DATA
# ============================================================================
y_input = 8.0
ax.add_patch(FancyBboxPatch((0.5, y_input-0.3), 2, 0.6, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_input, edgecolor='black', linewidth=2))
ax.text(1.5, y_input, 'INPUT DATA', fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white')

# Input details
input_text = [
    '• 2M cells',
    '• 40 markers',
    '• X-Y coords',
    '• 16 cell types'
]
for i, txt in enumerate(input_text):
    ax.text(1.5, y_input-0.7-i*0.3, txt, fontsize=9, ha='center')

# ============================================================================
# LEVEL 2: FEATURE EXTRACTION
# ============================================================================
y_features = 5.5

# Arrow from input to features
ax.add_patch(FancyArrowPatch((1.5, y_input-0.8), (5, y_features+1.5),
                             arrowstyle='->', lw=3, color='gray', 
                             mutation_scale=30, zorder=1))

# Feature box
ax.add_patch(FancyBboxPatch((3, y_features-0.3), 4, 1.5, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_features, edgecolor='black', linewidth=2))
ax.text(5, y_features+0.9, 'FEATURE EXTRACTION', fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white')

# Feature types
features = [
    '1. Biomarker Region (40)',
    '2. Biomarker Cell (40)',
    '3. Cell Type Proportion (16)',
    '4. Neighborhood Matrix (256)',
    '5. Ripley\'s K (16)'
]
for i, feat in enumerate(features):
    ax.text(5, y_features+0.4-i*0.25, feat, fontsize=8, ha='center')

# ============================================================================
# LEVEL 3A: RSF MODEL (Left Branch)
# ============================================================================
y_model = 3.0

# Arrow to RSF
ax.add_patch(FancyArrowPatch((3.5, y_features-0.4), (2, y_model+0.8),
                             arrowstyle='->', lw=3, color='gray', 
                             mutation_scale=30, zorder=1))

# RSF box
ax.add_patch(FancyBboxPatch((0.5, y_model-0.3), 3, 1.0, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_rsf, edgecolor='black', linewidth=2))
ax.text(2, y_model+0.4, 'Random Survival Forest', fontsize=11, fontweight='bold', 
        ha='center', va='center', color='white')

# RSF details
rsf_details = [
    '• Uses pre-computed features',
    '• 10-fold CV',
    '• 1000 trees',
    '• Patient-level split'
]
for i, txt in enumerate(rsf_details):
    ax.text(2, y_model-0.1-i*0.2, txt, fontsize=7, ha='center')

# ============================================================================
# LEVEL 3B: GAT MODEL (Right Branch)
# ============================================================================

# Arrow to GAT
ax.add_patch(FancyArrowPatch((6.5, y_features-0.4), (8, y_model+0.8),
                             arrowstyle='->', lw=3, color='gray', 
                             mutation_scale=30, zorder=1))

# GAT box
ax.add_patch(FancyBboxPatch((6.5, y_model-0.3), 3, 1.0, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_gat, edgecolor='black', linewidth=2))
ax.text(8, y_model+0.4, 'Graph Attention Network', fontsize=11, fontweight='bold', 
        ha='center', va='center', color='white')

# GAT details
gat_details = [
    '• Learns from cell graphs',
    '• 5-fold CV',
    '• 4 GAT layers, 8 heads',
    '• Attention-based interactions'
]
for i, txt in enumerate(gat_details):
    ax.text(8, y_model-0.1-i*0.2, txt, fontsize=7, ha='center')

# ============================================================================
# LEVEL 4: OUTPUT
# ============================================================================
y_output = 0.8

# Arrows to output
ax.add_patch(FancyArrowPatch((2, y_model-0.4), (3.5, y_output+0.6),
                             arrowstyle='->', lw=3, color='gray', 
                             mutation_scale=30, zorder=1))
ax.add_patch(FancyArrowPatch((8, y_model-0.4), (6.5, y_output+0.6),
                             arrowstyle='->', lw=3, color='gray', 
                             mutation_scale=30, zorder=1))

# Output box
ax.add_patch(FancyBboxPatch((3, y_output-0.3), 4, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_output, edgecolor='black', linewidth=2))
ax.text(5, y_output+0.25, 'SURVIVAL PREDICTION OUTPUT', fontsize=11, fontweight='bold', 
        ha='center', va='center', color='white')
ax.text(5, y_output-0.05, '• C-index (0.5 = random, 1.0 = perfect)  • Risk scores  • Feature importance', 
        fontsize=8, ha='center', va='center')

# ============================================================================
# RESULTS COMPARISON TABLE
# ============================================================================
table_x = 0.5
table_y = 0.3
table_w = 9
table_h = 0.25

# Table header
ax.add_patch(Rectangle((table_x, table_y-table_h-0.05), table_w, table_h, 
                       facecolor='lightgray', edgecolor='black', linewidth=1))
ax.text(table_x+1.5, table_y-table_h/2-0.05, 'Method', fontsize=9, fontweight='bold', ha='center', va='center')
ax.text(table_x+4.5, table_y-table_h/2-0.05, 'C-index', fontsize=9, fontweight='bold', ha='center', va='center')
ax.text(table_x+7.5, table_y-table_h/2-0.05, 'Status', fontsize=9, fontweight='bold', ha='center', va='center')

# RSF results
ax.add_patch(Rectangle((table_x, table_y-2*table_h-0.05), table_w, table_h, 
                       facecolor=color_rsf, alpha=0.3, edgecolor='black', linewidth=1))
ax.text(table_x+1.5, table_y-1.5*table_h-0.05, 'RSF (Best Features)', fontsize=8, ha='center', va='center')
ax.text(table_x+4.5, table_y-1.5*table_h-0.05, '0.70', fontsize=8, fontweight='bold', ha='center', va='center')
ax.text(table_x+7.5, table_y-1.5*table_h-0.05, '✓ Best for small data', fontsize=8, ha='center', va='center')

# GAT results
ax.add_patch(Rectangle((table_x, table_y-3*table_h-0.05), table_w, table_h, 
                       facecolor=color_gat, alpha=0.3, edgecolor='black', linewidth=1))
ax.text(table_x+1.5, table_y-2.5*table_h-0.05, 'GAT (Graph Learning)', fontsize=8, ha='center', va='center')
ax.text(table_x+4.5, table_y-2.5*table_h-0.05, '0.55 ± 0.10', fontsize=8, fontweight='bold', ha='center', va='center')
ax.text(table_x+7.5, table_y-2.5*table_h-0.05, '△ Needs more data', fontsize=8, ha='center', va='center')

# ============================================================================
# LEGEND
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=color_input, label='Input Data'),
    mpatches.Patch(facecolor=color_features, label='Feature Engineering'),
    mpatches.Patch(facecolor=color_rsf, label='RSF Model'),
    mpatches.Patch(facecolor=color_gat, label='GAT Model'),
    mpatches.Patch(facecolor=color_output, label='Output')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('output/ml_pipeline_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: output/ml_pipeline_visualization.png")
plt.show()
