"""
Create C-index Method Comparison Bar Chart for PPT
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
methods = [
    'Biomarker\nRegion',
    'Biomarker\nCell', 
    'Cell Type\nProportion',
    'Neighborhood\nMatrix',
    "Ripley's K",
    'GAT\n(Ours)'
]

c_indices = [0.66, 0.66, 0.70, 0.70, 0.68, 0.55]
std_devs = [0.08, 0.07, 0.06, 0.07, 0.09, 0.10]

# Colors
colors = ['#3498db', '#3498db', '#2ecc71', '#2ecc71', '#3498db', '#9b59b6']

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Bar positions
x = np.arange(len(methods))
bars = ax.bar(x, c_indices, yerr=std_devs, capsize=5, color=colors, 
              edgecolor='black', linewidth=1.5, alpha=0.85)

# Reference lines
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (0.5)')
ax.axhline(y=0.7, color='green', linestyle=':', linewidth=2, label='Good threshold (0.7)')

# Labels on bars
for i, (bar, val, std) in enumerate(zip(bars, c_indices, std_devs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Styling
ax.set_ylabel('Concordance Index (C-index)', fontsize=14, fontweight='bold')
ax.set_xlabel('Method', fontsize=14, fontweight='bold')
ax.set_title('Survival Prediction Performance Comparison', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(0.4, 0.9)
ax.set_yticks(np.arange(0.4, 0.95, 0.1))

# Grid
ax.yaxis.grid(True, linestyle='-', alpha=0.3)
ax.set_axisbelow(True)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='RSF + Biomarker Features'),
    Patch(facecolor='#2ecc71', edgecolor='black', label='RSF + Spatial Features (Best)'),
    Patch(facecolor='#9b59b6', edgecolor='black', label='Graph Attention Network'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Random baseline (0.5)'),
    plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2, label='Good threshold (0.7)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# Add annotation for GAT
ax.annotate('Limited by\nsmall data\n(~300 samples)', 
            xy=(5, 0.55), xytext=(5, 0.48),
            fontsize=9, ha='center', color='#9b59b6',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#9b59b6'))

plt.tight_layout()
plt.savefig('output/c_index_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: output/c_index_comparison.png")
plt.show()
