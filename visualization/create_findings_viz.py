"""
Create Key Findings Infographic for PPT
Summarizes main discoveries from the survival prediction pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge
import numpy as np

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Key Findings: Spatial Proteomics for Cancer Survival Prediction', 
             fontsize=20, fontweight='bold', y=0.98)

# ============================================================================
# PANEL A: Spatial Features Outperform Biomarkers
# ============================================================================
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('A) Spatial Features Beat Biomarkers Alone', fontsize=14, fontweight='bold', pad=10)

# Draw comparison arrows
# Biomarker only
ax1.add_patch(FancyBboxPatch((0.5, 7), 3, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#3498db', edgecolor='black', linewidth=2))
ax1.text(2, 7.6, 'Biomarker\nOnly', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
ax1.text(4, 7.6, '→', fontsize=30, ha='center', va='center')
ax1.add_patch(Circle((5.5, 7.6), 0.8, facecolor='#3498db', edgecolor='black', linewidth=2))
ax1.text(5.5, 7.6, '0.66', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# Spatial features
ax1.add_patch(FancyBboxPatch((0.5, 4.5), 3, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#2ecc71', edgecolor='black', linewidth=2))
ax1.text(2, 5.1, 'Spatial\nFeatures', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
ax1.text(4, 5.1, '→', fontsize=30, ha='center', va='center')
ax1.add_patch(Circle((5.5, 5.1), 0.8, facecolor='#2ecc71', edgecolor='black', linewidth=2))
ax1.text(5.5, 5.1, '0.70', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# Improvement arrow
ax1.annotate('', xy=(7.5, 7.6), xytext=(7.5, 5.1),
            arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=3))
ax1.text(8.5, 6.3, '+6%\nBetter!', fontsize=14, fontweight='bold', color='#e74c3c', ha='center')

# Insight box
ax1.add_patch(FancyBboxPatch((0.5, 1), 9, 2.5, boxstyle="round,pad=0.1",
                              facecolor='lightyellow', edgecolor='orange', linewidth=2))
ax1.text(5, 3, '[!] Key Insight', fontsize=12, fontweight='bold', ha='center', color='darkorange')
ax1.text(5, 2.2, 'Cell arrangement (who is next to whom)\nis MORE predictive than protein levels alone', 
         fontsize=10, ha='center', va='center')

# ============================================================================
# PANEL B: Best Predictive Features
# ============================================================================
ax2 = axes[0, 1]
features = ['Tumor-Immune\nNeighboring', 'Cell Type\nDiversity', 'CD8+ T cell\nProximity', 
            'Macrophage\nClustering', 'Ki67+\nDensity']
importance = [0.25, 0.20, 0.18, 0.15, 0.12]
colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f39c12']

bars = ax2.barh(features, importance, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Relative Importance', fontsize=12)
ax2.set_title('B) Top Predictive Features', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 0.35)

for bar, val in zip(bars, importance):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.0%}', 
             va='center', fontsize=11, fontweight='bold')

ax2.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
ax2.text(0.11, 4.5, 'Baseline', fontsize=9, color='gray')

# ============================================================================
# PANEL C: What Spatial Patterns Mean
# ============================================================================
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('C) Spatial Patterns & Survival', fontsize=14, fontweight='bold', pad=10)

# Good prognosis pattern
ax3.add_patch(FancyBboxPatch((0.3, 5.5), 4.2, 4, boxstyle="round,pad=0.1",
                              facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2))
ax3.text(2.4, 9, '✓ Better Survival', fontsize=12, fontweight='bold', color='#27ae60', ha='center')

# Draw scattered immune cells around tumor
np.random.seed(42)
for i in range(8):
    x, y = np.random.uniform(1, 3.8), np.random.uniform(6, 8.5)
    ax3.add_patch(Circle((x, y), 0.15, facecolor='#3498db', edgecolor='black'))  # Immune
ax3.add_patch(Circle((2.4, 7.2), 0.4, facecolor='#e74c3c', edgecolor='black', linewidth=2))  # Tumor
ax3.text(2.4, 6.2, 'Immune cells\ninfiltrating tumor', fontsize=9, ha='center')

# Poor prognosis pattern
ax3.add_patch(FancyBboxPatch((5.5, 5.5), 4.2, 4, boxstyle="round,pad=0.1",
                              facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2))
ax3.text(7.6, 9, '✗ Worse Survival', fontsize=12, fontweight='bold', color='#e74c3c', ha='center')

# Draw tumor cells clustered, immune excluded
ax3.add_patch(Circle((7.6, 7.5), 0.8, facecolor='#e74c3c', edgecolor='black', linewidth=2))  # Big tumor
for i in range(5):
    x = np.random.uniform(5.8, 6.3)
    y = np.random.uniform(6, 8.5)
    ax3.add_patch(Circle((x, y), 0.12, facecolor='#3498db', edgecolor='black'))  # Excluded immune
ax3.text(7.6, 6.2, 'Immune cells\nexcluded from tumor', fontsize=9, ha='center')

# Legend
ax3.add_patch(Circle((1.5, 5), 0.15, facecolor='#3498db', edgecolor='black'))
ax3.text(2, 5, 'Immune cell', fontsize=9, va='center')
ax3.add_patch(Circle((1.5, 4.5), 0.2, facecolor='#e74c3c', edgecolor='black'))
ax3.text(2, 4.5, 'Tumor cell', fontsize=9, va='center')

# Key finding
ax3.add_patch(FancyBboxPatch((0.3, 0.5), 9.4, 3.2, boxstyle="round,pad=0.1",
                              facecolor='lavender', edgecolor='purple', linewidth=2))
ax3.text(5, 3.2, '[*] Biological Finding', fontsize=12, fontweight='bold', ha='center', color='purple')
ax3.text(5, 2.2, 'Tumor-immune cell proximity correlates with\nbetter survival (immune infiltration = good)', 
         fontsize=10, ha='center')
ax3.text(5, 1.2, 'Neighborhood matrix captures this interaction!', fontsize=9, ha='center', style='italic')

# ============================================================================
# PANEL D: Summary Statistics
# ============================================================================
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('D) Pipeline Summary', fontsize=14, fontweight='bold', pad=10)

# Stats boxes
stats = [
    ('2M+', 'Cells Analyzed', '#3498db'),
    ('40', 'Protein Markers', '#e74c3c'),
    ('~300', 'Tissue Samples', '#2ecc71'),
    ('0.70', 'Best C-index', '#9b59b6'),
    ('16', 'Cell Types', '#f39c12'),
    ('5', 'Feature Sets', '#1abc9c')
]

positions = [(1, 8), (5, 8), (1, 5.5), (5, 5.5), (1, 3), (5, 3)]

for (val, label, color), (x, y) in zip(stats, positions):
    ax4.add_patch(FancyBboxPatch((x-0.8, y-1), 3.5, 2, boxstyle="round,pad=0.1",
                                  facecolor=color, edgecolor='black', linewidth=2, alpha=0.8))
    ax4.text(x+1, y+0.3, val, fontsize=20, fontweight='bold', color='white', ha='center', va='center')
    ax4.text(x+1, y-0.4, label, fontsize=10, color='white', ha='center', va='center')

# Conclusion box
ax4.add_patch(FancyBboxPatch((0.2, 0.3), 9.6, 1.5, boxstyle="round,pad=0.1",
                              facecolor='gold', edgecolor='black', linewidth=2, alpha=0.7))
ax4.text(5, 1.3, '>>> CONCLUSION', fontsize=12, fontweight='bold', ha='center')
ax4.text(5, 0.7, 'Spatial cell arrangements predict cancer survival better than protein levels alone', 
         fontsize=10, ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('output/pipeline_findings.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: output/pipeline_findings.png")
plt.show()
