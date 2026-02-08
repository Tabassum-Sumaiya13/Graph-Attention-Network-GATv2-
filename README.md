# Attention-Based Graph Neural Network for Survival Prediction

**Graph Attention Network (GATv2) Framework for Cancer Survival Prediction from Single-Cell Spatial Proteomics Data**

---

## Overview

This repository implements a **Graph Attention Network (GATv2)** pipeline for predicting patient survival outcomes from single-cell spatial proteomics data. The model constructs cell-cell interaction graphs from tissue samples and uses multi-head attention mechanisms to learn which cellular interactions are most predictive of survival.

### Key Innovation
- Models tissue as a **graph** where cells are nodes and spatial proximity defines edges
- Uses **attention mechanisms** to automatically learn which cell-cell interactions matter for survival
- Provides **interpretable** outputs through attention weight visualization

---

## Research Summary

### Problem Statement
> Can we predict cancer patient survival based on the **spatial arrangement of cells** in their tumor using attention-based deep learning?

### Dataset: UPMC Head & Neck Cancer Cohort

| Property | Value |
|----------|-------|
| Cancer Type | Head & Neck Squamous Cell Carcinoma (HNSCC) |
| Patients | ~80 unique patients |
| Tissue Samples | ~300 tissue regions |
| Total Cells | ~2,000,000 individual cells |
| Protein Markers | 40 biomarkers measured |
| Cell Types | 16 distinct cell types identified |
| Technology | Multiplexed Ion Beam Imaging (MIBI) |

---

## Model Architecture

### Graph Attention Network (GATv2)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GATv2 SURVIVAL PREDICTION MODEL                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT: Cell Graph G = (V, E)                                       │
│  • V = cells (nodes) with biomarker + cell-type features            │
│  • E = edges from k-NN spatial proximity                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  INPUT PROJECTION                                            │   │
│  │  Linear → BatchNorm → ReLU → Dropout                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  GAT LAYERS (×4)                                             │   │
│  │                                                              │   │
│  │  Multi-Head Attention (8 heads):                             │   │
│  │  α_ij = softmax(LeakyReLU(a^T[Wh_i || Wh_j]))               │   │
│  │                                                              │   │
│  │  h_i' = σ(Σ α_ij · Wh_j)                                    │   │
│  │                                                              │   │
│  │  → BatchNorm → ELU → Dropout                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  GLOBAL POOLING (Combined)                                   │   │
│  │  • Mean pooling                                              │   │
│  │  • Max pooling                                               │   │
│  │  • Attention pooling                                         │   │
│  │  → Concatenate all three                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  MLP HEAD                                                    │   │
│  │  Linear → BatchNorm → ReLU → Dropout → Linear → Risk Score   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  LOSS: Cox Proportional Hazards (Negative Log Partial Likelihood)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Model Type | GATv2 |
| Hidden Dimension | 128 |
| Attention Heads | 8 |
| Number of Layers | 4 |
| Dropout | 0.2 |
| Pooling | Combined (mean + max + attention) |
| Learning Rate | 0.001 |
| Epochs | 150 |

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  1. DATA LOADING                                                 │
│     • Cell locations (X, Y coordinates)                          │
│     • Protein expression (40 biomarkers per cell)                │
│     • Cell type labels (16 types)                                │
│     • Patient survival data (time + event)                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. GRAPH CONSTRUCTION                                           │
│     • k-NN graph (k=15) based on spatial coordinates             │
│     • Each cell = node                                           │
│     • Spatial neighbors = edges                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. FEATURE ENGINEERING                                          │
│                                                                  │
│  Node Features:                                                  │
│     • Biomarker expression (40 dims)                             │
│     • Cell type one-hot encoding (16 dims)                       │
│     • Local density features                                     │
│     • Spatial context features                                   │
│                                                                  │
│  Edge Features:                                                  │
│     • Euclidean distance                                         │
│     • Biomarker similarity                                       │
│     • Same cell-type indicator                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. TRAINING                                                     │
│     • 5-fold patient-stratified cross-validation                 │
│     • Cox PH loss function                                       │
│     • Early stopping (patience=20)                               │
│     • Learning rate scheduling                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. OUTPUT                                                       │
│     • Risk scores per patient                                    │
│     • Concordance index (C-index)                                │
│     • Attention weights for interpretation                       │
│     • Visualizations of important cell interactions              │
└──────────────────────────────────────────────────────────────────┘
```

---

## How Attention Works

### What is Attention in GAT?

The model learns **attention weights** between connected cells:

```
α_ij = attention weight from cell i to cell j

Higher α_ij means:
  → Cell j's features are MORE important for updating cell i's representation
  → The interaction between cell i and j is significant for survival prediction
```

### Biological Interpretation

| High Attention Between | Interpretation |
|------------------------|----------------|
| Tumor ↔ CD8+ T-cell | Active immune surveillance |
| Tumor ↔ Macrophage | Immune response or tumor-associated macrophages |
| Tumor ↔ Tumor | Tumor clustering patterns |
| Immune ↔ Immune | Immune cell coordination |

---
