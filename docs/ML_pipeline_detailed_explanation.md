# In-Depth Pipeline Explanation: Attention-Based GNN for Survival Prediction

## Table of Contents
1. [Complete Pipeline Flow](#complete-pipeline-flow)
2. [Step 1: Data Collection](#step-1-data-collection)
3. [Step 2: Preprocessing](#step-2-preprocessing)
4. [Step 3: Graph Construction](#step-3-graph-construction)
5. [Step 4: Feature Engineering](#step-4-feature-engineering)
6. [Step 5: Attention-Based GNN Model](#step-5-attention-based-gnn-model)
7. [Step 6: Survival Prediction](#step-6-survival-prediction)
8. [Step 7: Evaluation](#step-7-evaluation)
9. [Step 8: Interpretation](#step-8-interpretation)
10. [Summary Table](#summary-table)

---

## Complete Pipeline Flow

```
Raw Data → Preprocessing → Graph Construction → Feature Engineering → GNN Model → Survival Prediction → Interpretation
```

---

## Step 1: Data Collection

### Input
- **Raw spatial proteomics data** from technologies like CODEX, IMC, or MIBI

### Process
| Data Type | Description | Format |
|-----------|-------------|--------|
| **Cell Coordinates** | X, Y positions of each cell in tissue | `(cell_id, x, y)` |
| **Protein Intensities** | Expression levels of 30-50+ protein markers per cell | Matrix: `cells × proteins` |
| **Clinical Data** | Patient survival information | `(patient_id, survival_time, event_status)` |

### Output
- **Raw data tables**: Cell-level protein expression matrix + spatial coordinates + survival outcomes

### Example Data Structure:
```
Cell_ID | X_coord | Y_coord | CD3 | CD8 | CD45 | PD-L1 | ... | Patient_ID
--------|---------|---------|-----|-----|------|-------|-----|----------
cell_1  | 125.3   | 450.2   | 0.8 | 0.2 | 0.9  | 0.1   | ... | P001
cell_2  | 127.1   | 448.5   | 0.1 | 0.0 | 0.3  | 0.7   | ... | P001
```

---

## Step 2: Preprocessing

### Input
- Raw cell-level data from Step 1

### Process

| Sub-step | Method | Purpose |
|----------|--------|---------|
| **Normalization** | Z-score, quantile, or arcsinh transformation | Scale protein intensities to comparable ranges |
| **Quality Filtering** | Remove cells with low total signal or artifacts | Eliminate noise/debris |
| **Batch Correction** | ComBat, Harmony | Remove technical variation between samples |
| **Marker Selection** | Variance filtering, biological relevance | Focus on informative proteins |
| **Spatial Alignment** | Coordinate normalization per tissue | Enable cross-sample comparison |

### Methods Detail:

**Arcsinh Transformation** (common for mass cytometry):
$$x_{transformed} = \sinh^{-1}\left(\frac{x}{cofactor}\right)$$

**Z-score Normalization**:
$$z = \frac{x - \mu}{\sigma}$$

### Output
- **Clean, normalized data**: Same structure as input but processed
- Typically: `N cells × M proteins` matrix + `(x, y)` coordinates

---

## Step 3: Graph Construction ⭐

### Input
- Preprocessed cell data with coordinates and protein expression

### Process
This is **critical** — cells become **nodes**, spatial relationships become **edges**.

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Distance Threshold** | Connect cells within radius `r` (e.g., 50μm) | Simple, biologically motivated |
| **K-Nearest Neighbors (KNN)** | Connect each cell to K closest neighbors | More uniform connectivity |
| **Delaunay Triangulation** | Geometric triangulation | Biologically realistic |

### Graph Construction Algorithm:

```python
# Pseudocode for graph construction
for each cell_i in tissue:
    node_i.features = protein_expression_vector[cell_i]  # Shape: (M,)
    
    for each cell_j in neighbors(cell_i, method='KNN', k=10):
        edge_ij = create_edge(cell_i, cell_j)
        edge_ij.weight = compute_distance(cell_i, cell_j)
        edge_ij.features = [
            euclidean_distance,
            protein_similarity,  # e.g., cosine similarity
        ]
```

### Output
- **Graph G = (V, E)** where:
  - **V (Nodes)**: Each cell, with feature vector = protein intensities `[N × M]`
  - **E (Edges)**: Connections between neighboring cells `[E × edge_features]`
  - **Adjacency Matrix A**: `[N × N]` sparse matrix

### Visual Example:
```
    Cell A (CD8+ T-cell)
       /  \
     /      \
Cell B      Cell C
(Tumor)    (Macrophage)
    \      /
      \  /
    Cell D (Fibroblast)
```

---

## Step 4: Feature Engineering

### What is an "Enriched Graph"?

#### Starting Point: Basic Graph

After graph construction, you have a **basic graph** with minimal information:

```
BASIC GRAPH:
┌─────────────────────────────────────────────────────┐
│  Nodes: Cells with raw protein expression          │
│  Edges: Just connections (cell A neighbors cell B)  │
│  Edge weights: Maybe just distance                  │
└─────────────────────────────────────────────────────┘
```

#### Goal: Enriched Graph

An **enriched graph** has **additional computed features** that capture more biological meaning:

```
ENRICHED GRAPH:
┌─────────────────────────────────────────────────────┐
│  Nodes: Cells with:                                 │
│    - Raw protein expression                         │
│    - Cell type label                                │
│    - Local density                                  │
│    - Neighborhood diversity                         │
│    - Centrality score                               │
│                                                     │
│  Edges: Connections with:                           │
│    - Distance                                       │
│    - Protein similarity                             │
│    - Interaction type                               │
└─────────────────────────────────────────────────────┘
```

---

### A. Node Feature Engineering

We ADD new features to each cell (node):

#### 1. Raw Protein Features (already have)
```
Cell_1: [CD3=0.8, CD8=0.2, CD45=0.9, PD-L1=0.1, ...]
        └─────────────── M proteins ───────────────┘
```

#### 2. Cell Type Labels (computed or given)
Cluster cells into types using protein markers:

```python
# Example: Classify cell types based on protein markers
if CD3 > 0.5 and CD8 > 0.5:
    cell_type = "CD8+ T-cell"
elif CD3 > 0.5 and CD4 > 0.5:
    cell_type = "CD4+ T-cell"
elif CD68 > 0.5:
    cell_type = "Macrophage"
elif CK > 0.5:
    cell_type = "Tumor cell"
# ... etc

# Convert to one-hot encoding
cell_type_feature = [0, 1, 0, 0, 0]  # e.g., CD8+ T-cell
```

#### 3. Local Cell Density
How crowded is the neighborhood around each cell?

```
                    ○ ○
        ○         ○ ● ○ ○      ● = target cell
      ○ ● ○         ○ ○        ○ = neighbors
        ○           ○
     
     Low Density    High Density
     (sparse)       (crowded)
```

**Formula:**
$$\text{density}_i = \frac{|\{j : d(i,j) < r\}|}{\pi r^2}$$

Number of cells within radius r, normalized by area.

#### 4. Neighborhood Diversity (Shannon Entropy)
How diverse are the cell types around this cell?

```
Example 1: Low Diversity         Example 2: High Diversity
   T    T                           Mac   T
  T ● T   (all T-cells)           Tumor ● Fibro
   T    T                           T    Mac

Entropy ≈ 0                      Entropy ≈ 1.5
(homogeneous)                    (heterogeneous)
```

**Formula:**
$$H_i = -\sum_{t=1}^{T} p_t \log_2(p_t)$$

Where $p_t$ = proportion of cell type $t$ in neighborhood.

#### 5. Graph Centrality Metrics
How "important" is this cell in the network?

| Metric | Meaning | Biological Interpretation |
|--------|---------|---------------------------|
| **Degree Centrality** | Number of connections | Hub cells with many neighbors |
| **Betweenness Centrality** | How often cell lies on shortest paths | Bridge cells connecting regions |
| **Closeness Centrality** | Average distance to all other cells | Central vs. peripheral cells |

#### Final Node Feature Vector:

```
Original:     [protein_1, protein_2, ..., protein_M]
                           ↓
Enriched:     [protein_1, ..., protein_M, cell_type_onehot, density, entropy, degree, betweenness]
              └────────────────────────────────────────────────────────────────────────────────┘
                                        M + K additional features
```

---

### B. Edge Feature Engineering

We ADD meaningful features to each edge (connection):

#### 1. Euclidean Distance (basic)
$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

#### 2. Protein Expression Similarity
How similar are two connected cells?

**Cosine Similarity:**
$$\text{sim}_{ij} = \frac{\mathbf{h}_i \cdot \mathbf{h}_j}{||\mathbf{h}_i|| \cdot ||\mathbf{h}_j||}$$

```
Cell_i proteins: [0.8, 0.2, 0.9, 0.1]
Cell_j proteins: [0.7, 0.3, 0.8, 0.2]
                      ↓
Cosine similarity = 0.98 (very similar)
```

#### 3. Cell Type Interaction
Encode the TYPE of interaction:

```
T-cell ──── Tumor cell     → interaction_type = [1, 0, 0, ...]
T-cell ──── T-cell         → interaction_type = [0, 1, 0, ...]
Macrophage ── Tumor cell   → interaction_type = [0, 0, 1, ...]
```

#### 4. Relative Protein Differences
$$\Delta_{ij} = \mathbf{h}_i - \mathbf{h}_j$$

Captures which proteins differ between connected cells.

#### Final Edge Feature Vector:

```
Original:     [distance] or just binary (connected/not)
                           ↓
Enriched:     [distance, similarity, interaction_type_onehot, Δ_proteins]
```

---

### C. Visual Summary: Before vs. After Enrichment

```
BEFORE (Basic Graph):
═══════════════════════════════════════════════════════════
     [0.8, 0.2, 0.9]          [0.3, 0.7, 0.1]
          (A) ─────────────────── (B)
               edge: just "connected"

AFTER (Enriched Graph):
═══════════════════════════════════════════════════════════
     Node A Features:                Node B Features:
     ├─ proteins: [0.8, 0.2, 0.9]    ├─ proteins: [0.3, 0.7, 0.1]
     ├─ cell_type: T-cell            ├─ cell_type: Tumor
     ├─ density: 0.15                ├─ density: 0.32
     ├─ entropy: 1.2                 ├─ entropy: 0.8
     └─ degree: 5                    └─ degree: 8
          
          (A) ════════════════════════ (B)
                    Edge Features:
                    ├─ distance: 45.2 μm
                    ├─ similarity: 0.34
                    ├─ interaction: T-cell↔Tumor
                    └─ Δ_proteins: [0.5, -0.5, 0.8]
```

---

## Step 5: Attention-Based GNN Model ⭐⭐

### Input
- Enriched graph from Step 4

### The Core Idea

Traditional neural networks work on **fixed-size inputs** (images, vectors). But graphs have:
- Variable number of nodes
- Variable connectivity patterns
- No natural ordering

**GNNs solve this** by learning through **message passing** — each node learns by aggregating information from its neighbors.

---

### Message Passing Framework

The fundamental operation in ANY GNN:

```
┌─────────────────────────────────────────────────────────────┐
│                    MESSAGE PASSING                          │
│                                                             │
│   For each node i:                                          │
│                                                             │
│   1. COLLECT messages from neighbors                        │
│   2. AGGREGATE all messages                                 │
│   3. UPDATE node's own representation                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Mathematically:**
$$\mathbf{h}_i^{(l+1)} = \text{UPDATE}\left(\mathbf{h}_i^{(l)}, \text{AGGREGATE}\left(\{\mathbf{m}_{j \rightarrow i} : j \in \mathcal{N}(i)\}\right)\right)$$

Where:
- $\mathbf{h}_i^{(l)}$ = node i's features at layer l
- $\mathcal{N}(i)$ = neighbors of node i
- $\mathbf{m}_{j \rightarrow i}$ = message from neighbor j to node i

---

### Graph Attention Network (GAT): Detailed Breakdown

GAT is special because it uses **ATTENTION** to decide how much to listen to each neighbor.

#### Step 1: Linear Transformation

First, transform all node features through a learnable weight matrix:

$$\mathbf{z}_i = \mathbf{W} \cdot \mathbf{h}_i$$

```
Input features:  h_i = [0.8, 0.2, 0.9, 0.1, ...]  (dim = M)
                        ↓
Weight matrix:   W (learnable, dim = M × F)
                        ↓
Transformed:     z_i = [1.2, -0.3, 0.7, ...]      (dim = F)
```

#### Step 2: Compute Attention Scores

For each edge (i, j), compute how much node i should "attend" to neighbor j:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T \cdot [\mathbf{z}_i \| \mathbf{z}_j]\right)$$

```
z_i = [1.2, -0.3, 0.7]     (node i transformed)
z_j = [0.5, 0.8, -0.2]     (neighbor j transformed)
                ↓
Concatenate:    [1.2, -0.3, 0.7, 0.5, 0.8, -0.2]  (dim = 2F)
                ↓
Attention vec:  a = [a_1, a_2, ..., a_2F]  (learnable)
                ↓
Dot product:    e_ij = a · [z_i || z_j] = 0.85
                ↓
LeakyReLU:      e_ij = 0.85 (if positive)
```

#### Step 3: Normalize with Softmax

Convert raw scores to probabilities (sum to 1 over all neighbors):

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

```
Node i has neighbors: j, k, m

Raw scores:     e_ij = 0.85,  e_ik = 0.30,  e_im = 0.50
                           ↓ softmax
Attention:      α_ij = 0.45,  α_ik = 0.22,  α_im = 0.33
                (sum = 1.0)

Interpretation: 
  - Node i pays 45% attention to neighbor j
  - Node i pays 22% attention to neighbor k  
  - Node i pays 33% attention to neighbor m
```

#### Step 4: Weighted Aggregation

Combine neighbor features using attention weights:

$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \cdot \mathbf{z}_j\right)$$

```
h_i' = σ(0.45 × z_j + 0.22 × z_k + 0.33 × z_m)

     = σ(0.45 × [0.5, 0.8, -0.2] + 
         0.22 × [0.1, 0.3, 0.9] + 
         0.33 × [-0.2, 0.5, 0.4])

     = σ([0.225+0.022-0.066, 0.36+0.066+0.165, ...])
     
     = [new_feature_1, new_feature_2, ...]
```

---

### Multi-Head Attention

Problem: One attention head might miss important patterns.

Solution: Use **multiple attention heads** in parallel, then concatenate:

```
         ┌─── Head 1: α¹_ij → h'¹_i ───┐
         │                              │
Input ───┼─── Head 2: α²_ij → h'²_i ───┼─── Concatenate ─── Output
h_i      │                              │      or Mean
         │                              │
         └─── Head K: αᴷ_ij → h'ᴷ_i ───┘

Each head learns DIFFERENT attention patterns!
```

**Formula (concatenation):**
$$\mathbf{h}_i' = \Big\|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)}\mathbf{h}_j\right)$$

**Example with 4 heads:**
```
Head 1 output: [0.3, 0.5, 0.2, 0.8]  (4 dims)
Head 2 output: [0.1, 0.9, 0.4, 0.3]  (4 dims)
Head 3 output: [0.7, 0.2, 0.6, 0.1]  (4 dims)
Head 4 output: [0.4, 0.4, 0.5, 0.5]  (4 dims)
                      ↓ concatenate
Final output:  [0.3, 0.5, 0.2, 0.8, 0.1, 0.9, 0.4, 0.3, 0.7, 0.2, 0.6, 0.1, 0.4, 0.4, 0.5, 0.5]
               (16 dims = 4 heads × 4 dims)
```

---

### Full GAT Architecture for Survival Prediction

```
INPUT: Graph with N cells, each with F features
═══════════════════════════════════════════════════════════════

Layer 1: GAT (8 attention heads)
───────────────────────────────────────────────────────────────
  Input:  [N × F]           (N cells, F features each)
  
  For each cell i:
    • Compute attention to all neighbors (8 different ways)
    • Aggregate neighbor information
    
  Output: [N × (8 × 64)] = [N × 512]
  
  + Dropout(0.3) + ELU activation
  
───────────────────────────────────────────────────────────────

Layer 2: GAT (8 attention heads)
───────────────────────────────────────────────────────────────
  Input:  [N × 512]
  
  Same process: attention + aggregation
  
  Output: [N × (8 × 32)] = [N × 256]
  
  + Dropout(0.3) + ELU activation

───────────────────────────────────────────────────────────────

Layer 3: GAT (1 attention head - final layer)
───────────────────────────────────────────────────────────────
  Input:  [N × 256]
  Output: [N × 64]    (final cell embeddings)

═══════════════════════════════════════════════════════════════

GRAPH POOLING: Aggregate all cells → patient-level
───────────────────────────────────────────────────────────────
  Cell embeddings: [N × 64]
  
  Option A: Mean Pooling
    patient_embedding = mean([h_1, h_2, ..., h_N])
    
  Option B: Attention Pooling (learnable)
    β_i = softmax(MLP(h_i))   # importance of each cell
    patient_embedding = Σ β_i × h_i
    
  Output: [1 × 64]   (single vector per patient)

═══════════════════════════════════════════════════════════════

PREDICTION HEAD: MLP for survival risk
───────────────────────────────────────────────────────────────
  Input:  [1 × 64]
  
  FC Layer 1: 64 → 32 + ReLU
  FC Layer 2: 32 → 16 + ReLU  
  FC Layer 3: 16 → 1
  
  Output: scalar risk score (higher = worse survival)

═══════════════════════════════════════════════════════════════
```

---

### Why Attention Matters for This Problem

The attention weights $\alpha_{ij}$ tell us **which cell-cell interactions matter for survival**:

```
BIOLOGICAL INTERPRETATION:
═══════════════════════════════════════════════════════════════

High attention (α ≈ 0.8):
  T-cell ═══════════ Tumor cell
  
  → The model learned that T-cell–Tumor interactions 
    are IMPORTANT for predicting survival
  
───────────────────────────────────────────────────────────────

Low attention (α ≈ 0.05):
  Fibroblast - - - - - Fibroblast
  
  → The model learned that Fibroblast–Fibroblast 
    interactions are NOT very predictive

═══════════════════════════════════════════════════════════════
```

### Visualization of Learned Attention

```
TISSUE IMAGE WITH ATTENTION OVERLAY:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     ○───○                    🔴═══🔴                        │
│     │   │                    ║   ║                          │
│     ○───○                    🔴═══🔴                        │
│                                                             │
│   Low attention             High attention                  │
│   region (cold)             region (hotspot)                │
│                                                             │
│   → Normal tissue           → Immune-tumor interface        │
│     (not predictive)          (drives survival)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Colors: 🔴 High attention  🟡 Medium  🟢 Low  ○ Very low
```

### Output
- **Patient-level risk score** (single scalar per patient)
- **Attention weights** $\alpha_{ij}$ for all edges (interpretability)

---

## Step 6: Survival Prediction

### Input
- Risk scores from GNN
- True survival data: `(time, event)`

### Process: Survival Loss Functions

#### Cox Partial Likelihood Loss:
$$\mathcal{L}_{Cox} = -\sum_{i: E_i=1} \left( h_i - \log \sum_{j: T_j \geq T_i} \exp(h_j) \right)$$

Where:
- $h_i$ = predicted risk score for patient i
- $E_i$ = event indicator (1 = death, 0 = censored)
- $T_i$ = survival time

This loss **properly handles censored data** (patients who are still alive or lost to follow-up).

#### DeepSurv Extension:
Adds neural network complexity to Cox model with regularization:
$$\mathcal{L} = \mathcal{L}_{Cox} + \lambda ||\theta||^2$$

### Training Loop:
```python
for epoch in range(epochs):
    # Forward pass
    risk_scores = GAT_model(graph)
    
    # Compute survival loss
    loss = cox_partial_likelihood(risk_scores, survival_time, event)
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

### Output
- **Trained model** with optimized parameters
- **Risk scores** that correlate with survival probability

---

## Step 7: Evaluation

### Input
- Predicted risk scores
- True survival outcomes

### Process

| Metric | Formula/Method | Interpretation |
|--------|----------------|----------------|
| **C-index** (Concordance) | Proportion of correctly ordered pairs | 0.5 = random, 1.0 = perfect |
| **Kaplan-Meier Curves** | Stratify patients by predicted risk | Visual survival separation |
| **Log-rank Test** | Statistical test between risk groups | p < 0.05 = significant |

#### C-index Calculation:
$$C = \frac{\sum_{i,j} \mathbb{1}[h_i > h_j] \cdot \mathbb{1}[T_i < T_j] \cdot E_i}{\sum_{i,j} \mathbb{1}[T_i < T_j] \cdot E_i}$$

### Output
- **Performance metrics**: C-index, hazard ratios
- **Survival curves** for high vs. low risk groups
- **Statistical significance** (p-values)

---

## Step 8: Interpretation ⭐⭐⭐

### Input
- Trained model with attention weights
- Original tissue images/coordinates

### Process

| Method | Description | Insight |
|--------|-------------|---------|
| **Attention Visualization** | Overlay $\alpha_{ij}$ on tissue | Which cell interactions matter? |
| **High-Attention Cell Types** | Identify frequently attended cells | Which cell types drive prognosis? |
| **Ablation Studies** | Remove high-attention regions, measure performance drop | Validate importance |
| **Spatial Patterns** | Cluster attention patterns | Discover prognostic microenvironments |

### Visualization Pipeline:
```
Tissue Image
    │
    ▼ Overlay attention weights
    │
┌─────────────────────────────────────┐
│  🔴 High attention (survival-linked) │
│  🟡 Medium attention                 │
│  🟢 Low attention                    │
└─────────────────────────────────────┘
    │
    ▼ Biological interpretation
    │
"Immune-excluded regions correlate with poor survival"
```

### Output
- **Attention heatmaps** overlaid on tissue
- **Identified biomarkers** (spatial protein patterns)
- **Biological hypotheses** for validation

---

## Summary Table

### Input/Output at Each Step

| Step | Input | Output | Key Method |
|------|-------|--------|------------|
| **1. Data Collection** | Raw tissue | Cell × Protein matrix + coordinates + survival | CODEX/IMC/MIBI |
| **2. Preprocessing** | Raw data | Normalized, filtered data | Z-score, batch correction |
| **3. Graph Construction** | Cells + coordinates | Graph G = (V, E) | KNN, Delaunay |
| **4. Feature Engineering** | Basic graph | Enriched graph with features | Density, entropy |
| **5. GNN Model** | Graph | Risk scores + attention weights | GAT, multi-head attention |
| **6. Survival Prediction** | Risk scores + outcomes | Trained model | Cox loss |
| **7. Evaluation** | Predictions vs. truth | C-index, KM curves | Statistical tests |
| **8. Interpretation** | Attention weights | Biomarkers, heatmaps | Visualization |

---

### Concept Summary

| Concept | What It Is | Why It Matters |
|---------|------------|----------------|
| **Basic Graph** | Just cells + connections | Minimal information |
| **Enriched Graph** | Cells + connections + computed features | More biological meaning |
| **Node Features** | Cell type, density, entropy, centrality | Describe each cell's context |
| **Edge Features** | Distance, similarity, interaction type | Describe relationships |
| **Message Passing** | Nodes share info with neighbors | Core GNN operation |
| **Attention Weights** | How much to listen to each neighbor | Learnable importance |
| **Multi-Head Attention** | Multiple attention patterns | Capture diverse relationships |
| **Graph Pooling** | Aggregate cells → patient | Patient-level prediction |

---

## Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE PIPELINE OVERVIEW                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │    STEP 1    │     │    STEP 2    │     │    STEP 3    │     │    STEP 4    │    │
│  │    Data      │────▶│  Preprocess  │────▶│    Graph     │────▶│   Feature  |    │
│  │  Collection  │     │              │     │ Construction │     │ Engineering  │    │
│  └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘    │
│        │                    │                    │                    │             │
│        ▼                    ▼                    ▼                    ▼             │
│   Raw tissue          Normalized          Cell graph           Enriched             │
│   + survival          clean data          G = (V, E)           graph                │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │    STEP 5    │     │    STEP 6    │     │    STEP 7    │     │    STEP 8    │    │
│  │  Attention   │────▶│   Survival   │────▶│  Evaluation  │────▶│Interpretation│   │
│  │  GNN Model   │     │  Prediction  │     │              │     │              │    │
│  └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘    │
│        │                    │                    │                    │             │
│        ▼                    ▼                    ▼                    ▼             │
│   Risk scores         Trained model        C-index,             Biomarkers,         │
│   + attention         + predictions        KM curves            heatmaps            │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```
