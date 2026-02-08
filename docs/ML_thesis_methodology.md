# Chapter 3: Methodology

## 3.1 Overview of the Proposed Framework

This chapter presents a comprehensive methodology for developing an attention-based Graph Neural Network (GNN) framework to predict patient survival outcomes from single-cell spatial proteomics data. The proposed approach leverages the spatial organization of cells within the tumor microenvironment (TME) and employs Graph Attention Networks (GATs) to identify prognostically significant cell-cell interactions.

The methodology consists of eight interconnected stages: (1) data acquisition, (2) data preprocessing, (3) graph construction, (4) feature engineering, (5) attention-based GNN modeling, (6) survival prediction, (7) model evaluation, and (8) biological interpretation. Figure 3.1 illustrates the complete workflow of the proposed framework.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PROPOSED METHODOLOGY FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│   │   §3.2      │    │   §3.3      │    │   §3.4      │    │   §3.5      │     │
│   │    Data     │───▶│   Data      │───▶│   Graph     │───▶│  Feature    │     │
│   │ Acquisition │    │Preprocessing│    │Construction │    │ Engineering │     │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                   │             │
│                                                                   ▼             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│   │   §3.9      │    │   §3.8      │    │   §3.7      │    │   §3.6      │     │
│   │ Biological  │◀───│   Model     │◀───│  Survival   │◀───│  GAT Model  │     │
│   │Interpretation│   │ Evaluation  │    │ Prediction  │    │ Architecture│     │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                            Figure 3.1: Proposed Methodology Framework
```

---

## 3.2 Data Acquisition

### 3.2.1 Data Sources

The proposed framework utilizes single-cell spatial proteomics data generated from multiplexed imaging technologies. These technologies enable simultaneous measurement of multiple protein markers while preserving the spatial coordinates of individual cells within tissue sections. The primary data sources considered in this study include:

**Table 3.1: Spatial Proteomics Technologies**

| Technology | Full Name | Markers | Resolution | Principle |
|------------|-----------|---------|------------|-----------|
| CODEX | CO-Detection by indEXing | 40-60 | Single-cell | Cyclic antibody staining |
| IMC | Imaging Mass Cytometry | 40+ | ~1 μm | Metal-tagged antibodies |
| MIBI | Multiplexed Ion Beam Imaging | 40+ | ~260 nm | Secondary ion mass spectrometry |

### 3.2.2 Data Components

The acquired dataset comprises three essential components:

#### 3.2.2.1 Spatial Coordinates

Each cell $i$ is associated with a two-dimensional spatial coordinate $(x_i, y_i)$ representing its centroid position within the tissue section. These coordinates are measured in micrometers (μm) from a reference point, typically the top-left corner of the imaging field.

$$\mathbf{S} = \{(x_i, y_i) \mid i = 1, 2, \ldots, N\}$$

where $N$ denotes the total number of cells in the tissue sample.

#### 3.2.2.2 Protein Expression Matrix

For each cell, the expression levels of $M$ protein markers are quantified, forming a protein expression matrix:

$$\mathbf{X} \in \mathbb{R}^{N \times M}$$

where $X_{ij}$ represents the expression intensity of protein $j$ in cell $i$.

#### 3.2.2.3 Clinical Survival Data

Patient-level survival information is recorded as:

$$\mathbf{Y} = \{(T_p, E_p) \mid p = 1, 2, \ldots, P\}$$

where:
- $T_p$ = observed survival time (or time to last follow-up) for patient $p$
- $E_p$ = event indicator (1 = event occurred, 0 = censored)
- $P$ = total number of patients

### 3.2.3 Data Structure

The complete dataset for a single patient can be represented as:

**Table 3.2: Example Data Structure**

| Cell_ID | X_coord (μm) | Y_coord (μm) | CD3 | CD8 | CD45 | PD-L1 | ... | Patient_ID |
|---------|--------------|--------------|-----|-----|------|-------|-----|------------|
| cell_1  | 125.3        | 450.2        | 0.82| 0.21| 0.93 | 0.11  | ... | P001       |
| cell_2  | 127.1        | 448.5        | 0.15| 0.03| 0.34 | 0.72  | ... | P001       |
| cell_3  | 130.8        | 452.1        | 0.67| 0.58| 0.89 | 0.08  | ... | P001       |
| ...     | ...          | ...          | ... | ... | ...  | ...   | ... | ...        |

---

## 3.3 Data Preprocessing

Data preprocessing is essential for ensuring data quality and comparability across samples. This section describes the preprocessing pipeline applied to raw spatial proteomics data.

### 3.3.1 Intensity Normalization

Raw protein intensity values exhibit considerable variation due to technical factors. We employ multiple normalization strategies:

#### 3.3.1.1 Arcsinh Transformation  -------✔️

The arcsinh (inverse hyperbolic sine) transformation is commonly applied to mass cytometry data to stabilize variance and approximate normality:

$$x'_{ij} = \sinh^{-1}\left(\frac{x_{ij}}{c}\right) = \ln\left(\frac{x_{ij}}{c} + \sqrt{\left(\frac{x_{ij}}{c}\right)^2 + 1}\right)$$

where $c$ is a cofactor parameter, typically set to 5 for mass cytometry data.

#### 3.3.1.2 Z-score Standardization

Following transformation, per-marker z-score standardization is applied:

$$z_{ij} = \frac{x'_{ij} - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of marker $j$ across all cells.

#### 3.3.1.3 Quantile Normalization

For cross-sample comparability, quantile normalization ensures that the distribution of expression values is identical across samples:

$$x''_{ij} = F^{-1}_{ref}\left(F_s(x'_{ij})\right)$$

where $F_s$ is the empirical cumulative distribution function for sample $s$, and $F^{-1}_{ref}$ is the inverse of the reference distribution.

### 3.3.2 Quality Control and Filtering

#### 3.3.2.1 Cell-level Filtering

Low-quality cells are removed based on the following criteria:

1. **Total Signal Intensity**: Cells with total marker intensity below the 1st percentile or above the 99th percentile are excluded.

$$\text{Remove cell } i \text{ if: } \sum_{j=1}^{M} x_{ij} < Q_{0.01} \text{ or } \sum_{j=1}^{M} x_{ij} > Q_{0.99}$$

2. **Nuclear Marker Expression**: Cells lacking expected nuclear marker (e.g., DAPI) signal are flagged as potential debris.

3. **Morphological Criteria**: Cells with abnormal size or shape metrics are excluded.

#### 3.3.2.2 Marker-level Filtering

Protein markers are filtered based on:

1. **Variance Threshold**: Markers with near-zero variance are uninformative and excluded.
2. **Missing Value Rate**: Markers with excessive missing values (>20%) are removed.
3. **Biological Relevance**: Markers are retained based on known relevance to the tumor microenvironment.

### 3.3.3 Batch Effect Correction

When integrating data from multiple samples or experimental batches, batch effects must be addressed. We employ:

#### 3.3.3.1 ComBat

The ComBat algorithm uses an empirical Bayes framework to adjust for batch effects:

$$x^{adjusted}_{ijb} = \frac{x_{ijb} - \hat{\gamma}_{jb}}{\hat{\delta}_{jb}} \cdot \sigma_j + \mu_j$$

where $\hat{\gamma}_{jb}$ and $\hat{\delta}_{jb}$ are the estimated batch effect parameters for marker $j$ in batch $b$.

#### 3.3.3.2 Harmony

For single-cell data, Harmony iteratively clusters cells and removes batch-specific variation while preserving biological heterogeneity.

### 3.3.4 Spatial Coordinate Normalization

To enable cross-sample comparison, spatial coordinates are normalized per tissue section:

$$x'_i = \frac{x_i - x_{min}}{x_{max} - x_{min}}, \quad y'_i = \frac{y_i - y_{min}}{y_{max} - y_{min}}$$

This maps all coordinates to the unit square $[0, 1] \times [0, 1]$.

**Algorithm 3.1: Data Preprocessing Pipeline**

```
Input: Raw data matrix X ∈ ℝ^(N×M), coordinates S, batch labels B
Output: Preprocessed data X', filtered coordinates S'

1. Apply arcsinh transformation: X ← arcsinh(X/cofactor)
2. Apply z-score normalization per marker
3. Remove cells failing QC criteria
4. Remove low-variance markers
5. If multiple batches: Apply ComBat/Harmony correction
6. Normalize spatial coordinates to [0,1]
7. Return X', S'
```

---

## 3.4 Graph Construction

Graph construction transforms the preprocessed single-cell data into a graph representation suitable for GNN processing. Each cell becomes a node, and spatial relationships define edges.

### 3.4.1 Graph Definition

Formally, we construct a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where:

- $\mathcal{V} = \{v_1, v_2, \ldots, v_N\}$ is the set of nodes (cells)
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of edges (cell-cell connections)

Each node $v_i$ is associated with a feature vector $\mathbf{h}_i \in \mathbb{R}^F$ representing the cell's protein expression profile and engineered features.

### 3.4.2 Edge Construction Methods

We investigate three primary methods for constructing edges based on spatial proximity:

#### 3.4.2.1 Distance Threshold Method

Cells within a specified Euclidean distance $r$ are connected:

$$\mathcal{E}_{threshold} = \{(v_i, v_j) \mid d(v_i, v_j) \leq r, \, i \neq j\}$$

where the Euclidean distance is:

$$d(v_i, v_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

The threshold $r$ is typically set between 30-100 μm based on biological considerations of cell-cell interaction distances.

**Table 3.3: Recommended Distance Thresholds**

| Interaction Type | Typical Distance | Biological Rationale |
|------------------|------------------|----------------------|
| Direct contact   | 10-20 μm         | Physical membrane contact |
| Paracrine signaling | 50-100 μm     | Cytokine diffusion range |
| Local microenvironment | 100-200 μm | Tissue neighborhood |

#### 3.4.2.2 K-Nearest Neighbors (KNN) Method---------✔️

Each cell is connected to its $K$ spatially closest neighbors:

$$\mathcal{E}_{KNN} = \{(v_i, v_j) \mid v_j \in \text{KNN}_K(v_i)\}$$

This approach ensures uniform node degree regardless of local cell density.

**Advantages:**
- Handles variable cell densities
- Consistent graph structure
- Computationally efficient with KD-tree implementations

**Typical Parameter:** $K = 5$ to $K = 20$

#### 3.4.2.3 Delaunay Triangulation

Delaunay triangulation constructs a mesh by connecting cells such that no cell lies inside the circumcircle of any triangle:

$$\mathcal{E}_{Delaunay} = \{(v_i, v_j) \mid (v_i, v_j) \text{ is an edge in } \text{Delaunay}(\mathcal{V})\}$$

**Properties:**
- Maximizes minimum angle of triangles
- Biologically interpretable as natural tessellation
- Dual of Voronoi diagram

### 3.4.3 Adjacency Matrix Representation

The graph structure is encoded in an adjacency matrix $\mathbf{A} \in \{0, 1\}^{N \times N}$:

$$A_{ij} = \begin{cases} 1 & \text{if } (v_i, v_j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

For weighted graphs, edge weights can be incorporated:

$$W_{ij} = \begin{cases} w(v_i, v_j) & \text{if } (v_i, v_j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$

### 3.4.4 Graph Statistics

For each constructed graph, we compute summary statistics:

| Statistic | Definition | Typical Range |
|-----------|------------|---------------|
| Number of nodes | $|\mathcal{V}|$ | 1,000 - 100,000 |
| Number of edges | $|\mathcal{E}|$ | 5,000 - 500,000 |
| Average degree | $\frac{2|\mathcal{E}|}{|\mathcal{V}|}$ | 5 - 20 |
| Graph density | $\frac{2|\mathcal{E}|}{|\mathcal{V}|(|\mathcal{V}|-1)}$ | 0.0001 - 0.01 |
| Clustering coefficient | Average local clustering | 0.1 - 0.5 |

**Algorithm 3.2: Graph Construction**

```
Input: Preprocessed coordinates S' = {(x_i, y_i)}, method ∈ {threshold, KNN, Delaunay}, parameters
Output: Graph G = (V, E), Adjacency matrix A

1. Initialize V ← {v_1, ..., v_N} corresponding to N cells
2. Initialize E ← ∅
3. If method = "threshold":
      For each pair (v_i, v_j) where i < j:
          If d(v_i, v_j) ≤ r: E ← E ∪ {(v_i, v_j), (v_j, v_i)}
4. Else if method = "KNN":
      Build KD-tree from coordinates
      For each v_i:
          neighbors ← query_KNN(v_i, K)
          For each v_j in neighbors:
              E ← E ∪ {(v_i, v_j)}
5. Else if method = "Delaunay":
      Compute Delaunay triangulation
      E ← edges from triangulation
6. Construct adjacency matrix A from E
7. Return G = (V, E), A
```

---

## 3.5 Feature Engineering

Feature engineering enriches the graph representation by computing additional node and edge features that capture biological and topological information.

### 3.5.1 Node Feature Engineering

The initial node feature vector consists of normalized protein expression values. We augment this with derived features:

#### 3.5.1.1 Cell Type Assignment

Cells are classified into phenotypic categories based on protein marker combinations. We employ a gating strategy or clustering approach:

**Rule-based Gating Example:**

$$\text{CellType}(i) = \begin{cases}
\text{CD8}^+ \text{ T-cell} & \text{if } \text{CD3}_i > \tau \land \text{CD8}_i > \tau \\
\text{CD4}^+ \text{ T-cell} & \text{if } \text{CD3}_i > \tau \land \text{CD4}_i > \tau \\
\text{Macrophage} & \text{if } \text{CD68}_i > \tau \\
\text{Tumor cell} & \text{if } \text{CK}_i > \tau \\
\text{Other} & \text{otherwise}
\end{cases}$$

Cell types are encoded as one-hot vectors:

$$\mathbf{c}_i = [\mathbb{1}_{type_1}, \mathbb{1}_{type_2}, \ldots, \mathbb{1}_{type_T}] \in \{0, 1\}^T$$

#### 3.5.1.2 Local Cell Density

The local density around cell $i$ within radius $r$ is computed as:

$$\rho_i = \frac{|\{j : d(i, j) < r\}|}{\pi r^2}$$

This measures cells per unit area in the local neighborhood.

#### 3.5.1.3 Neighborhood Diversity (Shannon Entropy)

The heterogeneity of cell types in the neighborhood is quantified using Shannon entropy:

$$H_i = -\sum_{t=1}^{T} p_{it} \log_2(p_{it})$$

where $p_{it}$ is the proportion of cell type $t$ among the neighbors of cell $i$:

$$p_{it} = \frac{|\{j \in \mathcal{N}(i) : \text{type}(j) = t\}|}{|\mathcal{N}(i)|}$$

**Interpretation:**
- $H_i \approx 0$: Homogeneous neighborhood (all same cell type)
- $H_i \approx \log_2(T)$: Maximum diversity (equal proportions of all types)

#### 3.5.1.4 Graph Centrality Measures

We compute multiple centrality metrics to characterize each cell's topological importance:

**Degree Centrality:**
$$C_D(i) = \frac{|\mathcal{N}(i)|}{N - 1}$$

**Betweenness Centrality:**
$$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$, and $\sigma_{st}(i)$ is the number passing through $i$.

**Closeness Centrality:**
$$C_C(i) = \frac{N - 1}{\sum_{j \neq i} d_{shortest}(i, j)}$$

#### 3.5.1.5 Final Node Feature Vector

The complete node feature vector for cell $i$ is:

$$\mathbf{h}_i = [\underbrace{x_{i1}, \ldots, x_{iM}}_{\text{protein expression}}, \underbrace{c_{i1}, \ldots, c_{iT}}_{\text{cell type one-hot}}, \underbrace{\rho_i, H_i, C_D(i), C_B(i), C_C(i)}_{\text{derived features}}]$$

**Dimension:** $F = M + T + 5$

### 3.5.2 Edge Feature Engineering

Edge features capture the relationship between connected cells:

#### 3.5.2.1 Euclidean Distance

$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

Normalized to [0, 1] based on maximum edge distance in the graph.

#### 3.5.2.2 Protein Expression Similarity

Cosine similarity between protein expression profiles:

$$\text{sim}_{ij} = \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|}$$

#### 3.5.2.3 Cell Type Interaction Encoding

The interaction between cell types is encoded categorically:

$$\text{interaction}_{ij} = \text{type}(i) \times T + \text{type}(j)$$

This is converted to a one-hot vector of dimension $T^2$ (or $T(T+1)/2$ for symmetric encoding).

#### 3.5.2.4 Protein Expression Difference

The element-wise difference captures complementary marker expression:

$$\boldsymbol{\Delta}_{ij} = \mathbf{x}_i - \mathbf{x}_j \in \mathbb{R}^M$$

#### 3.5.2.5 Final Edge Feature Vector

$$\mathbf{e}_{ij} = [d_{ij}, \text{sim}_{ij}, \text{interaction}_{ij}^{onehot}, \boldsymbol{\Delta}_{ij}]$$

**Table 3.4: Summary of Engineered Features**

| Feature Type | Feature Name | Dimension | Description |
|--------------|--------------|-----------|-------------|
| **Node** | Protein expression | M | Normalized marker intensities |
| **Node** | Cell type | T | One-hot encoded phenotype |
| **Node** | Local density | 1 | Cells per unit area |
| **Node** | Neighborhood entropy | 1 | Cell type diversity |
| **Node** | Degree centrality | 1 | Number of connections |
| **Node** | Betweenness centrality | 1 | Bridging importance |
| **Node** | Closeness centrality | 1 | Proximity to all nodes |
| **Edge** | Distance | 1 | Euclidean distance |
| **Edge** | Similarity | 1 | Cosine similarity |
| **Edge** | Interaction type | T² | Cell type pair encoding |
| **Edge** | Expression difference | M | Protein differences |

---

## 3.6 Graph Attention Network Architecture

This section describes the Graph Attention Network (GAT) architecture employed for learning representations from the enriched cell graphs.

### 3.6.1 Theoretical Foundation

Graph Neural Networks operate on the principle of **message passing**, where nodes iteratively update their representations by aggregating information from neighboring nodes. The key insight of Graph Attention Networks is the use of **attention mechanisms** to learn the importance of each neighbor dynamically.

### 3.6.2 Single GAT Layer

A single GAT layer transforms node features through the following operations:

#### 3.6.2.1 Linear Transformation

Each node's feature vector is first projected into a latent space:

$$\mathbf{z}_i = \mathbf{W} \mathbf{h}_i$$

where $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is a learnable weight matrix, transforming features from dimension $F$ to $F'$.

#### 3.6.2.2 Attention Coefficient Computation

For each edge $(i, j)$, an attention coefficient is computed:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{z}_i \| \mathbf{z}_j]\right)$$

where:
- $\mathbf{a} \in \mathbb{R}^{2F'}$ is a learnable attention vector
- $[\cdot \| \cdot]$ denotes concatenation
- LeakyReLU uses negative slope $\alpha = 0.2$

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases}$$

#### 3.6.2.3 Attention Normalization

Attention coefficients are normalized across neighbors using softmax:

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

This ensures $\sum_{j \in \mathcal{N}(i)} \alpha_{ij} = 1$.

#### 3.6.2.4 Feature Aggregation

The new node representation is computed as an attention-weighted sum:

$$\mathbf{h}'_i = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{z}_j\right)$$

where $\sigma$ is a nonlinear activation function (ELU or ReLU).

### 3.6.3 Multi-Head Attention

To capture diverse interaction patterns, we employ $K$ independent attention heads:

$$\mathbf{h}'_i = \Big\|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)$$

where $\|$ denotes concatenation. For the final layer, averaging is used instead:

$$\mathbf{h}'_i = \sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)$$

### 3.6.4 Complete Model Architecture

The proposed architecture consists of multiple GAT layers followed by graph pooling and prediction layers:

**Figure 3.2: Model Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROPOSED GAT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT LAYER                                                                │
│  ───────────────────────────────────────────────────────────────────────    │
│  Node features: X ∈ ℝ^(N × F)    where F = M + T + 5                       │
│  Adjacency matrix: A ∈ {0,1}^(N × N)                                        │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  GAT LAYER 1                                                                │
│  ───────────────────────────────────────────────────────────────────────    │
│  • Multi-head attention: K₁ = 8 heads                                       │
│  • Output dimension per head: d₁ = 64                                       │
│  • Total output: N × (K₁ × d₁) = N × 512                                   │
│  • Activation: ELU                                                          │
│  • Dropout: p = 0.3                                                         │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  GAT LAYER 2                                                                │
│  ───────────────────────────────────────────────────────────────────────    │
│  • Multi-head attention: K₂ = 8 heads                                       │
│  • Output dimension per head: d₂ = 32                                       │
│  • Total output: N × (K₂ × d₂) = N × 256                                   │
│  • Activation: ELU                                                          │
│  • Dropout: p = 0.3                                                         │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  GAT LAYER 3 (Final)                                                        │
│  ───────────────────────────────────────────────────────────────────────    │
│  • Multi-head attention: K₃ = 1 head (averaged)                             │
│  • Output dimension: d₃ = 64                                                │
│  • Total output: N × 64 (cell embeddings)                                   │
│  • Activation: ELU                                                          │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  GRAPH POOLING                                                              │
│  ───────────────────────────────────────────────────────────────────────    │
│  • Attention-based pooling (learnable)                                      │
│  • Output: 1 × 64 (patient-level embedding)                                 │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  PREDICTION HEAD (MLP)                                                      │
│  ───────────────────────────────────────────────────────────────────────    │
│  • FC Layer 1: 64 → 32, ReLU, Dropout(0.3)                                 │
│  • FC Layer 2: 32 → 16, ReLU                                               │
│  • FC Layer 3: 16 → 1 (risk score)                                         │
│                                                                             │
│  OUTPUT: Scalar risk score h ∈ ℝ                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.6.5 Graph Pooling

To obtain a patient-level representation from cell-level embeddings, we employ attention-based graph pooling:

$$\mathbf{h}_{patient} = \sum_{i=1}^{N} \beta_i \mathbf{h}_i$$

where the attention weights are computed as:

$$\beta_i = \frac{\exp(\text{MLP}(\mathbf{h}_i))}{\sum_{j=1}^{N} \exp(\text{MLP}(\mathbf{h}_j))}$$

This allows the model to learn which cells are most informative for survival prediction.

### 3.6.6 Model Parameters

**Table 3.5: Model Hyperparameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Number of GAT layers | 3 | Depth of message passing |
| Attention heads (L1, L2) | 8 | Multi-head attention |
| Attention heads (L3) | 1 | Final layer (averaged) |
| Hidden dimensions | 512, 256, 64 | Per-layer output sizes |
| Dropout rate | 0.3 | Regularization |
| Negative slope (LeakyReLU) | 0.2 | Attention activation |
| Learning rate | 0.001 | Adam optimizer |
| Weight decay | 0.0001 | L2 regularization |

---

## 3.7 Survival Prediction

### 3.7.1 Problem Formulation

The survival prediction task aims to model the relationship between the graph representation and patient survival outcomes. Given:
- A cell graph $\mathcal{G}_p$ for patient $p$
- Observed survival time $T_p$
- Event indicator $E_p$

The model predicts a risk score $h_p = f(\mathcal{G}_p; \theta)$ where higher values indicate greater risk of the event (e.g., death).

### 3.7.2 Cox Proportional Hazards Framework

We adopt the Cox proportional hazards framework, which models the hazard function as:

$$\lambda(t | \mathcal{G}_p) = \lambda_0(t) \exp(h_p)$$

where $\lambda_0(t)$ is the baseline hazard and $h_p$ is the log-partial hazard predicted by the GNN.

### 3.7.3 Loss Function

#### 3.7.3.1 Cox Partial Likelihood Loss

The negative log partial likelihood serves as the loss function:

$$\mathcal{L}_{Cox} = -\sum_{i: E_i = 1} \left( h_i - \log \sum_{j \in \mathcal{R}(T_i)} \exp(h_j) \right)$$

where $\mathcal{R}(T_i) = \{j : T_j \geq T_i\}$ is the risk set at time $T_i$ (patients still at risk).

**Properties:**
- Handles right-censored data appropriately
- Does not require modeling the baseline hazard $\lambda_0(t)$
- Focuses on relative risk ranking

#### 3.7.3.2 Regularized Loss (DeepSurv)

To prevent overfitting, we add L2 regularization:

$$\mathcal{L} = \mathcal{L}_{Cox} + \lambda \|\theta\|_2^2$$

where $\theta$ represents all model parameters and $\lambda$ is the regularization strength.

### 3.7.4 Handling Censored Data

Right-censored observations (patients without observed events) are incorporated through the risk set formulation:
- Censored patients contribute to the risk set $\mathcal{R}(T_i)$ for events occurring before their censoring time
- They do not contribute to the sum over events (only patients with $E_i = 1$ are summed)

### 3.7.5 Training Procedure

**Algorithm 3.3: Model Training**

```
Input: Training graphs {G_p}, survival data {(T_p, E_p)}, hyperparameters
Output: Trained model parameters θ*

1. Initialize model parameters θ randomly
2. For epoch = 1 to max_epochs:
      a. Shuffle training data
      b. For each mini-batch B:
          i.   Forward pass: h_p = GAT(G_p; θ) for p ∈ B
          ii.  Compute Cox loss: L = L_Cox(h, T, E) + λ||θ||²
          iii. Backward pass: ∇_θ L
          iv.  Update: θ ← θ - η · Adam(∇_θ L)
      c. Evaluate on validation set
      d. Early stopping if no improvement for patience epochs
3. Return θ* with best validation C-index
```

### 3.7.6 Implementation Details

**Table 3.6: Training Configuration**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Learning rate scheduler | ReduceLROnPlateau |
| Batch size | 32 patients |
| Maximum epochs | 200 |
| Early stopping patience | 20 epochs |
| Regularization (λ) | 0.0001 |

---

## 3.8 Model Evaluation

### 3.8.1 Evaluation Metrics

#### 3.8.1.1 Concordance Index (C-index)

The primary metric is Harrell's concordance index, measuring the model's discriminative ability:

$$C = \frac{\sum_{i,j} \mathbb{1}[h_i > h_j] \cdot \mathbb{1}[T_i < T_j] \cdot E_i}{\sum_{i,j} \mathbb{1}[T_i < T_j] \cdot E_i}$$

**Interpretation:**
- $C = 0.5$: Random prediction (no discrimination)
- $C = 1.0$: Perfect discrimination
- $C > 0.7$: Generally considered acceptable for clinical use

#### 3.8.1.2 Time-Dependent AUC

For specific time points $t$, we compute the time-dependent AUC:

$$\text{AUC}(t) = P(h_i > h_j | T_i \leq t, T_j > t)$$

This assesses discrimination at clinically relevant timepoints (e.g., 1-year, 3-year, 5-year survival).

#### 3.8.1.3 Integrated Brier Score (IBS)

The Brier score measures calibration at time $t$:

$$BS(t) = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{S}(t | \mathcal{G}_i) - \mathbb{1}[T_i > t] \right)^2 \cdot w_i(t)$$

where $\hat{S}(t | \mathcal{G}_i)$ is the predicted survival probability and $w_i(t)$ are inverse probability of censoring weights. The IBS integrates over all time points.

### 3.8.2 Statistical Testing

#### 3.8.2.1 Log-rank Test

To assess whether predicted risk groups have significantly different survival, we stratify patients (e.g., by median risk score) and perform the log-rank test:

$$\chi^2 = \frac{\left( \sum_{i} (O_{1i} - E_{1i}) \right)^2}{\sum_i V_i}$$

where $O_{1i}$, $E_{1i}$, and $V_i$ are the observed events, expected events, and variance at time $i$ for group 1.

**Significance:** $p < 0.05$ indicates significant separation between risk groups.

#### 3.8.2.2 Confidence Intervals

Bootstrap resampling (1000 iterations) is used to estimate 95% confidence intervals for all metrics:

$$CI_{95\%} = [\hat{\theta}_{0.025}, \hat{\theta}_{0.975}]$$

### 3.8.3 Kaplan-Meier Analysis

Survival curves are estimated using the Kaplan-Meier estimator:

$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

where $d_i$ is the number of events at time $t_i$ and $n_i$ is the number at risk.

Patients are stratified into risk groups (e.g., low/high or tertiles) based on predicted risk scores, and Kaplan-Meier curves are plotted for visual assessment of discrimination.

### 3.8.4 Cross-Validation Strategy

We employ nested cross-validation to ensure robust evaluation:

**Outer Loop (5-fold):** Assessment of generalization performance
**Inner Loop (5-fold):** Hyperparameter tuning

```
For each outer fold k ∈ {1, ..., 5}:
    Test set: Fold k
    Training/Validation set: Remaining folds
    
    For each inner fold m ∈ {1, ..., 5}:
        Validation set: Fold m (from remaining)
        Training set: Other folds
        
        Train model, evaluate on validation
    
    Select best hyperparameters
    Retrain on full training set
    Evaluate on test set (Fold k)

Report: Mean ± SD of test performance across outer folds
```

### 3.8.5 Baseline Comparisons

The proposed method is compared against:

**Table 3.7: Baseline Methods**

| Method | Description |
|--------|-------------|
| Cox PH (clinical) | Cox regression on clinical variables only |
| Cox PH (protein) | Cox regression on mean protein expression |
| DeepSurv | Neural network-based Cox model |
| GraphSAGE-Surv | GraphSAGE with survival loss |
| GCN-Surv | Graph Convolutional Network with survival loss |

---

## 3.9 Interpretation and Biological Validation

### 3.9.1 Attention Weight Analysis

The learned attention weights $\alpha_{ij}$ provide insights into which cell-cell interactions are most predictive of survival.

#### 3.9.1.1 Edge-level Attention Aggregation

For interpretability, we aggregate attention across layers and heads:

$$\bar{\alpha}_{ij} = \frac{1}{L \cdot K} \sum_{l=1}^{L} \sum_{k=1}^{K} \alpha_{ij}^{(l,k)}$$

#### 3.9.1.2 Cell-level Importance

Node importance is computed by aggregating incoming attention:

$$\text{Importance}(i) = \sum_{j \in \mathcal{N}^{-1}(i)} \bar{\alpha}_{ji}$$

where $\mathcal{N}^{-1}(i)$ denotes nodes pointing to $i$.

#### 3.9.1.3 Cell Type Interaction Analysis

We construct an interaction importance matrix:

$$I_{st} = \frac{\sum_{(i,j) \in \mathcal{E}: \text{type}(i)=s, \text{type}(j)=t} \bar{\alpha}_{ij}}{|\{(i,j) \in \mathcal{E}: \text{type}(i)=s, \text{type}(j)=t\}|}$$

This reveals which cell type pairs have highest average attention.

### 3.9.2 Spatial Visualization

#### 3.9.2.1 Attention Heatmaps

Attention weights are overlaid on tissue coordinates to create spatial heatmaps:

1. Map each cell to its spatial position $(x_i, y_i)$
2. Color cells by their importance score
3. Draw edges with width/color proportional to attention weight

#### 3.9.2.2 High-Attention Region Identification

Regions with concentrated high attention are identified through spatial clustering:

1. Select cells with importance > threshold
2. Apply DBSCAN clustering to identify spatial clusters
3. Annotate clusters as "attention hotspots"

### 3.9.3 Ablation Studies

To validate the importance of identified features:

#### 3.9.3.1 Node Ablation

Remove high-attention cells and measure performance degradation:

$$\Delta C = C_{full} - C_{ablated}$$

Significant $\Delta C$ confirms the importance of these cells.

#### 3.9.3.2 Edge Ablation

Remove edges between specific cell type pairs:

$$\Delta C_{st} = C_{full} - C_{\text{remove } s \leftrightarrow t}$$

#### 3.9.3.3 Feature Ablation

Systematically remove feature groups to assess contribution:

| Removed Features | Expected Impact |
|------------------|-----------------|
| Cell type | Loss of phenotype information |
| Centrality metrics | Loss of network structure |
| Edge features | Loss of interaction context |

### 3.9.4 Biological Validation

#### 3.9.4.1 Known Biomarker Correlation

Compare high-attention patterns with established prognostic biomarkers:
- Tumor-infiltrating lymphocyte (TIL) density
- PD-L1 expression patterns
- Immune exclusion signatures

#### 3.9.4.2 Pathway Enrichment

Proteins associated with high-attention cells are analyzed for pathway enrichment using Gene Ontology (GO) and KEGG databases.

#### 3.9.4.3 External Cohort Validation

The model is validated on independent cohorts to assess generalizability:
- Apply trained model to external dataset
- Compare C-index and attention patterns
- Assess reproducibility of identified biomarkers

### 3.9.5 Interpretation Summary

**Table 3.8: Interpretation Outputs**

| Output | Description | Biological Utility |
|--------|-------------|-------------------|
| Attention heatmaps | Spatial visualization of predictive regions | Identify prognostic microenvironments |
| Cell type importance | Ranking of phenotypes by predictive value | Prioritize cell types for research |
| Interaction matrix | Importance of cell-cell interaction types | Understand protective/harmful interactions |
| Ablation results | Quantified feature importance | Validate biological relevance |

---

## 3.10 Implementation

### 3.10.1 Software Framework

The methodology is implemented using the following software stack:

**Table 3.9: Software Dependencies**

| Component | Library | Version |
|-----------|---------|---------|
| Deep Learning | PyTorch | 2.0+ |
| Graph Neural Networks | PyTorch Geometric | 2.3+ |
| Survival Analysis | lifelines, pycox | Latest |
| Data Processing | pandas, numpy | Latest |
| Visualization | matplotlib, seaborn | Latest |
| Spatial Analysis | squidpy, scanpy | Latest |

### 3.10.2 Computational Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | NVIDIA GTX 1080 (8GB) | NVIDIA A100 (40GB) |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB | 500 GB |
| Training time (per fold) | ~2 hours | ~30 minutes |

### 3.10.3 Reproducibility

To ensure reproducibility:
- Random seeds are fixed for all stochastic operations
- Model checkpoints are saved at each epoch
- Hyperparameters are logged using experiment tracking (e.g., MLflow)
- Code and data are version-controlled

---

## 3.11 Summary

This chapter presented a comprehensive methodology for survival prediction from single-cell spatial proteomics using attention-based Graph Neural Networks. The key methodological contributions include:

1. **Graph Construction**: Systematic approach to transform spatial proteomics data into cell graphs using distance-based, KNN, or Delaunay triangulation methods.

2. **Feature Engineering**: Comprehensive node and edge features capturing protein expression, cell phenotypes, local microenvironment characteristics, and network topology.

3. **GAT Architecture**: Multi-layer, multi-head Graph Attention Network with attention-based pooling for patient-level survival prediction.

4. **Survival Framework**: Integration of Cox proportional hazards loss for handling censored survival data.

5. **Interpretation Pipeline**: Attention weight analysis, spatial visualization, and ablation studies for biological insight extraction.

The proposed methodology addresses the research gap of jointly modeling spatial cell-cell interactions and protein expression for survival prediction, while maintaining interpretability through attention mechanisms.
