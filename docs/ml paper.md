
# Attention-Based Graph Neural Network for Survival Prediction from Single-Cell Spatial Proteomics

---

## 📌 Table of Contents

1. [Motivation](#-motivation)
2. [Proposed Pipeline](#-proposed-pipeline)
3. [Why This Approach Makes Sense](#-why-this-approach-makes-sense)
4. [Key Considerations & Improvements](#-key-considerations--improvements)
5. [Literature Review & Gap Analysis](#-literature-review--gap-analysis)
6. [Research Gap](#-research-gap)
7. [Target Goals](#-target-goals)
8. [Expected Outcomes](#-expected-outcomes)
9. [Applications](#-applications)
10. [References](#-references)

---
```
Raw Data → Preprocessing → Graph Construction → Feature Engineering → GNN Model → Survival Prediction → Interpretation

```


## 🎯 Motivation

Cancer prognosis and treatment planning critically depend on understanding the **tumor microenvironment (TME)**—the complex ecosystem of tumor cells, immune cells, stromal cells, and their spatial organization. Traditional approaches analyze cells in isolation, missing crucial information about **cell-cell interactions** and **spatial context** that drive disease progression and patient outcomes.

**Single-cell spatial proteomics** technologies (CODEX, IMC, MIBI) now enable simultaneous measurement of dozens of proteins at single-cell resolution while preserving spatial coordinates. This rich data captures:

- **What proteins each cell expresses** (functional state)
- **Where each cell is located** (spatial context)
- **Which cells are neighbors** (interaction potential)

However, current analytical methods fail to fully leverage this spatial relational information for clinical outcome prediction. **Graph Neural Networks (GNNs)** with **attention mechanisms** offer a powerful solution by:

1. Naturally representing cells as nodes and spatial relationships as edges
2. Learning which cell-cell interactions are most predictive of survival
3. Providbing interpretable attention weights for biomarker discovery

This project aims to bridge the gap between spatial proteomics data and clinical utility by developing an **interpretable, attention-based GNN framework** for survival prediction.

---

## 🔬 Proposed Pipeline

### Overview Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Collection │───▶│  Preprocessing  │───▶│ Graph Construction│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Interpretation │◀───│Survival Prediction│◀───│ Attention-based │
│                 │    │                  │    │     GNN Model   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Step-by-Step Details

| Stage | Description | Key Components |
|-------|-------------|----------------|
| **1. Data Collection** | Gather spatial proteomics data | • Cell coordinates (X, Y)<br>• Protein intensity values<br>• Clinical survival data (time + censoring) |
| **2. Preprocessing** | Clean and normalize data | • Normalize protein intensities<br>• Filter low-quality cells<br>• Align spatial coordinates<br>• Select relevant protein markers |
| **3. Graph Construction** | Build cell-cell interaction graph | • Cells → Nodes<br>• Neighboring cells → Edges (distance threshold)<br>• Edge features: distances, protein co-expression |
| **4. Feature Engineering** | Enrich node and edge representations | • Node features: protein intensities<br>• Edge features: spatial relationships, protein similarity<br>• Optional: cluster cells into microenvironments |
| **5. Attention-based Model** | Train Graph Attention Network (GAT) | • Multi-head attention mechanism<br>• Learn importance of cells/interactions for survival |
| **6. Survival Prediction** | Predict patient outcomes | • Output: survival risk scores<br>• Loss: Cox partial likelihood / DeepSurv |
| **7. Evaluation** | Assess model performance | • Concordance index (C-index)<br>• Kaplan–Meier survival curves<br>• Predicted vs observed survival |
| **8. Interpretation** | Extract biological insights | • Visualize attention weights<br>• Identify spatial protein patterns linked to survival |

---

## 💡 Why This Approach Makes Sense

### Biological Rationale

Spatial proteomics is **ideal for graph-based learning** because:

> 🔹 **Cells don't exist in isolation**—their spatial context matters for tumor microenvironment, immune infiltration, and disease progression
>
> 🔹 **Graph neural networks naturally capture** cell-cell interactions and spatial dependencies
>
> 🔹 **Attention mechanisms can identify** which cellular neighborhoods drive survival outcomes

### Active Research Areas Using Similar Frameworks

| Domain | Technologies | Application |
|--------|--------------|-------------|
| Spatial Transcriptomics | Visium, MERFISH | GNN-based tissue analysis |
| Multiplexed Imaging | CODEX, IMC, MIBI | Cancer prognosis |
| Single-cell RNA-seq | 10x Genomics | Graph-based survival models |

### Pipeline Strengths

✅ **Graph construction from spatial data** is well-motivated—cells within microenvironments interact functionally

✅ **Attention mechanisms provide interpretability** (crucial for biomarker discovery)

✅ **Survival-specific loss functions** (Cox, DeepSurv) are appropriate for censored data

✅ **Evaluation with C-index and KM curves** follows gold-standard survival analysis practices

---

## ⚙️ Key Considerations & Improvements

### Graph Construction Strategies

| Strategy | Description | Advantage |
|----------|-------------|-----------|
| **K-Nearest Neighbors (KNN)** | Connect each cell to K closest neighbors | Often outperforms fixed-radius graphs |
| **Delaunay Triangulation** | Geometric triangulation of cells | More biologically realistic connectivity |
| **Hybrid Edge Weights** | Combine distance + protein similarity | Captures both spatial and functional relationships |

### Feature Engineering Enhancements

- 📊 Add **cell type labels** (if available) as categorical node features
- 📈 Compute **local density** and **diversity metrics** (Shannon entropy of neighbors)
- 🔗 Include **higher-order graph features** (clustering coefficient, centrality)

### Model Architecture Options

| Architecture | Description | Use Case |
|--------------|-------------|----------|
| **GAT** | Graph Attention Network | Baseline attention-based approach |
| **GATv2** | Improved dynamic attention | Better captures complex relationships |
| **GraphSAGE** | Inductive representation learning | Scalable to large graphs |
| **Transformer-based GNNs** | Self-attention over graphs | State-of-the-art performance |

**Advanced Features:**
- Multi-scale attention (local + global) for microenvironment and tissue-level patterns
- Hierarchical pooling for multi-scale spatial representations

### Survival Modeling Considerations

- Neural Cox proportional hazards or discrete-time survival models
- Handle **competing risks** (e.g., death from other causes)
- Stratify by **clinical covariates** (age, stage) to avoid confounding

### Challenges & Mitigations

| Challenge | Mitigation Strategy |
|-----------|---------------------|
| **Sample size** | Aim for 100+ patients with events for stable models |
| **Batch effects** | Apply batch correction for multi-center data |
| **Overfitting** | Use cross-validation, dropout, edge dropout, graph augmentation |
| **Censoring bias** | Ensure loss function properly handles right-censored data |

### Interpretation & Validation

- 🎨 **Overlay attention weights** on tissue images to visualize "hotspots"
- 🧪 **Ablation studies**: remove high-attention regions, measure performance drop
- ✅ **External validation** with independent cohorts or published biomarkers
- 🔬 **Biological connection**: correlate high-attention areas with immune exclusion, hypoxia markers

---

## 📚 Detailed Literature Review

### Overview

This project—**graph/attention-based survival prediction from single-cell spatial proteomics**—sits at the intersection of three key research domains:

1. **Spatial single-cell omics**
2. **Graph neural networks with attention mechanisms**
3. **Survival analysis and clinical outcome prediction**

Below is a comprehensive analysis of how existing papers align with each component of our proposed pipeline, organized by research focus.

---

### 🔷 Category 1: Core "Blueprint" Papers (Most Closely Related)

1. **Single-cell spatial graphs + GAT for survival from pathology images**  
   - Cell-level segmentation and classification (tumor, lymphocytes, fibroblasts, etc.), construct a **cell graph**, then train a **graph attention network** for survival prediction, with explicit focus on **cell–cell interactions in the tumor microenvironment**  (Li et al., 2023).  
   - Provides a near-direct template for: node = cell, edges by proximity, GAT, Cox-based survival loss, and attribution to identify spatial patterns linked to prognosis  (Li et al., 2023).  
   - Very close to your “Graph Construction → Attention-based GNN → Survival Prediction → Interpretation” outline.

2. **Graph deep learning on spatial protein profiles for outcome prediction**  
   - Uses **multiplex immunofluorescence (40-plex) spatial protein profiles** to build **cellular graphs** that characterize local tumor microenvironments; predicts **recurrence and survival** in head-and-neck and colorectal cancer  (Wu et al., 2022).  
   - Shows that modeling **local subgraphs** of spatial proteomics outperforms approaches based only on cell-type composition  (Wu et al., 2022).  
   - This is currently the most direct example of **spatial proteomics + GNN + survival**.

3. **PathoGraph: attention-based GNN on immunohistochemistry for survival**  
   - Builds graphs from whole-slide images of CD276 immunohistochemistry; uses an **attention-based GNN (abGCN)** to predict **individual patient survival** and visualize informative cells and cell–cell interactions  (Alzoubi et al., 2024).  
   - Strong precedent for: cell-level graph from protein staining, attention visualization, and survival modeling.

### Methods connecting spatial omics / proteomics and GNNs

4. **DGAT – Dual-Graph Attention Network for spatial CITE-seq**  
   - Integrates **transcriptomic, proteomic, and spatial information** in a heterogeneous graph, encoded with **GATs**, to impute **spatial protein expression** and reveal spatial cell states and immune phenotypes  (Wang et al., 2025).  
   - Demonstrates graph construction from spatial coordinates, use of both RNA and protein, and attention-based encoding, but not survival.

5. **SSGATE – dual-path graph attention auto-encoder for spatial & single-cell multi-omics**  
   - Constructs neighborhood graphs from **expression or spatial coordinates** (nodes = cells, edges from Euclidean distance), then uses **graph attention autoencoders** for integration of transcriptomics and proteomics in spatial data  (Lv et al., 2024).  
   - Gives concrete design choices for graph construction from spatial coordinates and multi-omics integration that can be borrowed for your preprocessing and feature engineering.

6. **GraphDEC – GNN for proteomic deconvolution including spatial proteomics**  
   - Applies graph-based deep learning to **bulk and spatial proteomics**, with graphs capturing similarity and spatial relationships, and demonstrates strong performance on real spatial proteomics datasets  (Dai et al., 2025).  
   - Focus is cell-type proportion inference, not survival, but it is an important precedent for **GNNs on proteomics (including spatial)**.

7. **SNOWFLAKE & related single-cell proteomics GNNs (review)**  
   - Reviewed as integrating **single-cell proteomics, morphology, and structural imaging data**, modeling single-cell neighborhoods as **spatial graphs** to forecast immune responses and characterize microenvironments  (Li et al., 2025).  
   - Shows that cell-level spatial proteomics + GNN is emerging, but generally not yet tied to survival.

### GAT-based survival prediction and multimodal fusion

8. **Graph attention-based fusion of pathology images and gene expression for NSCLC survival**  
   - Builds undirected graphs from pathology images, fuses their embeddings with gene expression via **attention**, and predicts **patient-specific survival**; uses specialized interpretability (Survival Activation Maps) to localize survival-relevant regions  (Zheng et al., 2024).  
   - Strong reference for survival loss functions, evaluation (C-index, KM curves), and attention-based interpretability.

9. **SurvGraph – hybrid graph attention network for gastric cancer survival**  
   - **Hybrid graph construction** from multiple image feature types, multi-head attention GNN, and survival prediction with high C-index across internal and external cohorts  (Zhao et al., 2025).  
   - Useful for design of multi-feature nodes/edges and survival evaluation framework.

10. **GAT-based multi-omics survival models**  
   - NSCLC survival prediction using multi-omics GAT  (Elbashir et al., 2024).  
   - Multi-omics fusion GAT for digestive system tumors survival and drug response  (Zhou et al., 2023).  
   - Multitask GAT for survival analysis across cancers (DQSurv)  (Duan et al., 2023).  
   - Cancer subtype prediction with GAT embeddings that better separate risk groups  (Tanvir et al., 2024).  
   - These support your choices of Cox partial likelihood / DeepSurv losses and GAT architectures for survival.

### Survey / methodological overviews (for RRL background)

11. **GNNs for single-cell omics**  
   - Comprehensive reviews of GNNs/GATs in **spatial transcriptomics, proteomics, and cell–cell interaction inference**, including methods that construct **cell–cell graphs using spatial coordinates or expression similarity**  (Lazaros et al., 2023; Li et al., 2025).  
   - Ideal for literature review sections on “GNNs in spatial omics” and justifying your choice of GAT and graph construction.

### Suggested structure for your literature review and gap analysis

#### Sections you can build

- GNNs and GATs for cancer survival prediction from imaging and omics  (Li et al., 2023; Zheng et al., 2024; Zhao et al., 2025; Elbashir et al., 2024; Zhou et al., 2023; Duan et al., 2023; Alzoubi et al., 2024; Tanvir et al., 2024)- Spatial single-cell omics + GNNs (mainly transcriptomics, some proteomics)  (Wang et al., 2025; Lv et al., 2024; Wu et al., 2022; Lazaros et al., 2023; Li et al., 2025; Dai et al., 2025)- Graph construction strategies for spatial data (cell-based, spot-based, distance thresholds, heterogeneous graphs)  (Li et al., 2023; Wang et al., 2025; Lv et al., 2024; Wu et al., 2022; Baul et al., 2024; Lazaros et al., 2023; Li et al., 2025)- Attention-based interpretability for spatial models and survival activation maps  (Li et al., 2023; Zheng et al., 2024; Wu et al., 2022; Alzoubi et al., 2024; Zeng et al., 2025)#### Clear gaps you can claim

Based on these works, there appears to be no method that simultaneously:

- Uses **true single-cell spatial proteomics at per-cell resolution** (not spot/multicell) as the primary modality,  
- Constructs **cell–cell graphs with both spatial and protein-intensity edge features**,  
- Trains a **GAT (or related attention-based GNN)** specifically for **survival prediction**,  
- And systematically visualizes **attention over both cells and cell–cell interactions** to link **spatial protein patterns to clinical survival**.

Existing works either:
- use spatial proteomics GNNs for deconvolution or microenvironment characterization without survival  (Wang et al., 2025; Lv et al., 2024; Li et al., 2025; Dai et al., 2025), or  
- use GNN/GAT for survival on histopathology or immunohistochemistry (image-derived features), not full multiplex single-cell proteomics  (Li et al., 2023; Zheng et al., 2024; Zhao et al., 2025; Wu et al., 2022; Alzoubi et al., 2024).

Positioning your project as filling this gap—“attention-based graph survival modeling directly on single-cell spatial proteomics with interpretable microenvironment patterns”—is well supported by these references.

---
## 🎯 Target Goals

### Primary Goals

| # | Goal | Description | Success Metric |
|---|------|-------------|----------------|
| **G1** | Novel Graph Construction | Develop a graph construction method for single-cell spatial proteomics | Effective cell-cell graphs encoding both spatial and protein features |
| **G2** | Attention-based GNN Architecture | Design and train a GAT-based model for survival prediction | C-index ≥ 0.70 on held-out test data |
| **G3** | Interpretable Analysis | Enable biologically meaningful interpretation of predictions | Attention weights correlate with known biological markers |
| **G4** | Validation | Validate on real spatial proteomics datasets | External cohort validation with consistent performance |

### Secondary Goals

| # | Goal | Description |
|---|------|-------------|
| **S1** | Reproducible Codebase | Create modular, reusable Python package for spatial proteomics + GNN research |
| **S2** | Benchmarking | Compare against existing survival prediction methods |
| **S3** | Biomarker Discovery | Identify novel spatial protein biomarkers for cancer prognosis |
| **S4** | Multi-cancer Generalization | Test framework across different cancer types |

---

## 📈 Expected Outcomes

### Technical Outcomes

| Outcome | Deliverable | Impact |
|---------|-------------|--------|
| **Novel Method** | End-to-end pipeline for attention-based survival prediction from spatial proteomics | Advances state-of-the-art in spatial omics analysis |
| **Model Architecture** | GAT-based model with multi-scale spatial attention | Reusable architecture for similar problems |
| **Software Package** | Open-source Python package | Enables reproducibility and community adoption |
| **Benchmark Results** | Comparative analysis with existing methods | Demonstrates superiority of proposed approach |

### Scientific Outcomes

| Outcome | Description | Impact |
|---------|-------------|--------|
| **Biomarker Discovery** | Identification of spatial protein patterns linked to survival | Potential clinical translation |
| **Microenvironment Insights** | Understanding of cell-cell interactions in tumor prognosis | Advances cancer biology knowledge |
| **Attention Visualizations** | Maps overlaid on tissue images showing survival-relevant regions | Interpretable AI for pathology |

### Expected Performance Metrics

| Metric | Target | Comparison |
|--------|--------|------------|
| **C-index** | ≥ 0.70 | Competitive with existing survival models |
| **Kaplan-Meier Separation** | p < 0.05 | Statistically significant risk stratification |
| **External Validation** | C-index drop < 0.05 | Generalizable across cohorts |

### Clinical Translation Potential

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Spatial         │     │  Attention-based │     │  Personalized    │
│  Proteomics      │────▶│  GNN Analysis    │────▶│  Risk Score      │
│  Tissue Sample   │     │                  │     │  + Biomarkers    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
                                               ┌──────────────────────┐
                                               │ Clinical Decision    │
                                               │ Support:             │
                                               │ • Prognosis          │
                                               │ • Treatment planning │
                                               │ • Patient stratification│
                                               └──────────────────────┘
```

---

## 🏥 Applications

### Primary Applications

| Application | Description | Benefit |
|-------------|-------------|---------|
| **Cancer Prognosis** | Predict patient survival from tumor microenvironment analysis | Improved risk assessment |
| **Biomarker Discovery** | Identify spatial protein signatures predictive of outcomes | New therapeutic targets |
| **Treatment Stratification** | Classify patients into risk groups for personalized therapy | Optimized treatment selection |
| **Drug Response Prediction** | Correlate spatial patterns with treatment response | Precision medicine |

### Target Cancer Types

| Cancer | Rationale |
|--------|-----------|
| 🔬 **Head and neck cancer** | Existing spatial proteomics datasets available |
| 🔬 **Colorectal cancer** | Well-characterized tumor microenvironment |
| 🔬 **Breast cancer** | Rich immune infiltration patterns |
| 🔬 **Lung cancer** | Complex microenvironment with clinical relevance |

### Broader Impact

| Domain | Potential Application |
|--------|----------------------|
| **Immuno-oncology** | Predict response to immunotherapy based on immune cell spatial organization |
| **Computational Pathology** | Augment pathologist assessment with quantitative spatial analysis |
| **Drug Development** | Identify patient subpopulations for clinical trial stratification |
| **Research** | Enable new hypotheses about spatial biology and disease progression |

### Application Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Patient Tissue Sample                                   │
│    │                                                             │
│    ▼                                                             │
│  ┌──────────────────────────────────────┐                       │
│  │ Spatial Proteomics Sequencing        │                       │
│  │ (CODEX, IMC, MIBI)                   │                       │
│  └──────────┬───────────────────────────┘                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │ Graph Construction + GAT Analysis    │                       │
│  │ • Cell-cell graphs                   │                       │
│  │ • Attention-based learning           │                       │
│  └──────────┬───────────────────────────┘                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │ Outputs:                             │                       │
│  │ • Survival risk score                │                       │
│  │ • High-risk spatial patterns         │                       │
│  │ • Attention heatmaps                 │                       │
│  │ • Prognostic biomarkers              │                       │
│  └──────────┬───────────────────────────┘                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────────────────┐                       │
│  │ Clinical Decision Support            │                       │
│  │ • Risk stratification                │                       │
│  │ • Treatment recommendations          │                       │
│  │ • Follow-up scheduling               │                       │
│  └──────────────────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Real-World Use Cases

#### Use Case 1: Personalized Treatment Planning
**Scenario:** A newly diagnosed breast cancer patient undergoes spatial proteomics profiling.

**Workflow:**
1. Tissue sample analyzed → spatial protein data generated
2. GAT model predicts 5-year survival probability: 45% (high-risk)
3. Attention maps reveal immune-excluded tumor regions
4. **Clinical Action:** Recommend aggressive treatment + immunotherapy

#### Use Case 2: Clinical Trial Enrollment
**Scenario:** Pharmaceutical company developing new immunotherapy drug.

**Workflow:**
1. Screen 500 patients using spatial proteomics + GAT
2. Identify 120 patients with spatial patterns indicating potential response
3. **Result:** Enriched trial population, higher success rate

#### Use Case 3: Post-Treatment Monitoring
**Scenario:** Monitor patient response to neoadjuvant therapy.

**Workflow:**
1. Pre-treatment spatial analysis → high-risk score
2. Post-treatment re-analysis → risk score drops to low-risk
3. Attention maps show increased immune infiltration
4. **Outcome:** Evidence of treatment efficacy at cellular level

---
##  References (Extended Bibliography)

The following table provides a comprehensive list of relevant papers with key metadata including takeaways, citations, and journal information:

| # | Title | Authors | Year | Citations | Key Takeaway | Journal | DOI |
|---|-------|---------|------|-----------|--------------|---------|-----|
| 1 | **Single-Cell Spatial Analysis of Histopathology Images for Survival Prediction via Graph Attention Network** | Li Z, Jiang Y, Liu L, Xia Y, Li R | 2023 | 4 | Single-cell spatial analysis using GAT improves survival prediction by classifying cell types and their spatial interactions | - | 10.1007/978-3-031-47076-9_12 |
| 2 | **Graph attention-based fusion of pathology images and gene expression for prediction of cancer survival** | Zheng Y, Conrad RD, Green E, et al. | 2024 | 15 | Attention-based fusion integrating pathology images and gene expression outperforms existing approaches for NSCLC survival | IEEE Trans Med Imaging (Q1) | 10.1109/tmi.2024.3386108 |
| 3 | **SurvGraph: A hybrid-graph attention network for survival prediction using WSI in gastric cancer** | Zhao Y, Li L, Yu X, et al. | 2025 | 2 | Hybrid-graph attention network for survival prediction in gastric cancer | Neural Networks (Q1) | 10.1016/j.neunet.2025.107607 |
| 4 | **DGAT: A Dual-Graph Attention Network for Inferring Spatial Protein Landscapes from Transcriptomics** | Wang H, Cody B, Osmanbeyoglu H | 2025 | 0 | DGAT infers spatial protein expression from transcriptomics for cancer and immunology | bioRxiv | 10.1101/2025.07.05.662121 |
| 5 | **Enhancing NSCLC Survival Prediction through Multi-Omics Integration Using GAT** | Elbashir MK, Almotilag A, et al. | 2024 | 6 | GAT model integrating multi-omics data significantly improves NSCLC survival prediction | Diagnostics (Q2) | 10.3390/diagnostics14192178 |
| 6 | **Multi-omics integration for single-cell and spatially resolved data based on dual-path GAT auto-encoder** | Lv T, Zhang Y, Liu J, et al. | 2024 | 12 | SSGATE combines single-cell and spatially resolved data with improved performance | Brief Bioinform (Q1) | 10.1093/bib/bbae450 |
| 7 | **Graph deep learning for characterization of tumour microenvironments from spatial protein profiles** | Wu Z, Trevino AE, Wu E, et al. | 2022 | 106 | Graph deep learning on spatial protein profiles identifies spatial motifs for cancer recurrence | Nat Biomed Eng (Q1) | 10.1038/s41551-022-00951-w |
| 8 | **Multi-omics fusion based on attention mechanism for survival and drug response in Digestive System Tumors** | Zhou L, Wang N, Zhu Z, et al. | 2023 | 6 | MFGAN model improves survival and drug response prediction for personalized medicine | Neurocomputing (Q1) | 10.1016/j.neucom.2023.127168 |
| 9 | **Integrating spatial transcriptomics and bulk RNA-seq: gene expression prediction through GATs** | Baul S, Ahmed K, Jiang Q, et al. | 2024 | 10 | STGAT predicts gene expression at spot-level, improving breast cancer sub-type prediction | Brief Bioinform (Q1) | 10.1093/bib/bbae316 |
| 10 | **Orchestrating information across tissues via multitask GAT for survival analysis** | Duan M, Wang Y, Zhao D, et al. | 2023 | 6 | DQSurv multitask GAT improves cancer survival analysis on 10 benchmark cancer types | Brief Bioinform (Q1) | 10.1093/bib/bbad238 |
| 11 | **Graph neural network approaches for single-cell data: a recent overview** | Lazaros K, Koumadorakis DE, et al. | 2023 | 12 | GNNs are revolutionizing biomedicine by revealing deep connections among genes and cells | Neural Comput Appl (Q1) | 10.1007/s00521-024-09662-6 |
| 12 | **PathoGraph: Attention-Based GNN for Prognostication Based on CD276 Labelling in Glioma** | Alzoubi I, Zhang L, Zheng Y, et al. | 2024 | 5 | Attention-based GNN predicts individual patient survival using CD276 labeling | Cancers (Q1) | 10.3390/cancers16040750 |
| 13 | **Gated Graph Attention Network for Cancer Prediction** | Qiu L, Li H, Wang M, Wang X | 2021 | 10 | GGAT model with gating mechanism improves cancer prediction accuracy | Sensors (Q1) | 10.3390/s21061938 |
| 14 | **Cox proportional hazards model with Bayesian neural network for survival prediction** | Faghiri F, Kohansal A | 2025 | 1 | Bayesian neural networks with Cox modeling effectively predict survival outcomes | Sci Rep (Q1) | 10.1038/s41598-025-16993-4 |
| 15 | **Graph neural networks for single-cell omics data: a review of approaches and applications** | Li S, Hua H, Chen S | 2025 | 13 | GNNs successfully applied across broad range of single-cell omics tasks | Brief Bioinform (Q1) | 10.1093/bib/bbaf109 |
| 16 | **MOGAT: A Multi-Omics Integration Framework Using GATs for Cancer Subtype Prediction** | Tanvir R, Islam MM, et al. | 2024 | 29 | MOGAT outperforms existing methods in cancer subtype prediction and risk stratification | Int J Mol Sci (Q1) | 10.3390/ijms25052788 |
| 17 | **Deciphering Cell Type Abundance in Proteomics Data Through GNNs** | Dai Z, Song Y, Qi T, et al. | 2025 | 0 | GraphDEC decodes cell type proportions in proteomic data with cross-species generalization | Adv Sci (Q1) | 10.1002/advs.202502987 |
| 18 | **Application of GAT for Breast Cancer Data and Explanation of Prediction Basis** | Zeng Q, Tamori S, Harada T | 2025 | 0 | GAT predicts breast cancer prognosis and identifies 55 critical genes | IEEE Access (Q1) | 10.1109/access.2025.3575946 |
| 19 | **A GNN-based spatial multi-omics data integration method for deciphering spatial domains** | Gao C, Yang C, Zhang L | 2025 | 0 | SpaMI integrates spatial multi-omics data using GNN and contrastive learning | PLOS Comput Biol (Q1) | 10.1371/journal.pcbi.1013546 |
| 20 | **GAT-based prediction of MicroRNA-disease causality in head and neck neoplasms** | Yadalam P, Ayyachamy S, et al. | 2025 | 0 | GAT-based prediction of microRNA-disease causality | Sci Rep (Q1) | 10.1038/s41598-025-24130-4 |

---


