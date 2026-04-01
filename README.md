# 🔬 R-COV: Graph Neural Networks Use Graphs When They Shouldn’t  
**Replication Study • GDL 2025**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5%2B-orange?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**End-to-end replication** of the R-COV (Reduced Coefficient of Variation) method from Bechler-Speicher et al. (ICML 2024). We rigorously tested whether **adding synthetic edges to low-degree nodes** actually mitigates the topological overfitting bias in Graph Neural Networks.

> **Key Finding**: GNNs **do** exhibit a strong implicit bias toward graph structure — even when it carries zero information. However, **R-COV does NOT deliver consistent performance gains**, contrary to the original claims.

---

## ✨ Highlights
- **Confirmed the core hypothesis**: GNNs (GraphConv, GIN, GATv2, GraphTransformer) heavily rely on topology in the **Sum** synthetic task and real TUDatasets (Proteins, Enzymes, IMDB, Reddit, etc.).
- **Graph regularity matters**: More regular graphs (low COV) consistently outperform irregular ones (Star, BA).
- **R-COV result**: Mixed or neutral impact across 5 real-world datasets. No systematic improvement despite reducing degree variation.
- **Reproducibility challenges exposed**: Major discrepancies between paper and original code → we rebuilt everything from scratch in PyTorch Geometric.
- **Scale**: 10-fold CV, hyperparameter search, NVIDIA DGX Spark (20+ hours per config). Full results for 5/11 datasets due to compute limits.
- **Team**: Innocent Kisoka, Eduardas Lazebnyj, Manuel Romanelli (@usi.ch)

---

## 📊 What the Study Shows

### 1. Topological Overfitting is Real
- In the **Sum** synthetic task (labels depend only on node features), **GNN∅** (empty-graph baseline) beats every structured graph.
- **Weight ratio analysis** (`||W₂|| / ||W₁||`) proves models initially overweight topological weights → classic implicit bias.

### 2. Regularity is the Hidden Driver
- Regular & Erdős–Rényi graphs → near-empty-graph performance.
- Star & Barabási–Albert (high COV) → worst performance.
- R-COV was designed to fix this… but didn’t consistently help on real data.

### 3. R-COV Performance (Our Results vs Original)

**Table 1: Selected Real-World Results (mean ± std)**

| Model              | Graph     | Proteins     | Enzymes      | IMDB-B       | Reddit-B     |
|--------------------|-----------|--------------|--------------|--------------|--------------|
| DeepSets (Empty)   | –         | **80.5±3.2** | **69.5±4.8** | **75.1±3.9** | **91.0±1.8** |
| GraphConv          | Original  | 79.0±3.2     | 60.8±5.0     | 73.4±3.2     | 84.9±2.8     |
| GraphConv          | R-COV     | 78.0±2.9     | 55.8±5.2     | 74.7±2.8     | 85.5±2.1     |
| GIN                | Original  | 79.7±3.4     | 66.0±5.7     | 74.6±3.8     | 87.8±4.7     |
| GIN                | R-COV     | 79.8±3.8     | 64.2±5.0     | 74.9±3.5     | **91.0±2.2** |
| GATv2              | Original  | 78.9±3.6     | 57.7±5.4     | **76.1±3.7** | –            |
| GATv2              | R-COV     | 79.3±3.4     | 55.2±4.9     | 76.0±3.6     | –            |

*(Full tables in `results/` folder — 7 datasets, 4 architectures, 2 R-COV ratios)*

---

## 🛠️ Tech Stack & Implementation
- **Framework**: PyTorch + PyTorch Geometric (fully custom message-passing layers)
- **Models**: GraphConv, GIN, GATv2, GraphTransformer + DeepSets baseline
- **Edge feature trick**: Original edges = 1.0, R-COV synthetic = 0.5
- **Preprocessing**: R-COV algorithm (COV reduction to 80%/50%)
- **Evaluation**: 10-fold stratified CV (single seed due to compute), early stopping, Adam optimizer
- **Compute**: NVIDIA DGX Spark (Grace-Blackwell) — ~300+ GPU-hours total

---

## 📸 Key Visualizations (in repo)
- Accuracy curves across synthetic topologies (Empty vs Star vs Regular vs ER vs BA)
- Topological vs root weight ratio evolution
- PCA-style cluster views? No — training dynamics plots
- COV reduction effect on real datasets
- All figures from the report (high-res in `figures/`)

---

## 🚀 How to Reproduce

```bash
git clone https://github.com/yourusername/r-cov-gnn-replication-topological-overfitting.git
cd r-cov-gnn-replication-topological-overfitting
pip install -r requirements.txt

