# Spatial and Temporal Analysis of Cardiac Neural Crest and Schwann Progenitor Cells

Computational analysis of **cardiac neural crest cells (cNCCs)** and **Schwann progenitor cells (SPCs)** during human heart development, integrating **Spatial Transcriptomics (ST)** and **In Situ Sequencing (ISS)**.

The project characterizes the spatial and temporal distribution of cNCC- and SPC-like populations and tests whether patterns inferred from Spatial Transcriptomics are supported by single-cell spatial observations from In Situ Sequencing.

---

## Table of Contents

- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Analysis Pipeline](#analysis-pipeline)
- [Methods](#methods)
- [Developmental Timepoints](#developmental-timepoints)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Key Research Question](#key-research-question)
- [Results](#results)
- [Limitations](#limitations)
- [References](#references)
- [Author](#author)

---

## Overview

Cardiac neural crest cells and Schwann progenitor cells play important roles in the development of the peripheral nervous system and cardiovascular system. Understanding where these populations occur during human heart development can provide insight into their developmental trajectories and spatial organization.

This project integrates two complementary spatial technologies:

| Modality | What it provides |
|---|---|
| **Spatial Transcriptomics (ST)** | Gene-expression measurements across spatially resolved tissue spots |
| **In Situ Sequencing (ISS)** | Spatially resolved single-cell measurements via transcript localization |

**Central validation question:**

> Do regions with high cNCC/SPC signal in Spatial Transcriptomics correspond to regions containing high densities of cNCC/SPC cells identified through In Situ Sequencing?

The analysis spans multiple developmental stages measured in **post-conception weeks (PCW)**.

## Research Objectives

1. **Characterize cNCC and SPC populations** — identify cNCC-like and SPC-like cells in the developmental heart single-cell reference data and define marker-based expression signatures.
2. **Analyze spatial organization** — score ST spots for cNCC-like and SPC-like expression and visualize their distribution across developing heart tissue.
3. **Analyze temporal patterns** — compare cNCC/SPC-associated signals across developmental stages and examine how spatial distribution changes over time.
4. **Validate ST using ISS** — align ST and ISS spatial coordinates, aggregate ISS cell counts around ST spots, compare ST-derived scores with ISS-derived cell densities (Pearson/Spearman), and identify regions of disagreement.

## Analysis Pipeline

```text
                 Developmental Heart Data
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
  Single-cell Reference          Spatial Transcriptomics
          │                             │
  cNCC/SPC clustering                Normalization
  Marker scoring                     Gene ID mapping
  Subtype assignment                 PCA / UMAP
  PCA + differential expression      Spatial scoring
          │                             │
          └──────────┬──────────────────┘
                      ▼
            cNCC/SPC Signatures
                      ▼
             ST Spatial Score Maps
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   ST Scores                   ISS Cells
                          (cell-type annotation +
                           spatial coordinates)
        └─────────────┬─────────────┘
                      ▼
            Coordinate Alignment
                      ▼
        ISS → ST Spatial Aggregation
                      ▼
             Concordance Analysis
        ┌─────────────┴─────────────┐
        ▼                           ▼
 Pearson Correlation        Spearman Correlation
        └─────────────┬─────────────┘
                      ▼
             Validation Summary
```

## Methods

### 1. Single-Cell Reference Analysis

Cells annotated as *cardiac neural crest cells & Schwann progenitor cells* are extracted from the single-cell developmental heart dataset for downstream analysis.

Marker-based scores are computed from:

- **cNCC markers:** `ISL1`, `STMN2`
- **SPC marker:** `ALDH1A1`

Each cell is assigned a predicted subtype based on relative scores:

```text
cNCC score > SPC score  →  Cardiac neural crest-like
otherwise               →  Schwann progenitor-like
```

### 2. PCA and Differential Expression

PCA examines transcriptional separation between the predicted cNCC-like and SPC-like populations. Differential expression is then computed between the two groups, calculating for each gene:

- Mean expression per group
- Log2 fold change
- Welch's t-test p-value

The resulting differentially expressed genes form the expression signatures used in the spatial analysis.

### 3. Spatial Transcriptomics Analysis

ST data are library-size normalized and log-transformed, and Ensembl gene IDs are converted to gene symbols to align with the single-cell-derived signatures.

The most differentially enriched genes construct a **cNCC-like signature** and an **SPC-like signature**. Each ST spot then receives a `cNCC-like score` and an `SPC-like score`, visualized spatially to identify enriched regions. This stage also includes PCA, UMAP, spatial score maps, marker visualization, and expression heatmaps.

### 4. In Situ Sequencing Analysis

The ISS pipeline processes cell and transcript information, assigns cells to developmental populations (including cNCC and SPC), and retains spatial coordinates — providing the single-cell spatial reference against which ST spot-level predictions are evaluated.

### 5. Cross-Modality Validation

ST and ISS coordinates are independently normalized to a common coordinate system. For each ST spot, nearby ISS cells are identified within a set spatial radius using a **KD-tree spatial search**, and counted:

```text
ST cNCC score  ↔  Number of nearby ISS cNCC cells
ST SPC score   ↔  Number of nearby ISS SPC cells
```

Concordance is evaluated with Pearson and Spearman correlations (with p-values), computed independently for each developmental stage. Regions with large ST/ISS discrepancies are flagged as potential mismatches.

## Developmental Timepoints

| Stage | Data |
|---|---|
| 4.5 PCW | Spatial Transcriptomics + ISS |
| 6.5 PCW | Spatial Transcriptomics + ISS |
| 9.5 PCW | Spatial Transcriptomics + ISS |

Results are summarized both within individual stages and across PCW timepoints.

## Repository Structure

```text
.
├── README.md
├── report.pdf
│
├── code/
│   ├── st.py
│   ├── st-update.py
│   ├── iss.py
│   ├── cross_madality_analysis.py
│   ├── compute_partial_score.py
│   └── evaluation.py
│
├── plots/
│
└── results/
    ├── analysis_output/
    ├── iss_results/
    ├── results_st/
    └── validation_output/
```

## Requirements

Implemented in Python 3, using:

NumPy · Pandas · SciPy · scikit-learn · scikit-image · Scanpy · Squidpy · Matplotlib · Seaborn · UMAP · MyGene · tifffile

```bash
pip install numpy pandas scipy scikit-learn scikit-image \
    scanpy squidpy matplotlib seaborn umap-learn \
    mygene tifffile
```

## Dataset

This project uses publicly available developmental human heart datasets:

| Dataset | Source |
|---|---|
| Spatial Transcriptomics (v3) | [data.mendeley.com/datasets/compare/mbvhhf8m62](https://data.mendeley.com/datasets/compare/mbvhhf8m62?old=2&new=3) |
| In Situ Sequencing | [figshare.com — ISS data, developing human heart](https://figshare.com/articles/dataset/ISS_data_in_A_spatiotemporal_organ-wide_gene_expression_and_cell_atlas_of_the_developing_human_heart_/10058048/1) |

## Running the Pipeline

After downloading the datasets, place them in the directory structure shown above.

**1. Spatial Transcriptomics**

```bash
cd code
python3 st.py
```

Runs: single-cell reference processing → cNCC/SPC marker scoring → subtype prediction → PCA → differential expression → ST normalization → gene ID conversion → cNCC/SPC signature construction → ST spatial scoring → spatial visualization → ST PCA/UMAP.

**2. In Situ Sequencing**

```bash
cd code
python3 iss.py
```

Processes spatially resolved transcript/cell information into the cell-level data used downstream.

**3. Cross-Modality Validation**

```bash
cd code
python3 cross_madality_analysis.py
```

Loads ST scores and ISS annotations → normalizes coordinates → finds nearby ISS cells per ST spot → aggregates cNCC/SPC counts → computes Pearson/Spearman correlations → generates concordance plots → flags mismatched regions → produces per-stage and combined validation summaries.

> Uses a normalized spatial radius of `0.03` for aggregating ISS cells around ST spots.

**4. Evaluation**

```bash
cd code
python3 evaluation.py
```

## Outputs

**Single-cell analysis**
```text
pca_cnc_vs_schwann_like.png
cnc_vs_schwann_like_DEG_results.csv
predicted_cnc_schwann_subtypes_metadata.csv
selected_marker_heatmap.png
```

**Spatial Transcriptomics**
```text
ST_cNCC_map.png
ST_Schwann_map.png
ST_cNCC_Schwann_scores.csv
ST_PCA.png
```

**Cross-modality validation** (per stage, plus combined outputs)
```text
validation_output/
├── 4.5PCW/
│   ├── validation_results.csv
│   ├── correlation_metrics.json
│   ├── concordance_cNCC.png
│   ├── concordance_SPC.png
│   ├── mismatches.csv
│   └── validation_summary.txt
├── 6.5PCW/  └── ...
├── 9.5PCW/  └── ...
├── combined_metrics.csv
└── pearson_r_across_pcw.png
```

## Key Research Question

> If Spatial Transcriptomics identifies a region as enriched for cNCC or SPC signal, that region should also contain a higher density of the corresponding cell type when measured using single-cell spatial data from ISS.

This provides a way to assess whether population-level spatial signals inferred from ST are consistent with independent single-cell spatial observations.

## Results

The main results are documented in [`report.pdf`](./report.pdf). The `results/` directory holds outputs from ST analysis, ISS analysis, cross-modality validation, and downstream evaluation.

Because the repository contains both intermediate analysis outputs and final validation results, this README intentionally does not reproduce numerical results — see the full report for figures and statistical analyses.

## Limitations

- ST measures expression at the spot level rather than observing individual cells directly.
- ST and ISS have different spatial resolutions and measurement characteristics.
- cNCC/SPC subtype assignment relies on marker-based expression scoring.
- ST scores and ISS cell counts are not directly equivalent quantitative measurements.
- Coordinate normalization and the chosen spatial aggregation radius can affect cross-modality correlations.
- Correlation between modalities indicates spatial concordance, not biological causality.

## References

**Primary dataset**
Asp, M. et al. *The spatial and temporal organization of the developing human heart.*
ISS data: https://figshare.com/articles/dataset/ISS_data_in_A_spatiotemporal_organ-wide_gene_expression_and_cell_atlas_of_the_developing_human_heart_/10058048/1

**Spatial Transcriptomics dataset**
https://data.mendeley.com/datasets/compare/mbvhhf8m62?old=2&new=3

**Full methodology, figures, and discussion:** [Project Report](./report.pdf)

## Author

**Yongchean Chhy**
University of Minnesota — B.S. Computer Science, 2026
GitHub: [@Yongchean-Chhy](https://github.com/Yongchean-Chhy)

**Xiangyu Zou**
University of Minnesota — Email: zouxx223@umn.edu
