import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import mygene


# =========================
# 1. Load data
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))

count_path = os.path.join(
    base_dir,
    "Filtered",
    "Developmental_heart_filtered_scRNA-seq_and_meta_data",
    "all_cells_count_matrix_filtered.tsv"
)

meta_path = os.path.join(
    base_dir,
    "Filtered",
    "Developmental_heart_filtered_scRNA-seq_and_meta_data",
    "all_cells_meta_data_filtered.tsv"
)

counts = pd.read_csv(count_path, sep="\t", index_col=0)
meta = pd.read_csv(meta_path, sep="\t", index_col=0)

counts.index.name = "gene"
meta.index.name = "cell_id"

print("Counts shape:", counts.shape)
print("Meta shape:", meta.shape)

counts_only = set(counts.columns) - set(meta.index)
meta_only = set(meta.index) - set(counts.columns)

print("In counts but not meta:", len(counts_only))
print("In meta but not counts:", len(meta_only))
print("Example only in counts:", list(counts_only)[:10])
print("Example only in meta:", list(meta_only)[:10])

common_cells = counts.columns.intersection(meta.index)
counts = counts[common_cells]
meta = meta.loc[common_cells]

print("Aligned counts shape:", counts.shape)
print("Aligned meta shape:", meta.shape)

# =========================
# 2. Extract combined cluster
# =========================
target_label = "Cardiac neural crest cells & Schwann progenitor cells"
target_cells = meta.index[meta["celltype"] == target_label].tolist()

print("\nNumber of target cells:", len(target_cells))

sub_expr = counts[target_cells].T   # cells x genes
print("Subset expression shape:", sub_expr.shape)

# =========================
# 3. Marker-based scoring
# =========================
cnc_markers = ["ISL1", "STMN2"]
schwann_markers = ["ALDH1A1"]

available_cnc = [g for g in cnc_markers if g in sub_expr.columns]
available_schwann = [g for g in schwann_markers if g in sub_expr.columns]

print("\nAvailable CNC markers:", available_cnc)
print("Available Schwann markers:", available_schwann)

sub_expr["cnc_score"] = sub_expr[available_cnc].mean(axis=1)
sub_expr["schwann_score"] = sub_expr[available_schwann].mean(axis=1)

def assign_subtype(row):
    if row["cnc_score"] > row["schwann_score"]:
        return "Cardiac neural crest-like"
    else:
        return "Schwann progenitor-like"

sub_expr["predicted_subtype"] = sub_expr.apply(assign_subtype, axis=1)

print("\nPredicted subtype counts:")
print(sub_expr["predicted_subtype"].value_counts())

# =========================
# 4. Merge back to metadata
# =========================
sub_meta = meta.loc[target_cells].copy()
sub_meta = sub_meta.join(
    sub_expr[["cnc_score", "schwann_score", "predicted_subtype"]],
    how="left"
)

# =========================
# 5. PCA
# =========================
gene_expr_only = sub_expr.drop(columns=["cnc_score", "schwann_score", "predicted_subtype"])

gene_var = gene_expr_only.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(500).index
pca_input = gene_expr_only[top_genes]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(pca_input)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

sub_meta["PC1"] = X_pca[:, 0]
sub_meta["PC2"] = X_pca[:, 1]

plt.figure(figsize=(6, 5))
for subtype in sub_meta["predicted_subtype"].unique():
    idx = sub_meta["predicted_subtype"] == subtype
    plt.scatter(
        sub_meta.loc[idx, "PC1"],
        sub_meta.loc[idx, "PC2"],
        label=subtype,
        alpha=0.7
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA of cardiac neural crest / Schwann progenitor cluster")
plt.legend()
plt.tight_layout()
plt.savefig("pca_cnc_vs_schwann_like.png", dpi=300)
#plt.show()

# =========================
# 6. DEG
# =========================
group1 = sub_meta.index[sub_meta["predicted_subtype"] == "Cardiac neural crest-like"]
group2 = sub_meta.index[sub_meta["predicted_subtype"] == "Schwann progenitor-like"]

expr1 = gene_expr_only.loc[group1]
expr2 = gene_expr_only.loc[group2]

results = []
for gene in gene_expr_only.columns:
    x1 = expr1[gene]
    x2 = expr2[gene]
    stat, pval = ttest_ind(x1, x2, equal_var=False)
    mean1 = x1.mean()
    mean2 = x2.mean()
    log2fc = np.log2(mean1 + 1e-6) - np.log2(mean2 + 1e-6)
    results.append([gene, mean1, mean2, log2fc, pval])

deg = pd.DataFrame(results, columns=["gene", "mean_cnc", "mean_schwann", "log2FC", "pvalue"])
deg = deg.sort_values("pvalue")
deg.to_csv("cnc_vs_schwann_like_DEG_results.csv", index=False)

print("\nTop DEG results:")
print(deg.head(20))

top_up_cnc = deg.sort_values("log2FC", ascending=False).head(10)
top_up_schwann = deg.sort_values("log2FC", ascending=True).head(10)

print("\nTop genes enriched in Cardiac neural crest-like cells:")
print(top_up_cnc[["gene", "mean_cnc", "mean_schwann", "log2FC", "pvalue"]])

print("\nTop genes enriched in Schwann progenitor-like cells:")
print(top_up_schwann[["gene", "mean_cnc", "mean_schwann", "log2FC", "pvalue"]])

# =========================
# 7. Marker boxplots
# =========================
markers_to_plot = [g for g in ["ISL1", "STMN2", "ALDH1A1"] if g in gene_expr_only.columns]

cnc_cells = sub_meta.index[
    sub_meta["predicted_subtype"] == "Cardiac neural crest-like"
].tolist()

schwann_cells = sub_meta.index[
    sub_meta["predicted_subtype"] == "Schwann progenitor-like"
].tolist()

for gene in markers_to_plot:
    plt.figure(figsize=(5, 4))
    data = [
        gene_expr_only.loc[cnc_cells, gene],
        gene_expr_only.loc[schwann_cells, gene]
    ]
    plt.boxplot(data, tick_labels=["CNC-like", "Schwann-like"])
    plt.title(gene)
    plt.ylabel("Expression")
    plt.tight_layout()
    plt.savefig(f"{gene}_boxplot.png", dpi=300)
    #plt.show()


# =========================
# 8. Heatmaps
# =========================

heatmap_genes = [
    "ISL1", "CD24", "TAGLN3", "APLP1",
    "ALDH1A1", "ERBB3", "MPZ", "S100B", "PLP1"
]

heatmap_genes = [g for g in heatmap_genes if g in gene_expr_only.columns]

heatmap_df = gene_expr_only.loc[
    cnc_cells + schwann_cells,
    heatmap_genes
].copy()

row_labels = sub_meta.loc[cnc_cells + schwann_cells, "predicted_subtype"]

# z-score by gene
heatmap_df = (heatmap_df - heatmap_df.mean()) / heatmap_df.std(ddof=0)

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_df.T,
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Z-score"}
)
plt.title("Selected marker genes in CNC-like and Schwann-like cells")
plt.xlabel("Cells")
plt.ylabel("Genes")
plt.tight_layout()
plt.savefig("selected_marker_heatmap.png", dpi=300)
plt.savefig("subtype_plot.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# 9. Save metadata
# =========================
sub_meta.reset_index().rename(columns={"index": "cell_id"}).to_csv(
    "predicted_cnc_schwann_subtypes_metadata.csv",
    index=False
)

print("\nSaved outputs:")
print("- pca_cnc_vs_schwann_like.png")
print("- cnc_vs_schwann_like_DEG_results.csv")
print("- predicted_cnc_schwann_subtypes_metadata.csv")


# =========================
# 10. Load ST data
# =========================

st_count_path = os.path.join(base_dir, "Filtered", "filtered_ST_matrix_and_meta_data", "filtered_matrix.tsv")
st_meta_path = os.path.join(base_dir, "Filtered", "filtered_ST_matrix_and_meta_data", "meta_data.tsv")

st_counts = pd.read_csv(st_count_path, sep="\t", index_col=0)
st_meta = pd.read_csv(st_meta_path, sep="\t", index_col=0)

print("\nST counts shape:", st_counts.shape)
print("ST metadata shape:", st_meta.shape)

# =========================
# 11. ST preprocessing
# =========================

# library-size normalization
st_norm = st_counts.div(st_counts.sum(axis=0), axis=1) * 1e4
st_log = np.log1p(st_norm)


# =========================
# FIX: Ensembl ID -> gene symbol
# =========================

# remove version numbers
st_log.index = st_log.index.astype(str).str.replace(r"\.\d+$", "", regex=True)

mg = mygene.MyGeneInfo()

query_ids = st_log.index.tolist()

mapping = mg.querymany(
    query_ids,
    scopes="ensembl.gene",
    fields="symbol",
    species="human",
    as_dataframe=True
)

# build mapping dictionary
id_to_symbol = {}

for idx, row in mapping.iterrows():
    if "symbol" in row and pd.notnull(row["symbol"]):
        id_to_symbol[idx] = row["symbol"]

# replace Ensembl IDs with symbols
new_index = []
for g in st_log.index:
    if g in id_to_symbol:
        new_index.append(id_to_symbol[g].upper())
    else:
        new_index.append(g)

st_log.index = new_index

print("\nExample mapped ST gene names:")
print(list(st_log.index[:20]))


# =========================
# 12. Build signatures from DEG
# =========================

# Make gene names consistent
st_log.index = st_log.index.astype(str).str.upper()
deg["gene"] = deg["gene"].astype(str).str.upper()

print("\nExample ST gene names:")
print(list(st_log.index[:20]))

print("\nExample DEG gene names:")
print(deg["gene"].head(20).tolist())

overlap = set(deg["gene"]).intersection(set(st_log.index))
print("\nNumber of overlapping genes between DEG and ST:", len(overlap))
print("Example overlapping genes:", list(overlap)[:20])

cnc_signature_raw = deg.sort_values("log2FC", ascending=False).head(100)["gene"].tolist()
schwann_signature_raw = deg.sort_values("log2FC", ascending=True).head(100)["gene"].tolist()

cnc_signature = [g for g in cnc_signature_raw if g in st_log.index]
schwann_signature = [g for g in schwann_signature_raw if g in st_log.index]

print("\nCNC signature genes:", len(cnc_signature))
print("Schwann signature genes:", len(schwann_signature))
print("CNC signature:", cnc_signature[:20])
print("Schwann signature:", schwann_signature[:20])

# =========================
# 13. Score ST spots
# =========================

st_scores = pd.DataFrame(index=st_log.columns)

st_scores["cNCC_like_score"] = st_log.loc[cnc_signature].mean(axis=0)
st_scores["Schwann_like_score"] = st_log.loc[schwann_signature].mean(axis=0)
st_scores["score_diff"] = (
    st_scores["cNCC_like_score"] - st_scores["Schwann_like_score"]
)

# merge with spatial metadata
st_meta = st_meta.loc[st_scores.index]
st_meta = st_meta.join(st_scores)

# =========================
# 14. Spatial plots
# =========================

plt.figure(figsize=(6,5))
plt.scatter(
    st_meta["new_x"],
    st_meta["new_y"],
    c=st_meta["cNCC_like_score"],
    cmap="Reds",
    s=10
)
plt.colorbar(label="cNCC-like score")
plt.title("Spatial map of cNCC-like cells")
plt.tight_layout()
plt.savefig("ST_cNCC_map.png", dpi=300)


plt.figure(figsize=(6,5))
plt.scatter(
    st_meta["new_x"],
    st_meta["new_y"],
    c=st_meta["Schwann_like_score"],
    cmap="Blues",
    s=10
)
plt.colorbar(label="Schwann-like score")
plt.title("Spatial map of Schwann-like cells")
plt.tight_layout()
plt.savefig("ST_Schwann_map.png", dpi=300)

# =========================
# 15. Save results
# =========================

st_meta.reset_index().rename(columns={"index": "spot_id"}).to_csv(
    "ST_cNCC_Schwann_scores.csv",
    index=False
)

print("\nSaved ST outputs:")
print("- ST_cNCC_map.png")
print("- ST_Schwann_map.png")
print("- ST_cNCC_Schwann_scores.csv")

# =========================
# PCA + UMAP on ST spots
# =========================

print("\nRunning PCA/UMAP on ST spots...")

# normalize ST counts
st_norm = st_counts.div(st_counts.sum(axis=0), axis=1) * 1e4
st_log = np.log1p(st_norm)

# transpose: spots x genes
st_matrix = st_log.T

# use variable genes by variance
gene_var = st_matrix.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(2000).index

st_matrix_subset = st_matrix[top_genes]

# scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(st_matrix_subset)

# PCA
pca = PCA(n_components=20)
pca_result = pca.fit_transform(scaled_data)

print("PCA completed.")
print("Explained variance ratio (first 5 PCs):")
print(pca.explained_variance_ratio_[:5])

# plot PCA
plt.figure(figsize=(6,5))
plt.scatter(
    pca_result[:,0],
    pca_result[:,1],
    s=8,
    alpha=0.7
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of ST spots")
plt.tight_layout()
plt.savefig("ST_PCA.png", dpi=300)
plt.close()

# UMAP
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.3,
    random_state=42
)

umap_result = reducer.fit_transform(pca_result)

# save embedding
umap_df = pd.DataFrame({
    "UMAP1": umap_result[:,0],
    "UMAP2": umap_result[:,1],
    "cNCC_like_score": st_meta["cNCC_like_score"].values,
    "Schwann_like_score": st_meta["Schwann_like_score"].values
})

umap_df.to_csv("ST_UMAP_embedding.csv", index=False)

# UMAP colored by cNCC
plt.figure(figsize=(6,5))
plt.scatter(
    umap_result[:,0],
    umap_result[:,1],
    c=st_meta["cNCC_like_score"],
    cmap="Reds",
    s=8
)
plt.colorbar(label="cNCC-like score")
plt.title("UMAP of ST spots (cNCC-like)")
plt.tight_layout()
plt.savefig("ST_UMAP_cNCC.png", dpi=300)
plt.close()

# UMAP colored by Schwann
plt.figure(figsize=(6,5))
plt.scatter(
    umap_result[:,0],
    umap_result[:,1],
    c=st_meta["Schwann_like_score"],
    cmap="Blues",
    s=8
)
plt.colorbar(label="Schwann-like score")
plt.title("UMAP of ST spots (Schwann-like)")
plt.tight_layout()
plt.savefig("ST_UMAP_Schwann.png", dpi=300)
plt.close()

print("Saved:")
print("- ST_PCA.png")
print("- ST_UMAP_cNCC.png")
print("- ST_UMAP_Schwann.png")


# =========================
# Stage-wise abundance plots
# =========================

# =========================
# Stage-wise abundance plots
# =========================

print("\nGenerating stage-wise abundance plots...")

# group by developmental stage
stage_summary = st_meta.groupby("weeks")[
    ["cNCC_like_score", "Schwann_like_score"]
].mean().reset_index()

print(stage_summary)

# save table
stage_summary.to_csv(
    "ST_stagewise_abundance.csv",
    index=False
)

# sort stages
stage_summary = stage_summary.sort_values("weeks")

# cNCC plot
plt.figure(figsize=(6,5))

plt.plot(
    stage_summary["weeks"],
    stage_summary["cNCC_like_score"],
    marker="o",
    linewidth=2
)

plt.xlabel("Developmental week")
plt.ylabel("Mean cNCC-like score")
plt.title("Stage-wise cNCC abundance")
plt.tight_layout()

plt.savefig(
    "ST_stagewise_cNCC.png",
    dpi=300
)

plt.close()

# Schwann plot
plt.figure(figsize=(6,5))

plt.plot(
    stage_summary["weeks"],
    stage_summary["Schwann_like_score"],
    marker="o",
    linewidth=2
)

plt.xlabel("Developmental week")
plt.ylabel("Mean Schwann-like score")
plt.title("Stage-wise Schwann abundance")
plt.tight_layout()

plt.savefig(
    "ST_stagewise_Schwann.png",
    dpi=300
)

plt.close()

print("Saved:")
print("- ST_stagewise_abundance.csv")
print("- ST_stagewise_cNCC.png")
print("- ST_stagewise_Schwann.png")