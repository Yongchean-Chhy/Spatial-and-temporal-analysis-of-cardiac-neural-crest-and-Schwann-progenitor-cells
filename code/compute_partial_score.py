"""
compute_spatial_scores.py  (fixed)
─────────────────────────────────────────────────────────────────────────────
Root cause of NaN scores:
  The ST matrix uses Ensembl IDs (ENSG...) as row labels.
  DEG genes come from scRNA-seq, which already uses gene SYMBOLS.
  After stripping version numbers the ST index is still ENSG...,
  so zero signature genes match → mean([]) = NaN.

Fix:
  Map ST Ensembl IDs → symbols with mygene BEFORE scoring,
  with a clear diagnostic so you can see exactly what matched.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURE PATHS AND PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

base_dir = os.path.dirname(os.path.abspath(__file__))

SCRNA_COUNT_PATH = os.path.join(
    "..", "dataset", "dataset_st", "Filtered",
    "Developmental_heart_filtered_scRNA-seq_and_meta_data",
    "all_cells_count_matrix_filtered.tsv"
)
SCRNA_META_PATH = os.path.join(
    "..", "dataset", "dataset_st", "Filtered",
    "Developmental_heart_filtered_scRNA-seq_and_meta_data",
    "all_cells_meta_data_filtered.tsv"
)
ST_COUNT_PATH = os.path.join(
    "..", "dataset", "dataset_st", "Filtered",
    "filtered_ST_matrix_and_meta_data",
    "filtered_matrix.tsv"
)
ST_META_PATH = os.path.join(
    "..", "dataset", "dataset_st", "Filtered",
    "filtered_ST_matrix_and_meta_data",
    "meta_data.tsv"
)

ISS_PATH         = None        
ISS_X_COL        = "x"
ISS_Y_COL        = "y"
ISS_CELLTYPE_COL = "cell_type"   
ST_SPOT_RADIUS   = 55          

CNC_MARKERS      = ["ISL1", "STMN2"]
SCHWANN_MARKERS  = ["ALDH1A1"]
N_SIG_GENES      = 100
TARGET_LABEL     = "Cardiac neural crest cells & Schwann progenitor cells"

OUT_CSV          = os.path.join(base_dir, "ST_spatial_scores_aligned.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load & align scRNA-seq data
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Loading scRNA-seq data")
print("=" * 60)

counts = pd.read_csv(SCRNA_COUNT_PATH, sep="\t", index_col=0)
meta   = pd.read_csv(SCRNA_META_PATH,  sep="\t", index_col=0)
counts.index.name = "gene"
meta.index.name   = "cell_id"

common_cells = counts.columns.intersection(meta.index)
counts = counts[common_cells]
meta   = meta.loc[common_cells]
print(f"  Aligned: {counts.shape[0]:,} genes x {counts.shape[1]:,} cells")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Marker-based scoring on target cluster
# ──────────────────────────────────────────────────────────────────────────────

print("\nSTEP 2 — Marker-based cell scoring")

target_cells = meta.index[meta["celltype"] == TARGET_LABEL].tolist()
print(f"  Cells in target cluster: {len(target_cells)}")

sub_expr = counts[target_cells].T        

available_cnc     = [g for g in CNC_MARKERS     if g in sub_expr.columns]
available_schwann = [g for g in SCHWANN_MARKERS if g in sub_expr.columns]
print(f"  CNC markers found    : {available_cnc}")
print(f"  Schwann markers found: {available_schwann}")

if not available_cnc or not available_schwann:
    sys.exit("ERROR: No marker genes found in scRNA-seq matrix. Check gene names.")

sub_expr["cnc_score"]     = sub_expr[available_cnc].mean(axis=1)
sub_expr["schwann_score"] = sub_expr[available_schwann].mean(axis=1)
sub_expr["predicted_subtype"] = np.where(
    sub_expr["cnc_score"] > sub_expr["schwann_score"],
    "Cardiac neural crest-like",
    "Schwann progenitor-like"
)
print("\n  Predicted subtype counts:")
print(sub_expr["predicted_subtype"].value_counts().to_string())

# ──────────────────────────────────────────────────────────────────────────────
# 3.  DEG -> signature gene lists
# ──────────────────────────────────────────────────────────────────────────────

print("\nSTEP 3 — DEG analysis")

gene_expr_only = sub_expr.drop(
    columns=["cnc_score", "schwann_score", "predicted_subtype"]
)
g1 = sub_expr.index[sub_expr["predicted_subtype"] == "Cardiac neural crest-like"]
g2 = sub_expr.index[sub_expr["predicted_subtype"] == "Schwann progenitor-like"]

deg_rows = []
for gene in gene_expr_only.columns:
    x1, x2 = gene_expr_only.loc[g1, gene], gene_expr_only.loc[g2, gene]
    _, pval = ttest_ind(x1, x2, equal_var=False)
    m1, m2  = x1.mean(), x2.mean()
    lfc     = np.log2(m1 + 1e-6) - np.log2(m2 + 1e-6)
    deg_rows.append([gene, m1, m2, lfc, pval])

deg = pd.DataFrame(deg_rows,
                   columns=["gene", "mean_cnc", "mean_schwann", "log2FC", "pvalue"])
deg = deg.sort_values("pvalue")
deg["gene"] = deg["gene"].str.upper()
print(f"  DEG table: {len(deg):,} genes")
print(f"  Top 5 CNC-enriched : {deg.sort_values('log2FC', ascending=False)['gene'].head(5).tolist()}")
print(f"  Top 5 Sch-enriched : {deg.sort_values('log2FC', ascending=True)['gene'].head(5).tolist()}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Load ST matrix
# ──────────────────────────────────────────────────────────────────────────────

print("\nSTEP 4 — Loading ST data")

st_counts = pd.read_csv(ST_COUNT_PATH, sep="\t", index_col=0)
st_meta   = pd.read_csv(ST_META_PATH,  sep="\t", index_col=0)
print(f"  ST raw shape: {st_counts.shape[0]:,} genes x {st_counts.shape[1]:,} spots")

st_counts.index = st_counts.index.astype(str).str.replace(
    r"\.\d+$", "", regex=True
)

sample_ids = st_counts.index[:6].tolist()
looks_like_ensembl = any(str(g).startswith("ENSG") for g in sample_ids)
print(f"\n  Example ST gene IDs : {sample_ids}")
print(f"  Look like Ensembl   : {looks_like_ensembl}")

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Ensembl ID -> symbol mapping  (only if needed)
# ──────────────────────────────────────────────────────────────────────────────

if looks_like_ensembl:
    print("\nSTEP 5 — Mapping Ensembl IDs to gene symbols via mygene")
    try:
        import mygene
        mg = mygene.MyGeneInfo()

        query_ids = st_counts.index.tolist()
        print(f"  Querying {len(query_ids):,} IDs ...")

        mapping = mg.querymany(
            query_ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            as_dataframe=True,
            verbose=False
        )

        id_to_sym = {}
        for eid, row in mapping.iterrows():
            if "symbol" in row and pd.notnull(row["symbol"]):
                if eid not in id_to_sym:
                    id_to_sym[eid] = str(row["symbol"]).upper()

        mapped_count   = sum(1 for g in st_counts.index if g in id_to_sym)
        unmapped_count = len(st_counts.index) - mapped_count
        print(f"  Mapped   : {mapped_count:,}")
        print(f"  Unmapped : {unmapped_count:,}  (kept as original Ensembl ID)")
        print(f"  Example mapped symbols: "
              f"{[id_to_sym[g] for g in list(id_to_sym.keys())[:8]]}")

        st_counts.index = [id_to_sym.get(g, g) for g in st_counts.index]

    except ImportError:
        sys.exit(
            "ERROR: mygene not installed.\n"
            "Run:  pip install mygene\n"
            "Then re-run this script."
        )
else:
    print("\nSTEP 5 — ST genes already appear to be symbols, skipping Ensembl mapping.")

if st_counts.index.duplicated().any():
    n_dup = st_counts.index.duplicated().sum()
    print(f"  Deduplicating {n_dup} duplicated gene symbols (summing counts) ...")
    st_counts = st_counts.groupby(level=0).sum()

print(f"  ST after mapping: {st_counts.shape[0]:,} genes x {st_counts.shape[1]:,} spots")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Normalise ST (lib-size -> log1p)
# ──────────────────────────────────────────────────────────────────────────────

st_norm = st_counts.div(st_counts.sum(axis=0), axis=1) * 1e4
st_log  = np.log1p(st_norm)
st_log.index = st_log.index.str.upper()

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Overlap diagnostic + build signatures
# ──────────────────────────────────────────────────────────────────────────────

print("\nSTEP 6 — Checking gene overlap between DEG and ST")

deg_genes = set(deg["gene"])
st_genes  = set(st_log.index)
overlap   = deg_genes & st_genes

print(f"  DEG genes total         : {len(deg_genes):,}")
print(f"  ST genes total (symbols): {len(st_genes):,}")
print(f"  Overlap                 : {len(overlap):,}")

if len(overlap) == 0:
    print("\n  !! ZERO OVERLAP — gene name spaces still do not match !!")
    print("  Sample DEG genes :", list(deg_genes)[:10])
    print("  Sample ST  genes :", list(st_genes)[:10])
    sys.exit(
        "\nERROR: No overlapping genes between DEG results and ST matrix.\n"
        "Possible causes:\n"
        "  1. mygene mapping failed silently — check network access.\n"
        "  2. scRNA-seq uses a different gene name convention.\n"
        "  3. Species mismatch (script assumes human).\n"
        "Print the lists above and compare manually."
    )

cnc_sig     = [g for g in
               deg.sort_values("log2FC", ascending=False)["gene"].head(N_SIG_GENES)
               if g in st_log.index]
schwann_sig = [g for g in
               deg.sort_values("log2FC", ascending=True)["gene"].head(N_SIG_GENES)
               if g in st_log.index]

print(f"\n  CNC signature genes in ST    : {len(cnc_sig)} / {N_SIG_GENES}")
print(f"  Schwann sig genes in ST      : {len(schwann_sig)} / {N_SIG_GENES}")
print(f"  CNC top 10    : {cnc_sig[:10]}")
print(f"  Schwann top 10: {schwann_sig[:10]}")

if len(cnc_sig) == 0 or len(schwann_sig) == 0:
    sys.exit(
        "\nERROR: Signature lists are empty after ST overlap filter.\n"
        "Try increasing N_SIG_GENES (currently {N_SIG_GENES})."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 8.  Score ST spots
# ──────────────────────────────────────────────────────────────────────────────

print("\nSTEP 7 — Scoring ST spots")

st_scores = pd.DataFrame(index=st_log.columns)
st_scores["cnc_score"]     = st_log.loc[cnc_sig].mean(axis=0).values
st_scores["schwann_score"] = st_log.loc[schwann_sig].mean(axis=0).values
st_scores["score_diff"]    = st_scores["cnc_score"] - st_scores["schwann_score"]
st_scores["dominant_type"] = np.where(
    st_scores["cnc_score"] >= st_scores["schwann_score"],
    "cNCC-like", "Schwann-like"
)

nan_cnc = st_scores["cnc_score"].isna().sum()
nan_sch = st_scores["schwann_score"].isna().sum()
print(f"  NaN in cnc_score    : {nan_cnc}")
print(f"  NaN in schwann_score: {nan_sch}")

if nan_cnc == len(st_scores):
    sys.exit("ERROR: All cnc_score values are NaN — signature genes not found in ST matrix.")

print(f"\n  cnc_score     mean={st_scores['cnc_score'].mean():.4f} "
      f"min={st_scores['cnc_score'].min():.4f} "
      f"max={st_scores['cnc_score'].max():.4f}")
print(f"  schwann_score mean={st_scores['schwann_score'].mean():.4f} "
      f"min={st_scores['schwann_score'].min():.4f} "
      f"max={st_scores['schwann_score'].max():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 9.  Merge with ST spatial metadata
# ──────────────────────────────────────────────────────────────────────────────

print("\nSTEP 8 — Merging with ST metadata")

st_meta = st_meta.loc[st_scores.index]
result  = st_meta.join(st_scores)

# Guarantee canonical x / y column names
for old, new in [("new_x", "x"), ("new_y", "y")]:
    if old in result.columns and new not in result.columns:
        result = result.rename(columns={old: new})

print(f"  Output rows   : {len(result):,}")
print(f"  Output columns: {list(result.columns)}")

# ──────────────────────────────────────────────────────────────────────────────
# 10.  Optional ISS aggregation
# ──────────────────────────────────────────────────────────────────────────────

if ISS_PATH and os.path.exists(ISS_PATH):
    print(f"\nSTEP 9 — ISS aggregation  (radius = {ST_SPOT_RADIUS} units)")
    from scipy.spatial import cKDTree

    iss     = pd.read_csv(ISS_PATH)
    iss_xy  = iss[[ISS_X_COL, ISS_Y_COL]].values
    spot_xy = result[["x", "y"]].values
    tree    = cKDTree(iss_xy)

    total_c = np.zeros(len(result), dtype=int)
    cnc_c   = np.zeros(len(result), dtype=int)
    sch_c   = np.zeros(len(result), dtype=int)

    for i, (sx, sy) in enumerate(spot_xy):
        idxs = tree.query_ball_point([sx, sy], r=ST_SPOT_RADIUS)
        total_c[i] = len(idxs)
        if ISS_CELLTYPE_COL and ISS_CELLTYPE_COL in iss.columns:
            types   = iss.iloc[idxs][ISS_CELLTYPE_COL]
            cnc_c[i] = types.str.contains(
                "crest|CNC|cnc", case=False, na=False).sum()
            sch_c[i] = types.str.contains(
                "schwann|Schwann", case=False, na=False).sum()

    result["iss_total_cells"]   = total_c
    result["iss_cnc_cells"]     = cnc_c
    result["iss_schwann_cells"] = sch_c
    result["iss_cnc_frac"]      = np.where(
        total_c > 0, cnc_c / total_c, np.nan)
    result["iss_schwann_frac"]  = np.where(
        total_c > 0, sch_c / total_c, np.nan)

    print(f"  Spots with >=1 ISS cell : {(result['iss_total_cells'] > 0).sum():,}")

elif ISS_PATH:
    print(f"\nSTEP 9 — ISS file '{ISS_PATH}' not found, skipping.")
else:
    print("\nSTEP 9 — ISS_PATH not configured, skipping ISS aggregation.")

# ──────────────────────────────────────────────────────────────────────────────
# 11.  Reorder columns and save
# ──────────────────────────────────────────────────────────────────────────────

print("\nSTEP 10 — Saving output")

priority = ["x", "y", "cnc_score", "schwann_score", "score_diff",
            "dominant_type", "weeks"]
iss_cols = ["iss_total_cells", "iss_cnc_cells", "iss_schwann_cells",
            "iss_cnc_frac", "iss_schwann_frac"]

front  = [c for c in priority if c in result.columns]
back   = [c for c in iss_cols  if c in result.columns]
middle = [c for c in result.columns if c not in front + back]

result = result[front + middle + back]
result.index.name = "spot_id"
result.reset_index().to_csv(OUT_CSV, index=False)

print(f"\n  Saved -> {OUT_CSV}")
print(f"  Rows   : {len(result):,}")
print(f"  Columns: {list(result.columns)}")

print("\n--- Score summary ---")
print(result[["cnc_score", "schwann_score", "score_diff"]].describe().round(4))
print("\nDone.")
