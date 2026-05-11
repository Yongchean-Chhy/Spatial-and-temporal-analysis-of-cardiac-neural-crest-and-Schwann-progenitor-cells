import scanpy as sc
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    from mygene import MyGeneInfo
    MYGENE_AVAILABLE = True
except ImportError:
    MYGENE_AVAILABLE = False
    print("Warning: mygene not installed. Run: pip install mygene")

# --- WEEK -> PCW LABEL MAPPING ---
# 5w ~ 4.5 PCW, 6-7w ~ 6.5 PCW, 9w ~ 9.5 PCW
WEEK_TO_PCW = {
    5: "4.5PCW",
    6: "6.5PCW",
    7: "6.5PCW",
    9: "9.5PCW",
}
PCW_ORDER = ["4.5PCW", "6.5PCW", "9.5PCW"]

# --- 1. SETUP & UTILS ---
def setup_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# --- 2. DATA LOADERS ---
def load_scrna(tsv_path):
    """
    Loads scRNA-seq metadata TSV (rows = cells, columns = metadata only).
    Since this file contains no count matrix, .X is a zero placeholder.
    Cell type labels are read from the 'celltype' column.

    Expected columns: nGene, nUMI, experiment, Phase, res.0.7, celltype, state
    """
    print(f"  Loading scRNA-seq from: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)

    print(f"  Columns found: {df.columns.tolist()}")
    print(f"  Cell types found: {df['celltype'].unique().tolist() if 'celltype' in df.columns else 'NO celltype COLUMN'}")

    adata = ad.AnnData(
        X   = np.zeros((len(df), 1)), 
        obs = df.copy(),
        var = pd.DataFrame(index=["placeholder"])
    )
    print(f"  scRNA AnnData: {adata.n_obs} cells (metadata only)")
    return adata


def load_iss(csv_path, pcw_label):
    """
    Loads ISS data from CSV and tags all cells with the given PCW label.
    Spatial coordinates (x, y) are stored in .obsm['spatial'].

    Expected columns: parent_id, cell_type, cNCC_score, SPC_score, cell_id, x, y
    """
    print(f"  Loading ISS ({pcw_label}) from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "cell_id" in df.columns:
        df.index = df["cell_id"].astype(str)

    df["PCW"] = pcw_label  

    adata = ad.AnnData(
        X   = np.zeros((len(df), 1)),
        obs = df.copy(),
        var = pd.DataFrame(index=["placeholder"])
    )
    adata.obsm["spatial"] = df[["x", "y"]].values.astype(float)

    print(f"  ISS AnnData: {adata.n_obs} cells")
    return adata


def ensembl_to_symbols(ensembl_ids):
    """
    Converts versioned ENSEMBL IDs (e.g. ENSG00000000003.14)
    to gene symbols using MyGeneInfo.
    Returns a dict: {ensembl_base -> symbol}, unmapped IDs keep their base ID.
    """
  
    base_ids = [e.split(".")[0] for e in ensembl_ids]

    if not MYGENE_AVAILABLE:
        print("  Skipping ENSEMBL->symbol conversion (mygene not installed).")
        return {b: b for b in base_ids}

    print(f"  Querying MyGene.info for {len(base_ids)} ENSEMBL IDs...")
    mg = MyGeneInfo()
    results = mg.querymany(base_ids, scopes="ensembl.gene", fields="symbol", species="human", verbose=False)

    mapping = {}
    for r in results:
        base = r["query"]
        mapping[base] = r.get("symbol", base) 

    unmapped = sum(1 for v in mapping.values() if v == v and "ENSG" in str(v))
    if unmapped:
        print(f"  Warning: {unmapped} IDs could not be mapped to gene symbols.")

    return mapping


def load_st(metadata_path, matrix_path):
    """
    Loads ST data from a metadata TSV and a counts matrix TSV.

    Matrix layout: rows = genes (versioned ENSEMBL), columns = spots.
    Matrix is transposed so AnnData is spots x genes.
    ENSEMBL IDs are mapped to gene symbols for downstream scoring.

    Metadata layout: rows = spots, matched to matrix columns on index.
    """
    print(f"  Loading ST matrix from: {matrix_path}")
 
    matrix = pd.read_csv(matrix_path, sep="\t", index_col=0).T

    print(f"  Loading ST metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    common = matrix.index.intersection(metadata.index)
    if len(common) < len(matrix):
        print(f"  Warning: {len(matrix) - len(common)} spots dropped (not in metadata).")
    matrix   = matrix.loc[common]
    metadata = metadata.loc[common]

    symbol_map  = ensembl_to_symbols(matrix.columns.tolist())
    base_ids    = [e.split(".")[0] for e in matrix.columns]
    gene_symbols = [symbol_map.get(b, b) for b in base_ids]

    adata = ad.AnnData(
        X   = matrix.values.astype(float),
        obs = metadata.copy(),
        var = pd.DataFrame(index=gene_symbols)
    )
    adata.var_names_make_unique()

    if "new_x" in metadata.columns and "new_y" in metadata.columns:
        adata.obsm["spatial"] = metadata[["new_x", "new_y"]].values.astype(float)

    print(f"  ST AnnData: {adata.n_obs} spots x {adata.n_vars} genes")
    return adata


# --- 3. ST TEMPORAL ANALYSIS ---
def calculate_gene_scores(adata, score_name, gene_list):
    valid_genes = [g for g in gene_list if g in adata.var_names]
    if not valid_genes:
        print(f"  Warning: no valid genes found for '{score_name}', skipping.")
        return adata
    sc.tl.score_genes(adata, gene_list=valid_genes, score_name=score_name)
    return adata


def run_st_temporal_pipeline(st_metadata_path, st_matrix_path, save_dir):
    """
    Loads ST data, computes cNCC/Schwann module scores,
    maps weeks -> PCW label, saves temporal trend plot + CSV.
    """
    print("\n[ST] Processing temporal trends...")
    setup_output_dir(save_dir)

    adata_st = load_st(st_metadata_path, st_matrix_path)

    cnc_genes     = ["SOX10", "TFAP2A", "PHOX2B"]
    schwann_genes = ["MPZ", "PLP1", "MBP"]
    if "cnc_score" not in adata_st.obs.columns:
        adata_st = calculate_gene_scores(adata_st, "cnc_score", cnc_genes)
    if "schwann_score" not in adata_st.obs.columns:
        adata_st = calculate_gene_scores(adata_st, "schwann_score", schwann_genes)

    adata_st.obs["PCW"] = adata_st.obs["weeks"].map(WEEK_TO_PCW)
    unmapped = adata_st.obs["PCW"].isna().sum()
    if unmapped:
        print(f"  Warning: {unmapped} spots have unmapped week values.")

    temp_df = (
        adata_st.obs
        .groupby("PCW", observed=True)[["cnc_score", "schwann_score"]]
        .mean()
        .reindex(PCW_ORDER)
    )

    # Save CSV
    csv_path = os.path.join(save_dir, "st_temporal_scores.csv")
    temp_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # Plot
    ax = temp_df.plot(marker="o", figsize=(8, 5))
    ax.set_title("ST Module Scores across Developmental Stages")
    ax.set_xlabel("PCW")
    ax.set_ylabel("Mean Module Score")
    ax.set_xticks(range(len(PCW_ORDER)))
    ax.set_xticklabels(PCW_ORDER)
    plt.savefig(os.path.join(save_dir, "st_temporal_trends.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.join(save_dir, 'st_temporal_trends.png')}")

    return adata_st


# --- 4. ISS SPATIAL ANALYSIS ---
def add_spatial_uns(adata, library_id="tissue"):
    """
    Populates adata.uns['spatial'] with the minimal structure squidpy expects.
    Must be called after obsm['spatial'] is set.
    """
    adata.uns["spatial"] = {
        library_id: {
            "images": {},
            "scalefactors": {
                "spot_diameter_fullres": 1.0,
                "tissue_hires_scalef": 1.0,
            },
        }
    }
    adata.obs["library_id"] = pd.Categorical([library_id] * adata.n_obs)


def run_iss_phase(adata_iss, phase_label, save_dir):
    """Runs spatial scatter + neighbourhood enrichment for one PCW phase."""
    p_dir = setup_output_dir(os.path.join(save_dir, f"iss_{phase_label}"))

    # --- Spatial scatter ---
    obs = adata_iss.obs.copy()
    obs["x"] = adata_iss.obsm["spatial"][:, 0]
    obs["y"] = adata_iss.obsm["spatial"][:, 1]

    cell_types = obs["cell_type"].unique()
    fig, ax = plt.subplots(figsize=(8, 8))
    for ct in cell_types:
        sub = obs[obs["cell_type"] == ct]
        ax.scatter(sub["x"], sub["y"], label=ct, s=1.5, alpha=0.7)
    ax.set_title(f"Spatial Distribution [{phase_label}]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(markerscale=5, bbox_to_anchor=(1.05, 1), loc="upper left")
    path = os.path.join(p_dir, "spatial_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Neighbourhood enrichment ---
    add_spatial_uns(adata_iss, library_id=phase_label)

    adata_iss.obs["library_id"] = adata_iss.obs["library_id"].astype("category")
    adata_iss.obs["cell_type"] = adata_iss.obs["cell_type"].astype("category")

    sq.gr.spatial_neighbors(adata_iss, library_key="library_id")
    sq.gr.nhood_enrichment(adata_iss, cluster_key="cell_type")

    sq.pl.nhood_enrichment(
        adata_iss,
        cluster_key="cell_type",
        show=False
    )

    path = os.path.join(p_dir, "neighborhood_enrichment.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def run_multi_phase_iss(iss_files, save_dir):
    """
    Loads each ISS CSV independently (one per PCW),
    tags cells with the PCW label, then runs spatial analysis per phase.

    iss_files: dict mapping PCW label -> file path, e.g.
               {"4.5PCW": "path/to/4.5.csv", "6.5PCW": ..., "9.5PCW": ...}
    """
    print("\n[ISS] Processing spatial phases...")
    setup_output_dir(save_dir)

    for phase in PCW_ORDER:
        if phase not in iss_files:
            print(f"  Skipping {phase}: no file provided.")
            continue

        adata_iss = load_iss(iss_files[phase], pcw_label=phase)
        print(f"  Analyzing ISS phase: {phase} ({adata_iss.n_obs} cells)")
        run_iss_phase(adata_iss, phase, save_dir)


# --- 5. scRNA-seq TRAJECTORY ANALYSIS ---
def run_scrna_trajectory(sc_path, save_dir):
    """
    Loads scRNA-seq metadata TSV, subsets to cNCC + SPC cells,
    and plots a UMAP coloured by celltype.
    Note: since no count matrix is available, PCA/PAGA are skipped
    and UMAP is run on a random init (or skipped if too few cells).
    """
    print("\n[scRNA] Running trajectory analysis...")
    setup_output_dir(save_dir)

    adata_sc = load_scrna(sc_path)

    if "celltype" not in adata_sc.obs.columns:
        print("  Error: 'celltype' column not found. Skipping trajectory.")
        return

    target_types = ["cNCC", "SPC"]
    found = adata_sc.obs["celltype"].unique().tolist()
    subset = adata_sc[adata_sc.obs["celltype"].isin(target_types)].copy()

    if subset.n_obs == 0:
        print(f"  Warning: no cells found for {target_types}.")
        print(f"  Cell types in file: {found}")
        print("  Skipping trajectory — check that cell type labels match exactly.")
        return

    print(f"  Subset size: {subset.n_obs} cells")

    ct_counts = subset.obs["celltype"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ct_counts.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
    ax.set_title("cNCC vs SPC Cell Counts (scRNA-seq)")
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    path = os.path.join(save_dir, "celltype_counts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Save cell metadata for cNCC + SPC
    csv_path = os.path.join(save_dir, "cnc_spc_metadata.csv")
    subset.obs.to_csv(csv_path)
    print(f"  Saved: {csv_path}")


# --- 6. MAIN PIPELINE ---
def run_integrated_pipeline(
    sc_path,
    iss_files,
    st_metadata_path,
    st_matrix_path,
    out_dir="analysis_output"
):
    """
    sc_path          : path to scRNA-seq TSV
    iss_files        : dict mapping PCW label -> ISS CSV path, e.g.
                       {"4.5PCW": "iss_4.5.csv", "6.5PCW": ..., "9.5PCW": ...}
    st_metadata_path : path to ST metadata TSV
    st_matrix_path   : path to ST counts matrix TSV
    """
    root    = setup_output_dir(out_dir)
    sc_dir  = setup_output_dir(os.path.join(root, "scrna_trajectory"))
    iss_dir = setup_output_dir(os.path.join(root, "iss_spatial"))
    st_dir  = setup_output_dir(os.path.join(root, "st_temporal"))

    run_scrna_trajectory(sc_path, sc_dir)
    run_multi_phase_iss(iss_files, iss_dir)
    run_st_temporal_pipeline(st_metadata_path, st_matrix_path, st_dir)

    print(f"\nPipeline complete. Results saved to: {os.path.abspath(root)}")


# --- EXECUTION ---
if __name__ == "__main__":
    run_integrated_pipeline(
        sc_path = "../dataset/dataset_st/Filtered/Developmental_heart_filtered_scRNA-seq_and_meta_data/all_cells_meta_data_filtered.tsv",
        iss_files = {
            "4.5PCW": "../dataset/dataset_iss/PCW4.5_merge_data.csv",
            "6.5PCW": "../dataset/dataset_iss/PCW6.5_merge_data.csv",
            "9.5PCW": "../dataset/dataset_iss/PCW9.5_merge_data.csv",
        },
        st_metadata_path = "../dataset/dataset_st/Filtered/filtered_ST_matrix_and_meta_data/meta_data.tsv",
        st_matrix_path   = "../dataset/dataset_st/Filtered/filtered_ST_matrix_and_meta_data/filtered_matrix.tsv",
    )
