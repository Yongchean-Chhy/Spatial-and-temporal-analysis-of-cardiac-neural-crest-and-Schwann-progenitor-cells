import pandas as pd
import numpy as np
import tifffile as tif
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import scanpy as sc
import squidpy as sq
from scipy.stats import gaussian_kde
from scipy.spatial import distance_matrix
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no windows, much faster in batch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os

NNC_MARKER = ["ISL1", "STMN2"]   # Asp 2019, Figure 6A
SPC_MARKER = ["ALDH1A1"]         # Asp 2019, Figure 6A

# Styling Constants
BG_COLOR = "#0F1117"
AXIS_BG = "#1A1D27"
PALETTE = {"cNCC": "#E63946", "SPC": "#F4A261", "other": "#4A4E69"}

# ── Development stages ────────────────────────────────────────────────────────
# Each entry: (iss_csv, assigned_spots_csv, dapi_tif, dataset_tag)
DATASETS = [
    (
        'dataset/dataset_iss/spots_data/spots_PCW4.5_1.csv',
        'dataset/dataset_iss/spots_data/spots_w_cell_segmentation_PCW4.5_1.csv',
        'dataset_iss/dapi_data/nuclei_PCW4.5_1.tif',
        'PCW4.5'
    ),
    (
        'dataset/dataset_iss/spots_data/spots_PCW6.5_1.csv',
        'dataset/dataset_iss/spots_data/spots_w_cell_segmentation_PCW6.5_1.csv',
        'dataset_iss/dapi_data/nuclei_PCW6.5_1.tif',
        'PCW6.5'
    ),
    (
        'dataset/dataset_iss/spots_data/spots_PCW9.5_1.csv',
        'dataset/dataset_iss/spots_data/spots_w_cell_segmentation_PCW9.5_1.csv',
        'dataset_iss/dapi_data/nuclei_PCW9.5_1.tif',
        'PCW9.5'
    ),
]

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"


def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)


def results_path(tag, suffix):
    """Return a consistent path: results/{tag}_{suffix}"""
    return os.path.join(RESULTS_DIR, f"{tag}_{suffix}")


def plots_path(tag, suffix):
    """Return a consistent path: plots/{tag}_{suffix}"""
    return os.path.join(PLOTS_DIR, f"{tag}_{suffix}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_iss_data(file_path):
    df = pd.read_csv(file_path)
    print("Total transcripts:", len(df))
    print("Unique genes:", df['gene'].nunique())
    print(df[['spotX', 'spotY']].describe())
    return df

def load_tif_image(tif_file):
    img = tif.imread(f'dataset/{tif_file}')
    if img.ndim == 3:
        img = img[0]
    
    if img.ndim != 2:
        raise ValueError("Expected a 3D image, but got a {img.shape}.")
    return img

# ── Cell segmentation ─────────────────────────────────────────────────────────

def segment_cells_from_dapi(
    dapi_input,
    spot_df: pd.DataFrame,
    expand_pixels: int = 20,
    min_nucleus_size: int = 30,
    gaussian_sigma: float = 1.0,
    peak_min_distance: int = 6,
    threshold_mode: str = "otsu",
    spot_x_col: str = "spotX",
    spot_y_col: str = "spotY",
    gene_col: str = "gene",
    output_csv_path: str = None
):
    dapi_img = load_tif_image(dapi_input)

    required_cols = {gene_col, spot_x_col, spot_y_col}
    missing = required_cols - set(spot_df.columns)
    if missing:
        raise ValueError(f"spot_df is missing columns: {missing}")

    # Smooth image
    img = gaussian(dapi_img.astype(float), sigma=gaussian_sigma, preserve_range=True)

    # Threshold nuclei
    if threshold_mode.lower() == "otsu":
        thr = threshold_otsu(img)
    elif threshold_mode.lower() == "mean":
        thr = img.mean()
    else:
        raise ValueError("threshold_mode must be 'otsu' or 'mean'.")

    nuclei_mask = img > thr
    nuclei_mask = binary_opening(nuclei_mask, disk(1))
    nuclei_mask = remove_small_objects(nuclei_mask, min_size=min_nucleus_size)

    # Watershed seeds
    distance = ndi.distance_transform_edt(nuclei_mask)
    peaks = peak_local_max(distance, min_distance=peak_min_distance, labels=nuclei_mask)

    markers = np.zeros_like(dapi_img, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        markers, _ = ndi.label(nuclei_mask)

    nuclei_labels = watershed(-distance, markers=markers, mask=nuclei_mask)

    # Expand nuclei by 20 pixels
    expanded_region = ndi.binary_dilation(nuclei_labels > 0, iterations=expand_pixels)

    # Watershed to grow cells from nuclei seeds
    seed_dist = ndi.distance_transform_edt(nuclei_labels == 0)
    cell_mask = watershed(seed_dist, markers=nuclei_labels, mask=expanded_region).astype(np.int32)

    # Assign spots to cells
    out = spot_df.copy()
    out[spot_x_col] = pd.to_numeric(out[spot_x_col], errors="coerce")
    out[spot_y_col] = pd.to_numeric(out[spot_y_col], errors="coerce")
    out = out.dropna(subset=[spot_x_col, spot_y_col, gene_col]).copy()

    x = np.rint(out[spot_x_col].to_numpy()).astype(int)
    y = np.rint(out[spot_y_col].to_numpy()).astype(int)

    h, w = cell_mask.shape
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    parent_id = np.full(len(out), -1, dtype=int)
    parent_prob = np.zeros(len(out), dtype=float)
    cell_inside_dist = ndi.distance_transform_edt(cell_mask > 0)

    for idx in np.where(valid)[0]:
        yy, xx = y[idx], x[idx]
        cid = cell_mask[yy, xx]
        parent_id[idx] = int(cid) if cid > 0 else -1
        if cid > 0:
            parent_prob[idx] = float(np.clip(cell_inside_dist[yy, xx] / expand_pixels, 0.0, 1.0))

    out["parent_id"] = parent_id
    out["parent_prob"] = parent_prob
    out = out[out["parent_id"] > 0].copy()

    if output_csv_path:
        out.to_csv(output_csv_path, index=False)
        print(f"Assigned spots saved to: {output_csv_path}")

    return cell_mask, out, nuclei_mask

# ── Expression matrix ─────────────────────────────────────────────────────────

def build_cell_expression_matrix(
    spot_df: pd.DataFrame,
    gene_col: str = "gene",
    cell_col: str = "parent_id",
    prob_col: str = "parent_prob",
    prob_threshold: float = 0.5,
    min_transcripts_per_cell: int = 20,
    max_transcripts_per_cell: int = 5000,
    min_cells_per_gene: int = 5,
    normalize: bool = False,
    log_transform: bool = False, 
    protected_genes: list = None 
    ):

    required = {gene_col, cell_col, prob_col}
    missing = required - set(spot_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = spot_df.copy()
    df = df[df[prob_col] >= prob_threshold]
    df = df[df[cell_col] > 0]

    cell_gene = (
        df.groupby([cell_col, gene_col])
        .size()
        .unstack(fill_value=0)
    )

    cell_counts = cell_gene.sum(axis=1)

    valid_cells = (
        (cell_counts >= min_transcripts_per_cell) &
        (cell_counts <= max_transcripts_per_cell)
    )

    cell_gene = cell_gene.loc[valid_cells]

    gene_detection = (cell_gene > 0).sum(axis=0)
    is_protected = cell_gene.columns.isin(protected_genes)
    valid_genes = (gene_detection >= min_cells_per_gene) | is_protected
    cell_gene = cell_gene.loc[:, valid_genes]

    missing_markers = [g for g in protected_genes if g not in cell_gene.columns]
    if missing_markers:
        print(f"WARNING: protected marker gene(s) not found in spot data: {missing_markers}")

    if normalize:
        cell_gene = cell_gene.div(cell_gene.sum(axis=1), axis=0) * 1e4

    if log_transform:
        cell_gene = np.log1p(cell_gene)

    qc_metrics = {
        "n_cells": cell_gene.shape[0],
        "n_genes": cell_gene.shape[1],
        "mean_transcripts_per_cell": float(cell_counts.mean()),
        "median_transcripts_per_cell": float(cell_counts.median())
    }

    return cell_gene, qc_metrics

# ── Cell type scoring & labelling ─────────────────────────────────────────────

def score_cell_types(cell_gene_matrix, ncc_marker, spc_marker):
    ncc = [g for g in ncc_marker if g in cell_gene_matrix.columns]
    spc = [g for g in spc_marker if g in cell_gene_matrix.columns]

    if len(ncc) == 0 or len(spc) == 0:
        raise ValueError("Marker genes not found in dataset.")
    
    ncc_score = cell_gene_matrix[ncc].mean(axis=1)
    spc_score = cell_gene_matrix[spc].mean(axis=1)

    scores = pd.DataFrame({
        "cNCC_score": ncc_score,
        "SPC_score": spc_score
    })

    return scores

def assign_labels(scores, min_score=0.1, margin=0.05):
    labels = []

    for _, row in scores.iterrows():
        c = row["cNCC_score"]
        s = row["SPC_score"]

        if c < min_score and s < min_score:
            labels.append("other")
        elif c > s + margin:
            labels.append("cNCC")
        elif s > c + margin:
            labels.append("SPC")
        else:
            labels.append("other")

    return pd.Series(labels, index=scores.index)

# ── Spatial helpers ───────────────────────────────────────────────────────────

def compute_cell_centroids(assigned_spots):
    centroids = (
        assigned_spots
        .groupby("parent_id")[["spotX", "spotY"]]
        .mean()
    )

    centroids.columns = ["x", "y"]
    centroids.index.name = "cell_id"

    return centroids

def build_anndata(cell_gene_matrix, cell_metadata, cell_coords):
    shared_ids = cell_gene_matrix.index.intersection(
                     cell_metadata.index).intersection(cell_coords.index)
 
    cell_gene_matrix = cell_gene_matrix.loc[shared_ids]
    cell_metadata    = cell_metadata.loc[shared_ids]
    cell_coords      = cell_coords.loc[shared_ids]
 
    adata = sc.AnnData(cell_gene_matrix)
    adata.obs = cell_metadata.copy()
    adata.obsm["spatial"] = cell_coords[["x", "y"]].values
 
    return adata

def plot_spatial_cells(adata):
    sq.pl.spatial_scatter(
        adata,
        color="cell_type",
        size=20
    )

def compute_neighbors(adata, n_neighbors=6):
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors)

# ── Spatial analyses with plot saving ────────────────────────────────────────

def run_neighborhood_enrichment(adata, save_path=None, n_perms=100):
    if not hasattr(adata.obs["cell_type"], "cat"):
        adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"])
    # Default n_perms is 1000 — 100 is enough for exploratory work and ~10x faster
    sq.gr.nhood_enrichment(adata, cluster_key="cell_type", n_perms=n_perms)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_COLOR)
    sq.pl.nhood_enrichment(adata, cluster_key="cell_type", ax=ax)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Neighborhood enrichment plot saved to: {save_path}")
    plt.close(fig)  # don't render to screen — saves time in batch runs

def run_cooccurrence(adata, save_path=None, max_cells=2000):
    if not hasattr(adata.obs["cell_type"], "cat"):
        adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"])

    # co_occurrence is O(n^2) in cells — subsample to keep it tractable
    adata_cooc = adata.copy()
    if adata_cooc.n_obs > max_cells:
        print(f"  Subsampling {adata_cooc.n_obs} → {max_cells} cells for co-occurrence (O(n²) step)")
        sc.pp.subsample(adata_cooc, n_obs=max_cells, copy=False, random_state=42)

    # n_splits controls distance-bin resolution; 25 (vs default 50) halves the work
    sq.gr.co_occurrence(adata_cooc, cluster_key="cell_type", n_splits=25)

    # sq.pl.co_occurrence manages its own axes — do not pass ax=
    sq.pl.co_occurrence(adata_cooc, cluster_key="cell_type")
    if save_path:
        plt.gcf().savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Co-occurrence plot saved to: {save_path}")
    plt.close("all")

    # Copy results back to main adata so the pipeline can save them
    adata.uns["cell_type_co_occurrence"] = adata_cooc.uns["cell_type_co_occurrence"]

# ── Density / distance utilities ──────────────────────────────────────────────

def compute_density(cell_coords, labels, target_type):
    mask = labels == target_type
    coords = cell_coords[mask]

    if len(coords) < 10:
        return None
    
    kde = gaussian_kde(coords.T)
    return kde

def get_enrichment_matrix(adata):
    return adata.uns["cell_type_nhood_enrichment"]["zscore"]

def compute_intertype_distance(cell_coords, labels, type1, type2):
    coords1 = cell_coords[labels == type1]
    coords2 = cell_coords[labels == type2]

    if len(coords1) == 0 or len(coords2) == 0:
        return np.nan
    
    d = distance_matrix(coords1, coords2)
    return d.mean()

def permutation_test(cell_coords, labels, type1, type2, n_perm=100):
    real_dist = compute_intertype_distance(cell_coords, labels, type1, type2)

    perm_dists = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(labels)
        d = compute_intertype_distance(cell_coords, shuffled, type1, type2)
        perm_dists.append(d)

    return real_dist, perm_dists

def compute_local_composition(adata, radius=50):
    sq.gr.spatial_neighbors(adata, radius=radius)
    sq.gr.nhood_enrichment(adata, cluster_key="cell_type")

    return adata

# ── Dark-theme plot helpers ───────────────────────────────────────────────────

def _setup_dark_ax(ax, title):
    ax.set_facecolor(AXIS_BG)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.set_xlabel("x (µm)", color="#888", fontsize=8)
    ax.set_ylabel("y (µm)", color="#888", fontsize=8)
    ax.set_aspect("equal")

def _add_background_layers(ax, coords, dapi_img=None, show_cells=True):
    """Adds DAPI image and/or a faint grey scatter of all cells."""
    if dapi_img is not None:
        ax.imshow(
            dapi_img, cmap="gray", origin="upper",
            extent=[0, dapi_img.shape[1], dapi_img.shape[0], 0],
            alpha=0.25, zorder=0
        )
    if show_cells:
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=2, alpha=0.12, color="#555", rasterized=True, zorder=1)
        
        
def plot_discrete_labels(ax, adata, title="Cell Types"):
    """Panel A style: Scatter plot of all categorical cell labels."""
    coords = adata.obsm["spatial"]
    cell_types = adata.obs["cell_type"]
    
    _setup_dark_ax(ax, title)
    _add_background_layers(ax, coords, show_cells=False)
    
    for ct, col in PALETTE.items():
        mask = cell_types == ct
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=6, alpha=0.7, color=col, label=ct, zorder=2)
    
    leg = ax.legend(fontsize=8, labelcolor="white", framealpha=0.15, loc="upper right")
    leg.get_frame().set_edgecolor("#444")

def plot_cell_type_and_transcripts(ax, adata, target_type, markers, spot_df=None, dapi_img=None):
    """Panels B & C style: Highlight one cell type and its raw mRNA spots."""
    coords = adata.obsm["spatial"]
    cell_types = adata.obs["cell_type"]
    color = PALETTE.get(target_type, "#FFF")
    
    _setup_dark_ax(ax, f"{target_type} Distribution")
    _add_background_layers(ax, coords, dapi_img=dapi_img, show_cells=True)
    
    # Plot cells
    mask = cell_types == target_type
    ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.85, color=color, zorder=3)
    
    # Plot transcript dots
    if spot_df is not None:
        dot_colors = ["#FF6B8A", "#FFB3C1", "#FFD580"]
        for i, gene in enumerate(markers):
            g_df = spot_df[spot_df["gene"] == gene]
            if not g_df.empty:
                ax.scatter(g_df["spotX"], g_df["spotY"], s=1.5, alpha=0.4, 
                           color=dot_colors[i % len(dot_colors)], zorder=2, label=gene)
    
    ax.legend(fontsize=7, labelcolor="white", framealpha=0.1, loc="upper right")

def plot_score_heatmap(ax, fig, adata, score_key, color_end, dapi_img=None):
    """Panels E & F style: Continuous intensity map of a gene/cell score."""
    coords = adata.obsm["spatial"]
    scores = adata.obs[score_key].values
    
    _setup_dark_ax(ax, f"{score_key} Heatmap")
    _add_background_layers(ax, coords, dapi_img=dapi_img, show_cells=False)
    
    cmap = LinearSegmentedColormap.from_list("custom", [AXIS_BG, color_end], N=256)
    vmax = np.percentile(scores, 99)
    
    sc_plot = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap=cmap,
                    s=5, alpha=0.85, zorder=2, vmin=0, vmax=vmax)
    
    cb = fig.colorbar(sc_plot, ax=ax, shrink=0.6, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

def plot_spatial_cells_modular(adata, save_path=None, dapi_img=None, spot_df=None):
    fig, axes = plt.subplots(2, 3, figsize=(21, 13), facecolor=BG_COLOR)
    
    # Panel A: All Types
    plot_discrete_labels(axes[0, 0], adata, "A — All Cell Types")
    
    # Panel B: cNCC + transcripts
    plot_cell_type_and_transcripts(axes[0, 1], adata, "cNCC", NNC_MARKER, spot_df, dapi_img)
    
    # Panel C: SPC + transcripts
    plot_cell_type_and_transcripts(axes[0, 2], adata, "SPC", SPC_MARKER, spot_df, dapi_img)
    
    # Panel D: Co-localization
    _setup_dark_ax(axes[1, 0], "D — cNCC & SPC Overlay")
    _add_background_layers(axes[1, 0], adata.obsm["spatial"], dapi_img)
    for ct in ["cNCC", "SPC"]:
        m = adata.obs["cell_type"] == ct
        axes[1, 0].scatter(adata.obsm["spatial"][m,0], adata.obsm["spatial"][m,1], 
                           s=10, color=PALETTE[ct], label=ct, zorder=3)
    axes[1, 0].legend(labelcolor="white")

    # Panels E & F: Continuous Heatmaps
    plot_score_heatmap(axes[1, 1], fig, adata, "cNCC_score", PALETTE["cNCC"], dapi_img)
    plot_score_heatmap(axes[1, 2], fig, adata, "SPC_score", PALETTE["SPC"], dapi_img)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
        print(f"Spatial panel plot saved to: {save_path}")
    plt.close("all")  # free memory — don't render to screen in batch runs

# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_iss_analysis_pipeline(
    iss_csv_path,
    assigned_spots,
    tag,
    ncc_markers=NNC_MARKER,
    spc_markers=SPC_MARKER
):
    print(f"\n{'='*60}")
    print(f"  Processing dataset: {tag}")
    print(f"{'='*60}")

    print("Building cell-gene matrix...")
    cell_gene_matrix, qc = build_cell_expression_matrix(
        assigned_spots,
        normalize=True,
        log_transform=True,
        protected_genes=ncc_markers + spc_markers
    )

    # Save cell-gene matrix
    cgm_path = results_path(tag, "cell_gene_matrix.csv")
    cell_gene_matrix.to_csv(cgm_path)
    print(f"Cell-gene matrix saved to: {cgm_path}")

    # Save QC metrics
    qc_path = results_path(tag, "qc_metrics.csv")
    pd.DataFrame([qc]).to_csv(qc_path, index=False)
    print(f"QC metrics saved to: {qc_path}")

    print("Scoring cell types...")
    scores = score_cell_types(cell_gene_matrix, ncc_markers, spc_markers)
    cell_labels = assign_labels(scores)

    cell_metadata = pd.DataFrame({
        "cell_type": pd.Categorical(cell_labels),
        "cNCC_score": scores["cNCC_score"],
        "SPC_score": scores["SPC_score"]
    }, index=cell_gene_matrix.index)

    # Save cell metadata (cell type labels + scores)
    meta_path = results_path(tag, "cell_metadata.csv")
    cell_metadata.to_csv(meta_path)
    print(f"Cell metadata saved to: {meta_path}")

    cell_coords = compute_cell_centroids(assigned_spots)

    # Save cell centroids
    centroids_path = results_path(tag, "cell_centroids.csv")
    cell_coords.to_csv(centroids_path)
    print(f"Cell centroids saved to: {centroids_path}")

    # Save assigned spots (if not already written by segment_cells_from_dapi)
    spots_path = results_path(tag, "assigned_spots.csv")
    assigned_spots.to_csv(spots_path, index=False)
    print(f"Assigned spots saved to: {spots_path}")

    adata = build_anndata(cell_gene_matrix, cell_metadata, cell_coords)

    compute_neighbors(adata)

    import time
    t0 = time.time()
    print("  Running neighborhood enrichment (n_perms=100)...")
    run_neighborhood_enrichment(
        adata,
        save_path=plots_path(tag, "nhood_enrichment.png")
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  Running co-occurrence (subsampled, n_splits=25)...")
    run_cooccurrence(
        adata,
        save_path=plots_path(tag, "co_occurrence.png")
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    enrichment    = adata.uns["cell_type_nhood_enrichment"]
    co_occurrence = adata.uns["cell_type_co_occurrence"]

    # Save enrichment z-score matrix
    enrichment_path = results_path(tag, "nhood_enrichment_zscore.csv")
    pd.DataFrame(
        enrichment["zscore"],
        index=adata.obs["cell_type"].cat.categories,
        columns=adata.obs["cell_type"].cat.categories
    ).to_csv(enrichment_path)
    print(f"Neighborhood enrichment z-scores saved to: {enrichment_path}")

    # Save co-occurrence scores (mean across distance bins)
    cooc_path = results_path(tag, "co_occurrence_scores.csv")
    cooc_array = co_occurrence["occ"]          # shape: (n_types, n_types, n_intervals)
    cooc_mean  = cooc_array.mean(axis=-1)      # collapse distance dimension
    pd.DataFrame(
        cooc_mean,
        index=adata.obs["cell_type"].cat.categories,
        columns=adata.obs["cell_type"].cat.categories
    ).to_csv(cooc_path)
    print(f"Co-occurrence scores (mean) saved to: {cooc_path}")

    return {
        "assigned_spots":  assigned_spots,
        "cell_gene_matrix": cell_gene_matrix,
        "cell_metadata":   cell_metadata,
        "adata":           adata,
        "enrichment":      enrichment,
        "co_occurrence":   co_occurrence
    }

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _ensure_dirs()

    all_results = {}

    for iss_csv, assigned_csv, dapi_tif, tag in DATASETS:
        assigned_spots = pd.read_csv(assigned_csv)

        results = run_iss_analysis_pipeline(
            iss_csv_path=iss_csv,
            assigned_spots=assigned_spots,
            tag=tag,
        )

        my_adata = results["adata"]
        dapi_img = load_tif_image(dapi_tif)

        plot_spatial_cells_modular(
            my_adata,
            save_path=plots_path(tag, "spatial_panel.png"),
            dapi_img=dapi_img,
            spot_df=assigned_spots,
        )

        print(f"[{tag}] Processed {my_adata.n_obs} cells.")
        print(f"[{tag}] Cell types: {my_adata.obs['cell_type'].unique().tolist()}")

        all_results[tag] = results

    return all_results


if __name__ == "__main__":
    main()


