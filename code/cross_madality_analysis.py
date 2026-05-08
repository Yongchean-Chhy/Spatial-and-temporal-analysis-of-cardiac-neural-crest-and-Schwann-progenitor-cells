import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os
import json


def merge_dataset(dataset1, dataset2, save_path):
    anno = pd.read_csv(dataset1)
    coords = pd.read_csv(dataset2)
    merged = anno.merge(coords, left_on="parent_id", right_on="cell_id")
    merged.to_csv(save_path, index=False)


def load_st_data(st_file):
    st_df = pd.read_csv(st_file)
    required_cols = ["x", "y", "cnc_score", "schwann_score"]
    for col in required_cols:
        if col not in st_df.columns:
            raise ValueError(f"Missing column: {col}")
    return st_df


def load_iss_data(iss_file):
    return pd.read_csv(iss_file)


def normalize_coordinates(df, x_col="x", y_col="y"):
    df = df.copy()
    df["x_norm"] = (df[x_col] - df[x_col].min()) / (df[x_col].max() - df[x_col].min())
    df["y_norm"] = (df[y_col] - df[y_col].min()) / (df[y_col].max() - df[y_col].min())
    return df


def align_modalities(st_df, iss_df):
    st_df = normalize_coordinates(st_df)
    iss_df = normalize_coordinates(iss_df)
    return st_df, iss_df


def aggregate_iss_to_st(st_df, iss_df, radius=0.03):
    iss_coords = iss_df[["x_norm", "y_norm"]].values
    tree = cKDTree(iss_coords)
    results = []

    for idx, row in st_df.iterrows():
        spot_coord = [row["x_norm"], row["y_norm"]]
        nearby_idx = tree.query_ball_point(spot_coord, r=radius)
        nearby_cells = iss_df.iloc[nearby_idx]

        results.append({
            "spot_id": idx,
            "ST_cNCC": row["cnc_score"],
            "ST_SPC": row["schwann_score"],
            "ISS_cNCC": np.sum(nearby_cells["cell_type"] == "cNCC"),
            "ISS_SPC": np.sum(nearby_cells["cell_type"] == "SPC")
        })

    return pd.DataFrame(results)


def compute_correlations(validation_df):
    metrics = {}

    pearson_cnc  = pearsonr(validation_df["ST_cNCC"],  validation_df["ISS_cNCC"])
    spearman_cnc = spearmanr(validation_df["ST_cNCC"], validation_df["ISS_cNCC"])
    pearson_spc  = pearsonr(validation_df["ST_SPC"],   validation_df["ISS_SPC"])
    spearman_spc = spearmanr(validation_df["ST_SPC"],  validation_df["ISS_SPC"])

    metrics["cNCC"] = {
        "pearson_r":  pearson_cnc.statistic,
        "pearson_p":  pearson_cnc.pvalue,
        "spearman_r": spearman_cnc.statistic,
        "spearman_p": spearman_cnc.pvalue
    }
    metrics["SPC"] = {
        "pearson_r":  pearson_spc.statistic,
        "pearson_p":  pearson_spc.pvalue,
        "spearman_r": spearman_spc.statistic,
        "spearman_p": spearman_spc.pvalue
    }

    return metrics


def plot_concordance(validation_df, st_col, iss_col, title, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(validation_df[st_col], validation_df[iss_col], alpha=0.6)
    ax.set_xlabel(st_col)
    ax.set_ylabel(iss_col)
    ax.set_title(title)
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    plt.close(fig)


def identify_mismatches(validation_df, threshold=0.7):
    validation_df = validation_df.copy()
    validation_df["cNCC_diff"] = np.abs(validation_df["ST_cNCC"] - validation_df["ISS_cNCC"])
    validation_df["SPC_diff"]  = np.abs(validation_df["ST_SPC"]  - validation_df["ISS_SPC"])
    mismatches = validation_df[
        (validation_df["cNCC_diff"] > threshold) |
        (validation_df["SPC_diff"]  > threshold)
    ]
    return mismatches


def generate_validation_summary(metrics, pcw_label=None, save_path=None):
    header = f"===== VALIDATION SUMMARY{f' [{pcw_label}]' if pcw_label else ''} ====="
    lines = [f"\n{header}\n"]

    for celltype, vals in metrics.items():
        lines.append(f"{celltype}:")
        lines.append(f"  Pearson r  = {vals['pearson_r']:.3f} (p={vals['pearson_p']:.3e})")
        lines.append(f"  Spearman r = {vals['spearman_r']:.3f} (p={vals['spearman_p']:.3e})")

        if vals["pearson_r"] > 0.5:
            lines.append("  Strong concordance")
        elif vals["pearson_r"] > 0.2:
            lines.append("  Moderate concordance")
        else:
            lines.append("  Weak concordance")
        lines.append("")

    summary_text = "\n".join(lines)
    print(summary_text)

    if save_path:
        with open(save_path, "w") as f:
            f.write(summary_text)
        print(f"  Summary saved: {save_path}")


def run_cross_modality_validation(
    st_file,
    iss_file,
    pcw_label,
    radius=0.03,
    output_dir="validation_output"
):
    # Per-PCW subdirectory
    pcw_dir = os.path.join(output_dir, pcw_label)
    os.makedirs(pcw_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Running validation for {pcw_label}")
    print(f"{'='*50}")

    # Load & align
    st_df  = load_st_data(st_file)
    iss_df = load_iss_data(iss_file)
    st_df, iss_df = align_modalities(st_df, iss_df)

    # Aggregate
    validation_df = aggregate_iss_to_st(st_df, iss_df, radius=radius)

    # Save validation table
    validation_csv = os.path.join(pcw_dir, "validation_results.csv")
    validation_df.to_csv(validation_csv, index=False)
    print(f"  Validation table saved: {validation_csv}")

    # Correlations
    metrics = compute_correlations(validation_df)

    # Save metrics
    metrics_json = os.path.join(pcw_dir, "correlation_metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_json}")

    # Plots
    plot_concordance(
        validation_df, "ST_cNCC", "ISS_cNCC",
        f"ST cNCC vs ISS cNCC [{pcw_label}]",
        save_path=os.path.join(pcw_dir, "concordance_cNCC.png")
    )
    plot_concordance(
        validation_df, "ST_SPC", "ISS_SPC",
        f"ST SPC vs ISS SPC [{pcw_label}]",
        save_path=os.path.join(pcw_dir, "concordance_SPC.png")
    )

    # Mismatches
    mismatches = identify_mismatches(validation_df)
    mismatches_csv = os.path.join(pcw_dir, "mismatches.csv")
    mismatches.to_csv(mismatches_csv, index=False)
    print(f"  Mismatches saved: {mismatches_csv}")

    # Summary
    generate_validation_summary(
        metrics,
        pcw_label=pcw_label,
        save_path=os.path.join(pcw_dir, "validation_summary.txt")
    )

    return validation_df, metrics, mismatches


def save_combined_metrics(all_metrics, output_dir):
    """Save a single CSV comparing correlation metrics across all PCW timepoints."""
    rows = []
    for pcw_label, metrics in all_metrics.items():
        for celltype, vals in metrics.items():
            rows.append({
                "pcw": pcw_label,
                "cell_type": celltype,
                **vals
            })

    combined_df = pd.DataFrame(rows)
    save_path = os.path.join(output_dir, "combined_metrics.csv")
    combined_df.to_csv(save_path, index=False)
    print(f"\nCombined metrics saved: {save_path}")
    return combined_df


def plot_metrics_across_pcw(all_metrics, output_dir):
    """Bar plot of Pearson r for each cell type across PCW timepoints."""
    pcw_labels = list(all_metrics.keys())
    cell_types = ["cNCC", "SPC"]
    x = np.arange(len(pcw_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ct in enumerate(cell_types):
        pearson_vals = [all_metrics[pcw][ct]["pearson_r"] for pcw in pcw_labels]
        ax.bar(x + i * width, pearson_vals, width, label=ct, alpha=0.8)

    ax.set_xlabel("Post-Conception Week")
    ax.set_ylabel("Pearson r")
    ax.set_title("ST vs ISS Concordance Across PCW Timepoints")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(pcw_labels)
    ax.legend()
    ax.grid(axis="y")

    save_path = os.path.join(output_dir, "pearson_r_across_pcw.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cross-PCW plot saved: {save_path}")


def main():
    # Map each PCW label to its ISS data file
    PCW_ISS_FILES = {
        "4.5PCW": "../dataset/dataset_iss/PCW4.5_merge_data.csv",
        "6.5PCW": "../dataset/dataset_iss/PCW6.5_merge_data.csv",
        "9.5PCW": "../dataset/dataset_iss/PCW9.5_merge_data.csv",
    }

    ST_FILE    = "../dataset/dataset_st/Filtered/ST_spatial_scores_aligned.csv"
    OUTPUT_DIR = "validation_output"
    RADIUS     = 0.03

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics     = {}
    all_validation  = {}
    all_mismatches  = {}

    for pcw_label, iss_file in PCW_ISS_FILES.items():
        validation_df, metrics, mismatches = run_cross_modality_validation(
            st_file=ST_FILE,
            iss_file=iss_file,
            pcw_label=pcw_label,
            radius=RADIUS,
            output_dir=OUTPUT_DIR
        )
        all_metrics[pcw_label]    = metrics
        all_validation[pcw_label] = validation_df
        all_mismatches[pcw_label] = mismatches

    # Cross-PCW summary outputs
    save_combined_metrics(all_metrics, OUTPUT_DIR)
    plot_metrics_across_pcw(all_metrics, OUTPUT_DIR)

    return all_validation, all_metrics, all_mismatches


if __name__ == "__main__":
    main()