"""
visualize_features.py
=====================
Create publication-quality figures showing how provenance features
differ across evidence states. For TKDE paper and QE report.

Usage:
    python visualize_features.py \
        --train_csv results/graph_features_train.csv \
        --val_csv results/graph_features_val.csv \
        --output_dir figures/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize provenance features")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="figures")
    return parser.parse_args()


# Color scheme for evidence states
STATE_COLORS = {
    "sufficient": "#2ECC71",      # green
    "insufficient": "#E74C3C",    # red
    "contradicted": "#F39C12",    # orange
    "superseded": "#3498DB",      # blue
}

STATE_ORDER = ["sufficient", "insufficient", "contradicted", "superseded"]

FEATURE_DISPLAY_NAMES = {
    "source_diversity": "Source\nDiversity",
    "text_resolution_rate": "Text\nResolution Rate",
    "avg_evidence_length": "Avg Evidence\nLength",
    "min_evidence_length": "Min Evidence\nLength",
    "duplicate_rate": "Duplicate\nRate",
    "document_overlap_rate": "Document\nOverlap Rate",
    "entity_overlap": "Entity\nOverlap",
    "evidence_count": "Evidence\nCount",
}

FEATURE_NAMES = list(FEATURE_DISPLAY_NAMES.keys())


def load_data(train_csv, val_csv):
    """Load feature CSVs."""
    print(f"Loading {train_csv}...")
    train_df = pd.read_csv(train_csv)
    print(f"  Train: {len(train_df)} instances")
    
    print(f"Loading {val_csv}...")
    val_df = pd.read_csv(val_csv)
    print(f"  Val: {len(val_df)} instances")
    
    return train_df, val_df


# ============================================================
# FIGURE 1: Feature means by evidence state (bar chart)
# ============================================================

def plot_feature_means(df, output_dir, split_name="train"):
    """
    Bar chart showing mean value of each feature per evidence state.
    This is the main figure showing features differentiate states.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    
    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes[i]
        
        means = []
        stds = []
        colors = []
        
        for state in STATE_ORDER:
            state_data = df[df["evidence_state_label"] == state][feat]
            means.append(state_data.mean())
            stds.append(state_data.std())
            colors.append(STATE_COLORS[state])
        
        bars = ax.bar(range(4), means, color=colors, edgecolor="white",
                      linewidth=0.5, alpha=0.85)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean_val:.3f}" if mean_val < 10 else f"{mean_val:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        
        ax.set_xticks(range(4))
        ax.set_xticklabels(["SUF", "INS", "CON", "SUP"], fontsize=9)
        ax.set_title(FEATURE_DISPLAY_NAMES[feat], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=STATE_COLORS[s], edgecolor="white", label=s.capitalize())
        for s in STATE_ORDER
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(f"Provenance Feature Means by Evidence State ({split_name.capitalize()} Set)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"feature_means_{split_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# FIGURE 2: Feature distributions (box plots)
# ============================================================

def plot_feature_distributions(df, output_dir, split_name="train"):
    """
    Box plots showing distribution of each feature per evidence state.
    Shows not just means but spread and outliers.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    
    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes[i]
        
        data_to_plot = []
        for state in STATE_ORDER:
            state_data = df[df["evidence_state_label"] == state][feat].values
            data_to_plot.append(state_data)
        
        bp = ax.boxplot(data_to_plot, labels=["SUF", "INS", "CON", "SUP"],
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        
        for patch, state in zip(bp["boxes"], STATE_ORDER):
            patch.set_facecolor(STATE_COLORS[state])
            patch.set_alpha(0.7)
        
        ax.set_title(FEATURE_DISPLAY_NAMES[feat], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.suptitle(f"Provenance Feature Distributions by Evidence State ({split_name.capitalize()} Set)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"feature_distributions_{split_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# FIGURE 3: Heatmap of normalized feature means
# ============================================================

def plot_feature_heatmap(df, output_dir, split_name="train"):
    """
    Heatmap showing normalized feature means per evidence state.
    Each feature is min-max normalized so patterns are visible
    across features with different scales.
    """
    # Compute means per state
    means = {}
    for state in STATE_ORDER:
        state_data = df[df["evidence_state_label"] == state]
        means[state] = [state_data[feat].mean() for feat in FEATURE_NAMES]
    
    means_df = pd.DataFrame(means, index=FEATURE_NAMES).T
    
    # Normalize each column (feature) to 0-1
    normalized = (means_df - means_df.min()) / (means_df.max() - means_df.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    sns.heatmap(normalized, annot=means_df.round(3), fmt="", cmap="YlOrRd",
                xticklabels=[FEATURE_DISPLAY_NAMES[f].replace("\n", " ") for f in FEATURE_NAMES],
                yticklabels=[s.capitalize() for s in STATE_ORDER],
                linewidths=0.5, ax=ax, cbar_kws={"label": "Normalized Value"})
    
    ax.set_title(f"Feature Means Heatmap ({split_name.capitalize()} Set)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"feature_heatmap_{split_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# FIGURE 4: Per-dataset feature comparison
# ============================================================

def plot_per_dataset(df, output_dir, split_name="train"):
    """
    Grouped bar chart showing feature means per dataset per state.
    Shows whether features behave consistently across HotpotQA, 
    MuSiQue, and FEVER.
    """
    datasets = sorted(df["dataset"].unique())
    
    # Select 4 most discriminative features
    key_features = ["entity_overlap", "text_resolution_rate",
                    "duplicate_rate", "evidence_count"]
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    for i, feat in enumerate(key_features):
        ax = axes[i]
        
        x = np.arange(len(STATE_ORDER))
        width = 0.25
        
        for j, ds in enumerate(datasets):
            means = []
            for state in STATE_ORDER:
                mask = (df["evidence_state_label"] == state) & (df["dataset"] == ds)
                means.append(df[mask][feat].mean())
            
            offset = (j - 1) * width
            bars = ax.bar(x + offset, means, width, label=ds.capitalize(),
                         alpha=0.8, edgecolor="white", linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(["SUF", "INS", "CON", "SUP"], fontsize=9)
        ax.set_title(FEATURE_DISPLAY_NAMES[feat], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        
        if i == 0:
            ax.legend(fontsize=9)
    
    plt.suptitle(f"Key Features by Dataset and Evidence State ({split_name.capitalize()} Set)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"per_dataset_features_{split_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# FIGURE 5: Class distribution sanity check
# ============================================================

def plot_class_distribution(df, output_dir, split_name="train"):
    """
    Stacked bar chart showing class distribution per dataset.
    Confirms balanced classes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    counts = df["evidence_state_label"].value_counts()
    colors = [STATE_COLORS[s] for s in STATE_ORDER]
    bars = ax1.bar(STATE_ORDER, [counts.get(s, 0) for s in STATE_ORDER],
                   color=colors, edgecolor="white", linewidth=0.5)
    
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    
    ax1.set_title("Overall Class Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Per-dataset distribution
    datasets = sorted(df["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, state in enumerate(STATE_ORDER):
        counts = []
        for ds in datasets:
            mask = (df["evidence_state_label"] == state) & (df["dataset"] == ds)
            counts.append(mask.sum())
        
        offset = (i - 1.5) * width
        ax2.bar(x + offset, counts, width, label=state.capitalize(),
                color=STATE_COLORS[state], edgecolor="white", linewidth=0.5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([ds.capitalize() for ds in datasets], fontsize=10)
    ax2.set_title("Class Distribution per Dataset", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)
    
    plt.suptitle(f"Dataset Statistics ({split_name.capitalize()} Set)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"class_distribution_{split_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# FIGURE 6: Feature correlation matrix
# ============================================================

def plot_feature_correlation(df, output_dir, split_name="train"):
    """
    Correlation matrix between the 8 provenance features.
    Shows which features carry independent information.
    """
    feature_df = df[FEATURE_NAMES]
    corr = feature_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                xticklabels=[FEATURE_DISPLAY_NAMES[f].replace("\n", " ") for f in FEATURE_NAMES],
                yticklabels=[FEATURE_DISPLAY_NAMES[f].replace("\n", " ") for f in FEATURE_NAMES],
                linewidths=0.5, ax=ax)
    
    ax.set_title(f"Feature Correlation Matrix ({split_name.capitalize()} Set)",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"feature_correlation_{split_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PROVE-RAG Feature Visualization")
    print("=" * 60)
    
    train_df, val_df = load_data(args.train_csv, args.val_csv)
    
    for df, split_name in [(train_df, "train"), (val_df, "val")]:
        print(f"\n--- Generating figures for {split_name} ---")
        
        plot_feature_means(df, args.output_dir, split_name)
        plot_feature_distributions(df, args.output_dir, split_name)
        plot_feature_heatmap(df, args.output_dir, split_name)
        plot_per_dataset(df, args.output_dir, split_name)
        plot_class_distribution(df, args.output_dir, split_name)
        plot_feature_correlation(df, args.output_dir, split_name)
    
    print(f"\n{'='*60}")
    print(f"ALL FIGURES SAVED to {args.output_dir}/")
    print("=" * 60)
    print(f"\nFigures generated (per split):")
    print(f"  1. feature_means_*.pdf       — Bar chart of mean values per state")
    print(f"  2. feature_distributions_*.pdf — Box plots per state")
    print(f"  3. feature_heatmap_*.pdf     — Normalized heatmap")
    print(f"  4. per_dataset_features_*.pdf — Key features by dataset")
    print(f"  5. class_distribution_*.pdf  — Class balance check")
    print(f"  6. feature_correlation_*.pdf — Feature independence")


if __name__ == "__main__":
    main()