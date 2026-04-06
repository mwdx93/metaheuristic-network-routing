import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"


def run(tables_dir="results/Analysis/TABLES", save_dir="results/Analysis/PLOTS/non_parametric"):
    os.makedirs(save_dir, exist_ok=True)

    wilcoxon_path = os.path.join(tables_dir, "wilcoxon_pvalues.csv")
    friedman_path = os.path.join(tables_dir, "friedman_ranking.csv")

    # ─── 1) Friedman Average Ranks Heatmap ───────────────────────
    df_fr = pd.read_csv(friedman_path, index_col=0)
    df_fr["avg_rank"] = (
        df_fr["Average ranking on functions"].str.extract(r"([\d\.]+)").astype(float)
    )
    df_fr = df_fr.sort_values("avg_rank")
    data_fr = df_fr["avg_rank"].values.reshape(-1, 1)

    plt.figure(figsize=(6, len(df_fr) * 0.3))
    im = plt.imshow(data_fr, aspect="auto", cmap="Blues")
    plt.yticks(np.arange(len(df_fr)), df_fr.index, fontsize=8)
    plt.xticks([0], ["Avg Rank"], fontsize=8)

    for i in range(data_fr.shape[0]):
        plt.text(
            0, i, f"{data_fr[i,0]:.2f}", ha="center", va="center", color="black", fontsize=7
        )

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Average Rank", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    plt.title("Friedman Average Ranks", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "friedman_ranking_heatmap_annotated.pdf"))
    plt.close()
    print("💜 Friedman heatmap with annotations saved")

    # ─── 2) Wilcoxon p-values Heatmap ────────────────────────────
    df_wv = pd.read_csv(wilcoxon_path, index_col=0)
    df_wv = df_wv.replace("", np.nan).astype(float)
    data_wv = df_wv.values

    plt.figure(figsize=(8, 6))
    im = plt.imshow(data_wv, aspect="auto", cmap="Blues")
    plt.yticks(np.arange(len(df_wv)), df_wv.index, fontsize=8)
    plt.xticks(np.arange(len(df_wv.columns)), df_wv.columns, rotation=90, fontsize=8)

    for i in range(data_wv.shape[0]):
        for j in range(data_wv.shape[1]):
            val = data_wv[i, j]
            txt = f"{val:.3f}" if not np.isnan(val) else ""
            plt.text(j, i, txt, ha="center", va="center", color="black", fontsize=7)

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("p-value", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    plt.title("Wilcoxon Two-sided p-values", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wilcoxon_pvalues_heatmap_annotated.pdf"))
    plt.close()
    print("💜 Wilcoxon heatmap with annotations saved")
