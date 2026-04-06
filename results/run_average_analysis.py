import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
# 1) Identify all top‐level directories that contain a friedman CSV
root = "."  # or wherever your problem folders live
candidates = os.listdir(root)
problem_dirs = []
for d in candidates:
    path = os.path.join(root, d, "Analysis", "TABLES", "friedman_ranking.csv")
    if os.path.isfile(path):
        problem_dirs.append(d)


# 2) Read each CSV and pull out the avg‐rank series
all_series = {}
for d in problem_dirs:
    csv_path = os.path.join(root, d, "Analysis", "TABLES", "friedman_ranking.csv")
    df = pd.read_csv(csv_path, index_col="Algorithm")
    # extract the number before “(” and convert to float
    df["avg_rank"] = (
        df["Average ranking on functions"].str.extract(r"^([\d\.]+)").astype(float)
    )
    all_series[d] = df["avg_rank"]

# 3) Combine into one DataFrame
ranks_df = pd.DataFrame(all_series)

# 4) Compute the overall mean rank
ranks_df["mean_rank"] = ranks_df.mean(axis=1)

# 5) Sort by the overall mean
ranks_df = ranks_df.sort_values("mean_rank")

# 6) Plot heatmap of the overall mean ranks
plt.figure(figsize=(6, len(ranks_df) * 0.3))
data = ranks_df["mean_rank"].values.reshape(-1, 1)
im = plt.imshow(data, aspect="auto", cmap="YlOrBr")

# y-labels = algorithm names
plt.yticks(np.arange(len(ranks_df)), ranks_df.index, fontsize=8)
plt.xticks([0], ["Mean Rank"], fontsize=8)

# annotate each cell
for i, val in enumerate(data[:, 0]):
    plt.text(0, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=7)

cbar = plt.colorbar(im, pad=0.02)
cbar.set_label("Average Rank", fontsize=8)
cbar.ax.tick_params(labelsize=8)

plt.title("Overall Friedman Average Ranks", fontsize=9)
plt.tight_layout()

# save
save_dir = "AVERAGE_RESULTS/SUMMARY_PLOTS"
os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, "friedman_overall_mean_heatmap.png")
plt.savefig(out_path, bbox_inches="tight", dpi=600)
plt.close()

print(f"✅ Overall Friedman heatmap saved to {out_path}")


# ─── 7) Aggregate Wilcoxon p-values across all problems ───────────────────
wilcoxon_mats = []
alg_names = None

for d in problem_dirs:
    wil_path = os.path.join(root, d, "Analysis", "TABLES", "wilcoxon_pvalues.csv")
    if not os.path.isfile(wil_path):
        continue
    df_w = pd.read_csv(wil_path, index_col=0)
    df_w = df_w.replace("", np.nan).astype(float)
    if alg_names is None:
        alg_names = df_w.index.tolist()
    wilcoxon_mats.append(df_w.values)

# stack into (n_problems, n_alg, n_alg) and mean over axis=0
mean_pvals = np.nanmean(np.stack(wilcoxon_mats, axis=0), axis=0)

# wrap back into a DataFrame
mean_wv_df = pd.DataFrame(mean_pvals, index=alg_names, columns=alg_names)

# save the averaged p-value matrix
avg_wv_csv = os.path.join(save_dir, "wilcoxon_avg_pvalues.csv")
mean_wv_df.to_csv(avg_wv_csv)
print(f"📁 Averaged Wilcoxon p-values saved to {avg_wv_csv}")

# ─── 8) Plot the averaged Wilcoxon heatmap ───────────────────────────────
plt.figure(figsize=(len(alg_names) * 0.3, len(alg_names) * 0.3))
im = plt.imshow(mean_pvals, aspect="auto", cmap="YlOrBr")
plt.yticks(np.arange(len(alg_names)), alg_names, fontsize=6)
plt.xticks(np.arange(len(alg_names)), alg_names, rotation=90, fontsize=6)

# annotate each cell
for i in range(mean_pvals.shape[0]):
    for j in range(mean_pvals.shape[1]):
        val = mean_pvals[i, j]
        if not np.isnan(val):
            plt.text(
                j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=5
            )

cbar = plt.colorbar(im, pad=0.02)
cbar.set_label("Mean p-value", fontsize=8)
cbar.ax.tick_params(labelsize=6)

plt.title("Average Wilcoxon Two-sided p-values", fontsize=9)
plt.tight_layout()
out_wil = os.path.join(save_dir, "wilcoxon_avg_pvalues_heatmap.png")
plt.savefig(out_wil, bbox_inches="tight", dpi=600)
plt.close()
print(f"✅ Averaged Wilcoxon heatmap saved to {out_wil}")
