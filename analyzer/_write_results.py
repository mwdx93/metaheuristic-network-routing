import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run(base_path="results/best_fit", save_dir="results/Analysis/TABLES"):
    os.makedirs(save_dir, exist_ok=True)

    optim_files = sorted(f for f in os.listdir(base_path) if f.endswith(".csv"))
    optim_names = [os.path.splitext(f)[0] for f in optim_files]

    df_dict = {
        opt: pd.read_csv(os.path.join(base_path, fname))
        for fname, opt in zip(optim_files, optim_names)
    }

    mean_df = pd.DataFrame({opt: df.mean() for opt, df in df_dict.items()})
    std_df = pd.DataFrame({opt: df.std() for opt, df in df_dict.items()})
    best_df = pd.DataFrame({opt: df.min() for opt, df in df_dict.items()})
    worst_df = pd.DataFrame({opt: df.max() for opt, df in df_dict.items()})

    alpha = 0.05

    def pairwise_brt(a, b):
        try:
            _, p = wilcoxon(a, b)
        except:
            return (0, 1, 0)
        if p < alpha:
            return (1, 0, 0) if np.median(a) < np.median(b) else (0, 0, 1)
        return (0, 1, 0)

    brt_df = pd.DataFrame(index=mean_df.index, columns=optim_names)
    for func in mean_df.index:
        for opt in optim_names:
            w = t = l = 0
            for other in optim_names:
                if other == opt:
                    continue
                wi, ti, li = pairwise_brt(df_dict[opt][func], df_dict[other][func])
                w += wi
                t += ti
                l += li
            brt_df.at[func, opt] = f"{w}/{t}/{l}"

    rank_df = mean_df.rank(axis=1, method="min")

    numeric = {
        "Avg": mean_df,
        "Std": std_df,
        "Best": best_df,
        "Worst": worst_df,
        "BRT": "'" + brt_df,
        "Rank": rank_df,
    }

    metrics_order = ["Avg", "Std", "Best", "Worst", "BRT", "Rank"]
    rows, idx = [], []
    for func in mean_df.index:
        for m in metrics_order:
            idx.append((func, m))
            rows.append([numeric[m].at[func, opt] for opt in optim_names])

    final = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["Function", "Metric"]),
        columns=optim_names,
    )

    total_brt_row = []
    for opt in optim_names:
        w_sum = t_sum = l_sum = 0
        for func in brt_df.index:
            w, t, l = map(int, brt_df.at[func, opt].split("/"))
            w_sum += w
            t_sum += t
            l_sum += l
        total_brt_row.append(f"{w_sum}/{t_sum}/{l_sum}")

    final.loc[("Total", "BRT"), :] = total_brt_row

    out_path = os.path.join(save_dir, "final_results_transposed.csv")
    final.to_csv(out_path)
    print("💜 Final results saved successfully")
