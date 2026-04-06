import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, rankdata
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run(base_path="results/best_fit", save_dir="results/Analysis/TABLES"):
    os.makedirs(save_dir, exist_ok=True)

    # ─── Discover CSVs ───────────────────────────────────────────
    optim_files = sorted(f for f in os.listdir(base_path) if f.endswith(".csv"))
    optim_names = [os.path.splitext(f)[0] for f in optim_files]

    # ─── Read raw dataframes ─────────────────────────────────────
    # Each CSV: rows = trials, columns = benchmark functions
    df_dict = {
        opt: pd.read_csv(os.path.join(base_path, fname))
        for fname, opt in zip(optim_files, optim_names)
    }

    # ─── Summary stats: mean across trials for each function ─────
    # mean_numeric: index = functions, columns = optimizers
    mean_numeric = pd.DataFrame({opt: df.mean() for opt, df in df_dict.items()})

    # ─── Raw per-trial vectors for statistical tests ──────────────
    # trial_dict[opt] = 1-D array of all trial values (across all functions)
    trial_dict = {opt: df.values.flatten() for opt, df in df_dict.items()}
    n_obs = len(next(iter(trial_dict.values())))

    # ─── Average ranking table ───────────────────────────────────
    ranks = mean_numeric.rank(axis=1, method="min", ascending=True)
    avg_rank = ranks.mean()
    position = avg_rank.rank(method="min", ascending=True).astype(int)

    rank_table = pd.DataFrame(
        {
            "Average ranking on functions": avg_rank.round(4).astype(str)
            + " ("
            + position.astype(str)
            + ")"
        }
    )
    rank_table.index.name = "Algorithm"
    rank_table = rank_table.sort_values("Average ranking on functions")
    rank_table.to_csv(os.path.join(save_dir, "friedman_ranking.csv"))
    print("💜 Average ranking table saved successfully")

    # ─── Helper: safe wilcoxon ────────────────────────────────────
    def safe_wilcoxon(a, b, **kwargs):
        """Return (stat, p) or (nan, nan) when the test cannot run."""
        try:
            return wilcoxon(a, b, **kwargs)
        except ValueError:
            return (np.nan, np.nan)

    if n_obs < 2:
        print(f"⚠️  Only {n_obs} observation(s) — skipping Wilcoxon tests (need ≥ 2).")
        # Write empty placeholder CSVs so downstream steps don't crash
        empty = pd.DataFrame(index=optim_names, columns=optim_names)
        empty.to_csv(os.path.join(save_dir, "wilcoxon_pvalues.csv"))
        empty.to_csv(os.path.join(save_dir, "wilcoxon_one_sided_less.csv"))
        for ref in optim_names:
            empty.to_csv(os.path.join(save_dir, f"wilcoxon_{ref}_vs_all.csv"))
        return

    # ─── All-pairs two-sided Wilcoxon p-values ───────────────────
    pval_matrix = pd.DataFrame(index=optim_names, columns=optim_names, dtype=float)
    for i, opt1 in enumerate(optim_names):
        for j, opt2 in enumerate(optim_names):
            if i < j:
                _, p = safe_wilcoxon(trial_dict[opt1], trial_dict[opt2])
                pval_matrix.at[opt1, opt2] = p
                pval_matrix.at[opt2, opt1] = p
            elif i == j:
                pval_matrix.at[opt1, opt2] = np.nan

    pval_matrix = pval_matrix.map(lambda x: f"{x:.4e}" if pd.notnull(x) else "")
    pval_matrix.to_csv(os.path.join(save_dir, "wilcoxon_pvalues.csv"))
    print("💜 Two-sided Wilcoxon p-values saved successfully")

    # ─── All-pairs one-sided Wilcoxon ────────────────────────────
    pval_one = pd.DataFrame(index=optim_names, columns=optim_names, dtype=float)
    for i, opt1 in enumerate(optim_names):
        for j, opt2 in enumerate(optim_names):
            if i < j:
                _, p = safe_wilcoxon(trial_dict[opt1], trial_dict[opt2], alternative="less")
                pval_one.at[opt1, opt2] = p

    pval_one = pval_one.map(lambda x: f"{x:.4e}" if pd.notnull(x) else "")
    pval_one.to_csv(os.path.join(save_dir, "wilcoxon_one_sided_less.csv"))
    print("💜 All-pairs one-sided Wilcoxon (opt_i < opt_j) saved successfully")

    # ─── Per-optimizer signed-rank ───────────────────────────────
    for ref in optim_names:
        rows = []
        for other in optim_names:
            if other == ref:
                continue

            x = trial_dict[ref]
            y = trial_dict[other]
            d = x - y

            nz = d != 0
            if nz.sum() < 2:
                rows.append({"versus": other, "R+": np.nan, "R-": np.nan, "p-value": "N/A"})
                continue

            r = rankdata(np.abs(d[nz]))
            R_plus = r[d[nz] < 0].sum()
            R_minus = r[d[nz] > 0].sum()

            _, p = safe_wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            p_str = f"{p:.4e}" if not np.isnan(p) else "N/A"

            rows.append({"versus": other, "R+": R_plus, "R-": R_minus, "p-value": p_str})

        df_vs = pd.DataFrame(rows).set_index("versus")
        outp = os.path.join(save_dir, f"wilcoxon_{ref}_vs_all.csv")
        df_vs.to_csv(outp)
        print(
            f"💜 {outp.replace('results/Analysis/TABLES/','').replace('.csv','')} saved successfully"
        )
