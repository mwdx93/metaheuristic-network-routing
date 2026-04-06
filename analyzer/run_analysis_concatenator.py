import pandas as pd
from pathlib import Path

groups = ["classical", "cec2017", "cec2019", "cec2021", "cec2022", "real_world_problems"]
base_dir = Path("results")
tables_subpath = Path("Analysis/TABLES")

# discover all optimizer names
all_paths = []
for grp in groups:
    all_paths += list((base_dir/grp/tables_subpath).glob("wilcoxon_*_vs_all.csv"))
optimizers = {
    p.stem.removeprefix("wilcoxon_").removesuffix("_vs_all")
    for p in all_paths
}

for opt in optimizers:
    dfs = []
    for grp in groups:
        p = base_dir / grp / tables_subpath / f"wilcoxon_{opt}_vs_all.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        # assume first column is the “comparison key” (e.g. test name); set it as index so rows align
        key_col = df.columns[0]
        df = df.set_index(key_col)
        # prefix all other columns with the group name to avoid name clashes
        df = df.add_prefix(f"{grp}_")
        dfs.append(df)

    if dfs:
        # horizontal concat: axis=1
        big = pd.concat(dfs, axis=1)
        # put the key back as a column
        big.reset_index(inplace=True)
        out_path = base_dir / f"wilcoxon_{opt}_vs_all.csv"
        big.to_csv(out_path, index=False)
        print(f"▶ wrote {out_path.name}")
