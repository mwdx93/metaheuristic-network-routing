import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"


colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#637939", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#d62728", "#8c6d31", "#843c39", "#7b4173",
    "#a55194", "#6b6ecf", "#ce6dbd", "#9c9ede", "#bd9e39",
]


def run(base_path="results/convergence", save_dir="results/Analysis/PLOTS"):
    os.makedirs(save_dir, exist_ok=True)

    algorithms = os.listdir(base_path)
    algorithms.sort()
    function_files = os.listdir(os.path.join(base_path, algorithms[0]))

    for func_file in function_files:
        plt.figure(figsize=(4, 4))
        ax = plt.gca()

        final_fitnesses = []

        for i, optimizer in enumerate(algorithms):
            file_path = os.path.join(base_path, optimizer, func_file)
            df = pd.read_csv(file_path, sep=",")
            avg_curve = df.mean(axis=1).values

            x = np.arange(len(avg_curve))
            ax.plot(
                x, avg_curve, label=optimizer, color=colors[i % len(colors)], linewidth=0.5
            )
            ax.plot(
                x[-1],
                avg_curve[-1],
                marker="o",
                color=colors[i % len(colors)],
                markersize=1.5,
            )
            final_fitnesses.append((avg_curve[-1], optimizer))

        ax.set_title(f"{func_file.replace('.csv','')}", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Objective", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{func_file.replace('.csv', '')}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.close()

    print("💜 Convergence plots saved successfully")
