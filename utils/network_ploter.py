import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

fig_size = (10, 8)

# Use standard academic fonts/style
plt.style.use("seaborn-v0_8-paper")  # Or 'ggplot' if not found


def plot_matrix(matrix, title, results_path, is_bandwidth=False):
    matrix = np.array(matrix, dtype=float)
    matrix[matrix >= 1e9] = np.nan  # Handle unconnected nodes

    # 1. Setup small, precise figure size for a 2-column paper
    plt.figure(figsize=fig_size)

    # 2. Select paper-friendly colormaps
    # Sequential, perceptually uniform maps work best for printing
    # Select color maps
    base_cmap = "flare" if is_bandwidth else "crest"
    cmap = plt.get_cmap(base_cmap).copy()

    # Give the "N/A" cells a distinct neutral gray color
    cmap.set_bad(color="#FFFFFF")

    # 3. Create the heatmap with Seaborn
    # If the matrix is small (e.g., 10x10 or less), set annot=True
    ax = sns.heatmap(
        matrix,
        cmap=cmap,
        annot=False,  # Set to True to print numbers in cells
        fmt=".1f",  # 1 decimal place if annotated
        cbar_kws={"label": "Bandwidth (Mbps)" if is_bandwidth else "Delay (ms)"},
        linewidths=0.5,
        linecolor="white",  # Adds a clean grid
    )

    # 4. Clean up labels
    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("Destination Node", fontsize=9)
    plt.ylabel("Source Node", fontsize=9)

    # Save the file
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(
        os.path.join(results_path, f"{title}.png"),  # PDFs scale perfectly
        dpi=600,  # High resolution for print
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_graph(graph, results_path):
    INF = 1e9
    G = nx.DiGraph()
    N = len(graph)

    for i in range(N):
        for j in range(N):
            if graph[i][j] < INF:
                G.add_edge(i, j, weight=graph[i][j])

    plt.figure(figsize=fig_size)

    pos = nx.spring_layout(G, seed=42, k=0.3)  # better spacing

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="black")

    # Edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=False)  # reduce clutter

    # Labels (small!)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="black")

    plt.title("Network Topology", fontsize=14)
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(results_path, "network_graph.png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
