from utils.problems import BestPathProblem, generate_network
from mealpy import Multitask
import os
import time
import numpy as np
from utils.network_ploter import plot_graph, plot_matrix
from utils import algorithms as algos
from mealpy.evolutionary_based import SHADE
from mealpy.swarm_based import GWO, ABC, DO, BA, PSO, FOX
from mealpy.evolutionary_based import GA
from mealpy.bio_based import BBO

print("⏳ Please wait...")


# ── Network parameters ─────────────────────────────────────────────
NODES = 100
SOURCE = 0
DESTINATION = NODES - 1
MAX_DELAY = 200  # ms
MIN_BW = 20  # Mbps
topology = "scale_free"  # "random", "grid", or "scale_free "

# OPTIMIZER SETTINGS

n_trials = 30
epoch = 500
n_agents = 50
results_path = f"results/{NODES}_nodes_{topology}"
os.makedirs(results_path, exist_ok=True)


# ── Generate a guaranteed-connected network ────────────────────────
# Change topology= to "grid" or "scale_free" for other experiments
graph, delay_matrix, bandwidth_matrix = generate_network(
    N=NODES,
    connectivity=0.2,
    topology=topology,
)
plot_graph(graph, results_path)
plot_matrix(delay_matrix, "Delay Matrix", results_path)
plot_matrix(bandwidth_matrix, "Bandwidth Matrix", results_path, is_bandwidth=True)

# ── Build problem ──────────────────────────────────────────────────
best_path_problem = BestPathProblem(
    graph,
    delay_matrix,
    bandwidth_matrix,
    source=SOURCE,
    destination=DESTINATION,
    max_delay=MAX_DELAY,
    min_bandwidth=MIN_BW,
    weights=(0.4, 0.3, 0.3),  # (cost, delay, bandwidth)
)
problem = best_path_problem.create_problem()

# ── Algorithms ─────────────────────────────────────────────────────
algorithms = [
    algos.IFOX13(epoch=epoch, pop_size=n_agents, name="IFOX"),
    GA.BaseGA(epoch=epoch, pop_size=n_agents, name="GA"),
    PSO.OriginalPSO(epoch=epoch, pop_size=n_agents, name="PSO"),
    GWO.OriginalGWO(epoch=epoch, pop_size=n_agents, name="GWO"),
    SHADE.L_SHADE(epoch=epoch, pop_size=n_agents, name="LSHADE"),
    ABC.OriginalABC(epoch=epoch, pop_size=n_agents, name="ABC"),
    DO.OriginalDO(epoch=epoch, pop_size=n_agents, name="DO"),
    BA.OriginalBA(epoch=epoch, pop_size=n_agents, name="BA"),
    FOX.OriginalFOX(epoch=epoch, pop_size=n_agents, name="FOX"),
    BBO.OriginalBBO(epoch=epoch, pop_size=n_agents, name="BBO"),
    # Add more comparisons:
]

# ── Run ────────────────────────────────────────────────────────────
start_time = time.perf_counter()
multitask = Multitask(algorithms, [problem], modes=("swarm",), n_workers=10)
multitask.execute(
    n_trials=n_trials,
    n_jobs=10,
    save_path=results_path,
    save_as="csv",
    save_convergence=True,
    verbose=True,
)
print(
    f"{20*'---'}\n"
    f"Total run time: {round((time.perf_counter() - start_time) / 60, 2)} minutes"
)

# ── Analyze results ────────────────────────────────────────────────────
print("\n📊 Running analyzer...")
from analyzer import _write_results, _write_analysis, _plot_ranks, _plot_convergence

_write_results.run(
    os.path.join(results_path, "best_fit"),
    os.path.join(results_path, "Analysis/TABLES"),
)

_write_analysis.run(
    os.path.join(results_path, "best_fit"),
    os.path.join(results_path, "Analysis/TABLES"),
)
_plot_ranks.run(
    os.path.join(results_path, "Analysis/TABLES"),
    os.path.join(results_path, "Analysis/PLOTS/non_parametric"),
)
_plot_convergence.run(
    os.path.join(results_path, "convergence"),
    os.path.join(results_path, "Analysis/PLOTS"),
)
print("✅ Analysis complete.")
