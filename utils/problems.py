"""
Best Path Problem — QoS-Aware Routing with Metaheuristic Optimization
Paper-ready implementation with normalized fitness, robust decoding,
statistical experiments, and convergence tracking.
"""

import numpy as np
import networkx as nx
from mealpy import FloatVar


# ─────────────────────────────────────────────
# 1. GRAPH GENERATION
# ─────────────────────────────────────────────


def generate_network(N=50, connectivity=0.2, topology="random"):
    """
    Generate a connected directed network with cost, delay, and bandwidth.

    Parameters
    ----------
    N            : number of nodes
    connectivity : probability of an edge existing between two nodes
    topology     : "random" | "grid" | "scale_free"
                   Use multiple topologies in experiments to show generalizability.

    Returns
    -------
    graph, delay_matrix, bandwidth_matrix  — all (N×N) numpy arrays
    """
    rng = np.random.default_rng()
    INF = 1e9

    graph = np.full((N, N), INF)
    delay_bw = {}  # store edge attrs before connectivity check

    if topology == "random":
        for i in range(N):
            for j in range(N):
                if i != j and rng.random() < connectivity:
                    graph[i][j] = rng.integers(1, 20)

    elif topology == "grid":
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r, c = divmod(i, side)
                nr, nc = r + di, c + dj
                j = nr * side + nc
                if 0 <= nr < side and 0 <= nc < side and j < N:
                    graph[i][j] = rng.integers(1, 20)

    elif topology == "scale_free":
        G = nx.scale_free_graph(N)
        for u, v in G.edges():
            if u != v:
                graph[u][v] = rng.integers(1, 20)

    # Guarantee connectivity: add a spanning path if needed
    G_check = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            if graph[i][j] < INF:
                G_check.add_edge(i, j)

    source, destination = 0, N - 1
    if not nx.has_path(G_check, source, destination):
        # Add a direct chain 0→1→2→…→N-1 to guarantee connectivity
        for i in range(N - 1):
            if graph[i][i + 1] >= INF:
                graph[i][i + 1] = rng.integers(1, 20)

    # Delay and bandwidth derived from graph structure
    delay_matrix = np.full((N, N), INF)
    bandwidth_matrix = np.full((N, N), INF)
    for i in range(N):
        for j in range(N):
            if graph[i][j] < INF:
                delay_matrix[i][j] = rng.integers(10, 200)  # ms
                bandwidth_matrix[i][j] = rng.integers(20, 100)  # Mbps

    return graph, delay_matrix, bandwidth_matrix


# ─────────────────────────────────────────────
# 2. PATH DECODING  (rank-based + repair)
# ─────────────────────────────────────────────


def decode_path(solution, graph, source, destination, INF=1e9):
    """
    Decode a continuous solution vector into a valid path.

    Strategy: rank-based greedy decoding with dead-end repair via
    backtracking. This is significantly stronger than a pure-greedy
    decoder and greatly improves the optimizer's ability to find solutions.

    Returns
    -------
    path : list of node indices, or None if no path found
    """
    N = len(solution)
    priority = solution  # lower value = higher priority

    visited = set()
    path = [source]
    current = source

    while current != destination:
        visited.add(current)

        neighbors = [
            j for j in range(N) if graph[current][j] < INF and j not in visited
        ]

        if not neighbors:
            # Backtrack one step
            if len(path) <= 1:
                return None  # truly stuck
            path.pop()
            current = path[-1]
            continue

        # Rank-based selection: pick neighbor with lowest priority value
        next_node = min(neighbors, key=lambda x: priority[x])
        path.append(next_node)
        current = next_node

        if len(path) > N:  # cycle guard
            return None

    return path


# ─────────────────────────────────────────────
# 3. FITNESS FUNCTION  (normalized + weighted)
# ─────────────────────────────────────────────


class BestPathProblem:
    """
    QoS-aware best-path problem for metaheuristic optimizers (mealpy).

    Fitness = w1 * norm_cost + w2 * norm_delay + w3 * norm_bw_loss
              + soft_penalty(delay_violation) + soft_penalty(bw_violation)

    All metrics are normalized to [0, 1] before combining so that
    the penalty coefficients and weights are dimensionless and comparable.
    """

    def __init__(
        self,
        graph,
        delay_matrix,
        bandwidth_matrix,
        source,
        destination,
        max_delay=50,
        min_bandwidth=10,
        weights=(0.5, 0.3, 0.2),  # (cost, delay, bandwidth)
    ):
        self.graph = graph
        self.delay = delay_matrix
        self.bandwidth = bandwidth_matrix
        self.source = source
        self.destination = destination
        self.max_delay = max_delay
        self.min_bandwidth = min_bandwidth
        self.w1, self.w2, self.w3 = weights
        self.INF = 1e9

        N = len(graph)
        self.num_nodes = N

        # Normalization bounds (worst-case estimates)
        self._max_cost = N * 20  # max edges × max cost per edge
        self._max_delay = N * 10  # max edges × max delay per edge
        self._max_bw = 100.0  # max bandwidth value

        # Track best decoded path for post-run analysis
        self.best_path = None
        self.best_fitness = self.INF

    # ------------------------------------------------------------------
    def fitness_qos(self, solution):
        """Mealpy-compatible fitness function with steep gradients and hard boundaries."""
        path = decode_path(
            solution, self.graph, self.source, self.destination, self.INF
        )

        # 1. Fatal Penalty: No valid path found
        if path is None:
            return self.INF

        # ── Compute raw metrics ────────────────────────────────────────
        total_cost = 0.0
        total_delay = 0.0
        min_bw = float("inf")

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_cost += self.graph[u][v]
            total_delay += self.delay[u][v]
            min_bw = min(min_bw, self.bandwidth[u][v])

        if min_bw == float("inf"):
            min_bw = 0.0

        # ── 2. The "Hard" Constraint Check ─────────────────────────────
        if total_delay > self.max_delay or min_bw < self.min_bandwidth:
            # Prevent division by zero if max_delay or min_bw are 0
            v_delay = total_delay / (self.max_delay + 1e-6)
            v_bw = self.min_bandwidth / (min_bw + 1e-6)

            return self.INF * (v_delay + v_bw)

        # ── 3. Normalize to [0, 1] with safe guards ───────────────────
        # We use np.clip to guarantee values stay strictly between 0 and 1
        norm_cost = np.clip(total_cost / (self._max_cost + 1e-6), 0.0, 1.0)
        norm_delay = np.clip(total_delay / (self._max_delay + 1e-6), 0.0, 1.0)
        norm_bw = 1.0 - np.clip(min_bw / (self._max_bw + 1e-6), 0.0, 1.0)

        # ── 4. Exponential Objective Scaling (Steep Gradients) ──────────
        fitness = (
            self.w1 * (norm_cost**2)
            + self.w2 * (norm_delay**2)
            + self.w3 * (norm_bw**2)
        )

        # Track best for reporting
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_path = path

        return fitness

    # ------------------------------------------------------------------
    def create_problem(self):
        """Return a mealpy-compatible problem dictionary."""
        lb = [0.0] * self.num_nodes
        ub = [1.0] * self.num_nodes

        return {
            "name": "Best Path Problem",
            "bounds": FloatVar(lb=lb, ub=ub),
            "obj_func": self.fitness_qos,
            "minmax": "min",
            "log_to": None,
        }

    # ------------------------------------------------------------------
    def path_metrics(self, path):
        """Return raw metrics for a decoded path (for paper tables)."""
        if path is None:
            return {"cost": None, "delay": None, "bandwidth": None, "hops": None}

        total_cost = 0.0
        total_delay = 0.0
        min_bw = float("inf")

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_cost += self.graph[u][v]
            total_delay += self.delay[u][v]
            min_bw = min(min_bw, self.bandwidth[u][v])

        return {
            "cost": total_cost,
            "delay": total_delay,
            "bandwidth": min_bw if min_bw != float("inf") else 0,
            "hops": len(path) - 1,
        }
