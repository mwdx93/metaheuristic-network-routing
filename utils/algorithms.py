
import numpy as np
import cma
from mealpy.optimizer import Optimizer


class ALSHADE(Optimizer):
    """
    Adaptive L-SHADE implemented in Mealpy, converted from MATLAB.
    Features:
      - Historical memory of CR and F (size H)
      - External archive A for diversity
      - Linearly decreasing population size
      - Cauchy-based perturbation for scaling factors
    """
    def __init__(self, epoch=1000, pop_size=50, NPmin=4, p=0.11, rarc=2.6,
                 H=6, mu_CR=0.5, mu_F=0.5, **kwargs):
        super().__init__(**kwargs)
        # Core parameters
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.NPinit = self.validator.check_int("pop_size", pop_size, [NPmin+1, 10000])
        self.pop_size = self.NPinit
        self.NPmin = NPmin
        self.p = p
        self.rarc = rarc
        self.H = H
        # Historical memory
        self.MCR = np.full(H, mu_CR)
        self.MF = np.full(H, mu_F)
        self.MCR[-1], self.MF[-1] = 0.9, 0.9
        self.iM = 0  # memory pointer
        # Archive
        self.A, self.A_fitness = [], []
        # Cauchy sequence
        self.Chy, self.iChy = None, 0
        # Eval counter
        self.FEs = 0
        # Problem placeholders
        self.dim, self.lb, self.ub = None, None, None

    def cauchy_rnd(self, a, b, size):
        p = np.random.rand(size)
        return a + b * np.tan(np.pi * (p - 0.5))

    def initialize(self):
        # Setup problem dimensions and bounds
        self.dim = self.problem.n_dims
        self.lb, self.ub = self.problem.lb, self.problem.ub
        # Initial population
        pop = [self.generate_empty_agent(
            np.random.uniform(self.lb, self.ub, self.dim)) for _ in range(self.NPinit)]
        self.pop = self.update_target_for_population(pop)
        # Sort by fitness and set initial archive
        fitnesses = [ag.target.fitness for ag in self.pop]
        idx = np.argsort(fitnesses)
        self.pop = [self.pop[i] for i in idx]
        best = self.pop[0]
        self.A, self.A_fitness = [best.solution.copy()], [best.target.fitness]
        # Initialize Cauchy sequence
        self.Chy = self.cauchy_rnd(0, 0.1, self.NPinit + 200)
        self.iChy = 0
        # Eval count and global best
        self.FEs = len(self.pop)
        self.g_best = best

    def evolve(self, epoch):
        # First iteration: initialize
        if self.FEs == 0:
            self.initialize()
            return
        # Update population size
        NP_k = int(round(self.NPinit - (self.NPinit - self.NPmin) * epoch / self.epoch))
        NP_k = max(self.NPmin, NP_k)
        self.pop_size = NP_k
        # Extract current solutions and fitness
        X = np.array([ag.solution for ag in self.pop])  # shape: (N, dim)
        fitness = np.array([ag.target.fitness for ag in self.pop])
        N = len(self.pop)
        dim = self.dim
        # Prepare trial arrays
        V = np.zeros((N, dim))
        U = np.zeros((N, dim))
        CR = np.zeros(N)
        F = np.zeros(N)
        S_CR, S_F, S_df = [], [], []
        # Generate memory indices
        r = np.random.randint(0, self.H, size=N)
        for i in range(N):
            # pbest selection
            p_num = max(2, int(round(self.p * N)))
            pbest_idx = np.random.randint(0, p_num)
            # Sample CR from memory
            CR[i] = np.clip(self.MCR[r[i]] + 0.1 * np.random.randn(), 0, 1)
            # Sample F via memory + Cauchy noise
            Fi = -1
            while Fi <= 0:
                Fi = self.MF[r[i]] + self.Chy[self.iChy]
                self.iChy = (self.iChy + 1) % len(self.Chy)
            F[i] = min(Fi, 1)
            # Mutation
            idxs = list(range(N)); idxs.remove(i)
            r1 = np.random.choice(idxs)
            PA = np.vstack((X, np.array(self.A)))  # rows are solutions
            r2 = np.random.randint(PA.shape[0])
            if np.random.rand() < 0.5:
                Vi = X[i] + F[i] * (X[pbest_idx] - X[i]) + F[i] * (X[r1] - PA[r2])
            else:
                xmean = np.mean(np.array(self.A), axis=0)
                Vi = X[i] + F[i] * (xmean - X[i]) + F[i] * (X[r1] - PA[r2])
            V[i] = np.clip(Vi, self.lb, self.ub)
            # Crossover
            jrand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR[i]
            mask[jrand] = True
            U[i] = np.where(mask, V[i], X[i])
        # Evaluate trial solutions
        fitness_U = np.array([self.get_target(U[i]).fitness for i in range(N)])
        new_pop = []
        for i in range(N):
            if fitness_U[i] <= fitness[i]:
                S_CR.append(CR[i]); S_F.append(F[i]); S_df.append(abs(fitness_U[i] - fitness[i]))
                # Archive update
                if len(self.A) < int(round(self.rarc * self.NPinit)):
                    self.A.append(U[i].copy())
                    self.A_fitness.append(fitness_U[i])
                else:
                    idx_rep = np.random.randint(len(self.A))
                    self.A[idx_rep] = U[i].copy()
                    self.A_fitness[idx_rep] = fitness_U[i]
                new_pop.append(self.generate_empty_agent(U[i].copy()))
            else:
                new_pop.append(self.generate_empty_agent(X[i].copy()))
        # Update historical memory
        nS = len(S_CR)
        if nS > 0:
            w = np.array(S_df) / sum(S_df)
            if all(cr == 0 for cr in S_CR):
                self.MCR[self.iM] = -1
            else:
                self.MCR[self.iM] = np.sum(w * np.array(S_CR)**2) / np.sum(w * np.array(S_CR))
            self.MF[self.iM] = np.sum(w * np.array(S_F)**2) / np.sum(w * np.array(S_F))
            self.iM = (self.iM + 1) % (self.H - 1)
        # Build new population and update g_best
        self.pop = self.update_target_for_population(new_pop[:NP_k])
        fits = [ag.target.fitness for ag in self.pop]
        best_idx = int(np.argmin(fits))
        if fits[best_idx] < self.g_best.target.fitness:
            self.g_best = self.pop[best_idx]
        # Update evaluation count
        self.FEs += N

class CMAES(Optimizer):
    """CMA-ES optimizer implemented in pure Python, integrated into Mealpy.
    Reference:
    Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies.
    """
    def __init__(
        self,
        epoch: int = 1000,
        pop_size: int = 100,
        sigma0: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Validate and set parameters
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.sigma0 = float(sigma0)
        self.set_parameters(["epoch", "pop_size", "sigma0"])
        np.random.seed(691)
    
    def initialize_variables(self):
        # Strategy state initialization
        N = self.problem.n_dims  # problem dimension
        self.N = N
        self.mean = np.random.rand(N)  # initial mean, can be improved
        self.sigma = self.sigma0
        self.C = np.eye(N)
        self.pc = np.zeros(N)
        self.ps = np.zeros(N)

        # Strategy parameter setting: selection
        self.lam = self.pop_size
        self.mu = self.lam // 6
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)

        # Strategy parameter setting: adaptation
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs

        # Internal eigen decomposition state
        self.eigen_eval = 0
        self.B = np.eye(N)
        self.D = np.ones(N)
        self.invsqrtC = np.eye(N)
        self.chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    def evolve(self, epoch: int):
        """
        Perform one generation of CMA-ES.
        """
        N, lam = self.N, self.lam
        # Decomposition update
        if self.eigen_eval <= 0:
            self.eigen_eval = int(lam / (self.c1 + self.cmu) / N / 10)
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # enforce symmetry
            D2, B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(D2, 1e-10))
            self.B = B
            self.invsqrtC = B @ np.diag(self.D ** -1) @ B.T
        self.eigen_eval -= 1

        # Sample new population
        arz = np.random.randn(N, lam)
        ary = self.B @ np.diag(self.D) @ arz
        arx = self.mean[:, None] + self.sigma * ary

        # Evaluate fitness
        fitness = [self.get_target(arx[:, i]).fitness for i in range(lam)]
        idx = np.argsort(fitness)
        arx = arx[:, idx]
        arz = arz[:, idx]
        fitness = [fitness[i] for i in idx]

        # Store agents
        pop_new = []
        for i in range(lam):
            sol = arx[:, i]
            sol = self.correct_solution(sol)
            agent = self.generate_empty_agent(sol)
            agent.target = self.get_target(sol)
            pop_new.append(agent)

        # Selection and recombination
        old_mean = self.mean.copy()
        self.mean = arx[:, : self.mu] @ self.weights

        # Cumulation: evolution paths
        y = (self.mean - old_mean) / self.sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * (self.invsqrtC @ y)
        hsig = (
            np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (epoch + 1)))
            / self.chiN
            < 2.4 + 1 / (N + 1)
        )
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y

        # Adapt covariance matrix
        artmp = (1 / self.sigma) * (arx[:, : self.mu] - old_mean[:, None])
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * artmp @ np.diag(self.weights) @ artmp.T
        )

        # Adapt step-size sigma
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

        # Update population in Mealpy
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)
        else:
            self.pop = pop_new
        # Global best update
        self.g_best = min(self.pop, key=lambda ag: ag.target.fitness)

class HO(Optimizer):
    """
    Hippopotamus Optimization (HO) implemented in Mealpy.
    Mimics exploration (river), defense, and escape behaviors of hippopotamuses.
    Phases:
      1) Exploration via group interactions
      2) Defense against predators
      3) Escaping predators (local exploitation)
    """
    def __init__(self, epoch=1000, pop_size=50, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])

    def evolve(self, epoch):
        dim = self.g_best.solution.size
        X = np.array([agent.solution for agent in self.pop])
        fitness = np.array([agent.target.fitness for agent in self.pop])
        # Update global best
        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]
        if epoch == 1 or best_fit < self.g_best.target.fitness:
            self.g_best.solution = X[best_idx].copy()
            self.g_best.target = self.get_target(self.g_best.solution)
        lb, ub = self.problem.lb, self.problem.ub
        new_X = X.copy()
        half = self.pop_size // 2
        # Phase 1: Exploration
        for i in range(half):
            Dominant = self.g_best.solution
            # random parameters
            I1 = np.random.randint(1, 3)
            I2 = np.random.randint(1, 3)
            Ip1 = np.random.randint(0, 2, 2)
            group_size = np.random.randint(1, self.pop_size + 1)
            group = np.random.choice(self.pop_size, group_size, replace=False)
            if group_size > 1:
                MeanGroup = np.mean(X[group], axis=0)
            else:
                MeanGroup = X[group[0]].copy()
            # Alfa choices
            A_choices = [
                I2 * np.random.rand(dim) + (1 - Ip1[0]),
                2 * np.random.rand(dim) - 1,
                np.random.rand(dim),
                I1 * np.random.rand(dim) + (1 - Ip1[1]),
                np.random.rand(dim)
            ]
            A = A_choices[np.random.randint(0, 5)]
            B = A_choices[np.random.randint(0, 5)]
            # Update candidates
            X_P1 = X[i] + np.random.rand() * (Dominant - I1 * X[i])
            T = np.exp(-epoch / self.epoch)
            if T > 0.6:
                X_P2 = X[i] + A * (Dominant - I2 * MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i] + B * (MeanGroup - Dominant)
                else:
                    X_P2 = lb + np.random.rand(dim) * (ub - lb)
            # Bound and evaluate
            X_P1 = np.clip(X_P1, lb, ub)
            X_P2 = np.clip(X_P2, lb, ub)
            f1 = self.get_target(X_P1).fitness
            if f1 < fitness[i]: new_X[i] = X_P1; fitness[i] = f1
            f2 = self.get_target(X_P2).fitness
            if f2 < fitness[i]: new_X[i] = X_P2; fitness[i] = f2
        # Phase 2: Defense
        for i in range(half, self.pop_size):
            predator = lb + np.random.rand(dim) * (ub - lb)
            F_HL = self.get_target(predator).fitness
            distance = np.abs(predator - X[i])
            b = np.random.uniform(2, 4)
            c = np.random.uniform(1, 1.5)
            d = np.random.uniform(2, 3)
            l = np.random.uniform(-2 * np.pi, 2 * np.pi)
            RL = 0.05 * np.random.standard_cauchy(size=dim)
            if fitness[i] > F_HL:
                X_P3 = RL * predator + (b / (c - d * np.cos(l))) * (1.0 / distance)
            else:
                X_P3 = RL * predator + (b / (c - d * np.cos(l))) * (1.0 / (2.0 * distance + np.random.rand(dim)))
            X_P3 = np.clip(X_P3, lb, ub)
            f3 = self.get_target(X_P3).fitness
            if f3 < fitness[i]: new_X[i] = X_P3; fitness[i] = f3
        # Phase 3: Exploitation
        for i in range(self.pop_size):
            LO = lb / epoch
            HI = ub / epoch
            D_choices = [
                2 * np.random.rand(dim) - 1,
                np.random.rand(dim),
                np.random.randn(dim)
            ]
            D = D_choices[np.random.randint(0, 3)]
            X_P4 = X[i] + np.random.rand() * (LO + D * (HI - LO))
            X_P4 = np.clip(X_P4, lb, ub)
            f4 = self.get_target(X_P4).fitness
            if f4 < fitness[i]: new_X[i] = X_P4; fitness[i] = f4
        # Build new population and update best
        pop_new = [self.generate_empty_agent(sol) for sol in new_X]
        self.pop = self.update_target_for_population(pop_new)
        # Final global best check
        fits = [agent.target.fitness for agent in self.pop]
        idx = np.argmin(fits)
        if fits[idx] < self.g_best.target.fitness:
            self.g_best.solution = self.pop[idx].solution.copy()
            self.g_best.target = self.pop[idx].target

class CPO(Optimizer):
    """
    Crested Porcupine Optimizer (CPO) implemented in Mealpy.
    Mimics defense and social behaviors of crested porcupines to solve constrained optimization problems.
    """
    def __init__(self, epoch: int = 1000, pop_size: int = 50, **kwargs):
        super().__init__(**kwargs)
        
        # Validate and set parameters
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"]);

    def evolve(self, epoch: int):
        dim = self.g_best.solution.size
        # Gather current positions and fitness
        X = np.array([agent.solution for agent in self.pop])
        fitness = np.array([agent.target.fitness for agent in self.pop])
        Xp = X.copy()
        lb, ub = self.problem.lb, self.problem.ub
        eps = np.finfo(float).eps

        for i in range(self.pop_size):
            # Random masks and indices
            U1 = np.random.rand(dim) > np.random.rand(dim)
            idx1, idx2, idx3 = np.random.randint(0, self.pop_size, 3)

            if np.random.rand() < np.random.rand():
                # Social defense phase
                if np.random.rand() < np.random.rand():
                    # Perturb towards global best
                    y = (X[i] + X[idx1]) / 2
                    X[i] += np.random.randn(dim) * np.abs(2 * np.random.rand(dim) * self.g_best.solution - y)
                else:
                    # Combine neighbors
                    y = (X[i] + X[idx1]) / 2
                    X[i] = U1 * X[i] + (~U1) * (y + np.random.rand(dim) * (X[idx2] - X[idx3]))
            else:
                # Social exploration phase
                Yt = 2 * np.random.rand(dim) * (1 - epoch/self.epoch)**(epoch/self.epoch)
                U2 = np.where(np.random.rand(dim) < 0.5, 2, -1)
                S = np.random.rand(dim) * U2
                if np.random.rand() < 0.8:
                    St = np.exp(fitness[i] / (np.sum(fitness) + eps))
                    S *= Yt * St
                    X[i] = (1 - U1)*X[i] + U1*(X[idx1] + St*(X[idx2] - X[idx3]) - S)
                else:
                    Mt = np.exp(fitness[i] / (np.sum(fitness) + eps))
                    vt = X[i].copy()
                    Vtp = X[idx1]
                    Ft = np.random.rand(dim) * (Mt * (-vt + Vtp))
                    S *= Yt * Ft
                    X[i] = self.g_best.solution + 0.2*(1 - np.random.rand(dim)) \
                           + np.random.rand(dim)*(U2*self.g_best.solution - X[i]) - S

            # Boundary check
            X[i] = np.clip(X[i], lb, ub)
            # Evaluate fitness
            fit_new = self.get_target(X[i]).fitness
            if fit_new < fitness[i]:
                Xp[i] = X[i].copy()
                fitness[i] = fit_new
                # Update global best
                if fit_new < self.g_best.target.fitness:
                    self.g_best.solution = X[i].copy()
                    self.g_best.target = self.get_target(X[i])
            else:
                X[i] = Xp[i].copy()

        # Build new population and update targets
        pop_new = [self.generate_empty_agent(sol) for sol in X]
        self.pop = self.update_target_for_population(pop_new)

class IFOX(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.decay_rate = 1e-3  
        self.gravity_min = 0.5 * 9.81
        self.gravity_max = 9.81
        self.min_alpha = 1/(self.epoch*2)
    def evolve(self, epoch: int):
        pop_new = []
        # alpha =  1*np.exp(-self.decay_rate * epoch)
        # alpha = max(alpha, self.min_alpha)

        # alpha = self.min_alpha + (1 - self.min_alpha) * (1 - epoch / (self.epoch))
        # slower decay: only hits min_alpha at final epoch
        
        alpha = 1 - (epoch/self.epoch)*(1-self.min_alpha)
        for idx in range(self.pop_size):
            t = self.generator.random()
            gravity = self.generator.uniform(self.gravity_min, self.gravity_max) #New
            jump = gravity * t ** 2 #New
            jump = max(jump, 1e-6) #New

            dis = 0.5 * self.g_best.solution
            beta = self.generator.uniform(-alpha, alpha, size=self.g_best.solution.size)
            
            exploit_sol = dis * jump * beta                     #Exploration

            explore_sol = self.g_best.solution + beta * alpha  #Exploitation
            

            # pick any improvement over current g_best
            # 1. Compute fitness of each candidate
            f1 = self.get_target(exploit_sol).fitness
            f2 = self.get_target(explore_sol).fitness
            fg = self.g_best.target.fitness
            # 2. Pick the candidate with the lower fitness
            if f1 < f2:
                best_candidate = exploit_sol
                f_current   = f1
            else:
                best_candidate = explore_sol
                f_current   = f2
            
            #NEW compare further (Elitism)
            # 3. If that candidate is better than the current global best, accept it;
            #    otherwise do a small random move around the global best
            if f_current < fg:
                pos_new = best_candidate
            else:
                pos_new = self.g_best.solution + beta * 1e-3


            

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)

class IFOX2(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.decay_rate = 1e-3  
        self.gravity_min = 0.5 * 9.81
        self.gravity_max = 9.81
        self.min_alpha = 1/(self.epoch*2)
    def evolve(self, epoch: int):
        pop_new = []
        # alpha =  1*np.exp(-self.decay_rate * epoch)
        # alpha = max(alpha, self.min_alpha)

        # alpha = self.min_alpha + (1 - self.min_alpha) * (1 - epoch / (self.epoch))
        # slower decay: only hits min_alpha at final epoch
        
        alpha = 1 - (epoch/self.epoch)*(1-self.min_alpha)
        for idx in range(self.pop_size):
            t = self.generator.random()
            # gravity = self.generator.uniform(self.gravity_min, self.gravity_max) #New
            jump = self.gravity_min * t ** 2 #New
            jump = max(jump, 1e-6) #New

            dis = 0.5 * self.g_best.solution
            beta = self.generator.uniform(-alpha, alpha, size=self.g_best.solution.size)
            
            exploit_sol = dis * jump * beta                     #Exploration

            explore_sol = self.g_best.solution + beta * alpha  #Exploitation
            

            # pick any improvement over current g_best
            # 1. Compute fitness of each candidate
            f1 = self.get_target(exploit_sol).fitness
            f2 = self.get_target(explore_sol).fitness
            fg = self.g_best.target.fitness
            # 2. Pick the candidate with the lower fitness
            if f1 < f2:
                best_candidate = exploit_sol
                f_current   = f1
            else:
                best_candidate = explore_sol
                f_current   = f2
            
            #NEW compare further (Elitism)
            # 3. If that candidate is better than the current global best, accept it;
            #    otherwise do a small random move around the global best
            if f_current < fg:
                pos_new = best_candidate
            else:
                pos_new = self.g_best.solution + beta * 1e-5


            

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)


class IFOX3(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.decay_rate = 1e-3  
        self.gravity_min = 0.5 * 9.81
        self.gravity_max = 9.81
        self.min_alpha = 1/(self.epoch*2)
    def evolve(self, epoch: int):
        pop_new = []
        # alpha =  1*np.exp(-self.decay_rate * epoch)
        # alpha = max(alpha, self.min_alpha)

        # alpha = self.min_alpha + (1 - self.min_alpha) * (1 - epoch / (self.epoch))
        # slower decay: only hits min_alpha at final epoch
        
        alpha = 1 - (epoch/self.epoch)*(1-self.min_alpha)
        for idx in range(self.pop_size):
            t = self.generator.random()
            # gravity = self.generator.uniform(self.gravity_min, self.gravity_max) #New
            jump = self.gravity_min * t ** 2 #New
            jump = max(jump, 1e-6) #New

            dis = 0.5 * self.g_best.solution
            beta = self.generator.uniform(-alpha, alpha, size=self.g_best.solution.size)
            
            exploit_sol = dis * jump * beta                     #Exploration

            explore_sol = self.g_best.solution + beta * alpha  #Exploitation
            

            # pick any improvement over current g_best
            # 1. Compute fitness of each candidate
            f1 = self.get_target(exploit_sol).fitness
            f2 = self.get_target(explore_sol).fitness
            fg = self.g_best.target.fitness
            # 2. Pick the candidate with the lower fitness
            if f1 < f2:
                best_candidate = exploit_sol
                f_current   = f1
            else:
                best_candidate = explore_sol
                f_current   = f2
            
            #NEW compare further (Elitism)
            # 3. If that candidate is better than the current global best, accept it;
            #    otherwise do a small random move around the global best
            if f_current < fg:
                pos_new = best_candidate
            else:
                pos_new = self.g_best.solution + abs(beta*1e-5) # because the beta has negative values and the solutions may have it also the over all direction changed


            

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)

class IFOX4(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.decay_rate = 1e-3  
        self.gravity_min = 0.5 * 9.81
        self.gravity_max = 9.81
        self.min_alpha = 1/(self.epoch*2)
    def evolve(self, epoch: int):
        pop_new = []
        # alpha =  1*np.exp(-self.decay_rate * epoch)
        # alpha = max(alpha, self.min_alpha)

        # alpha = self.min_alpha + (1 - self.min_alpha) * (1 - epoch / (self.epoch))
        # slower decay: only hits min_alpha at final epoch
        
        alpha = 1 - (epoch/self.epoch)*(1-self.min_alpha)
        
        beta = self.generator.uniform(-alpha, alpha, size=self.g_best.solution.size) #Outside the pop's loop to reduce the aggression of movement #NEW
        for idx in range(self.pop_size):
            t = self.generator.random() - alpha #NEW
            # gravity = self.generator.uniform(self.gravity_min, self.gravity_max) #New
            jump = self.gravity_min * t ** 2 #New
            jump = max(jump, 1e-6) #New
            dis = 0.5 * self.g_best.solution
            
            explore_sol = dis * jump * beta                     #Exploration
            exploit_sol = self.g_best.solution + beta * alpha  #Exploitation

            # pick any improvement over current g_best
            # 1. Compute fitness of each candidate
            f1 = self.get_target(explore_sol).fitness
            f2 = self.get_target(exploit_sol).fitness
            fg = self.g_best.target.fitness
            # 2. Pick the candidate with the lower fitness
            if f1 < f2:
                best_candidate = explore_sol
                f_current   = f1
            else:
                best_candidate = exploit_sol
                f_current   = f2
            
            #NEW compare further (Elitism)
            # 3. If that candidate is better than the current global best, accept it;
            #    otherwise do a small random move around the global best
            if f_current < fg:
                pos_new = best_candidate
            else:
                pos_new = self.g_best.solution + beta * 1e-3 # because the beta has negative values and the solutions may have it also the over all direction changed


            

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)

class IFOX5(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.decay_rate = 1e-3  
        self.gravity_min = 0.5 * 9.81
        self.gravity_max = 9.81
        self.min_alpha = 1/(self.epoch*2)
    def evolve(self, epoch: int):
        pop_new = []
        # alpha = self.min_alpha + (1 - self.min_alpha) * (1 - epoch / (self.epoch))
        # slower decay: only hits min_alpha at final epoch
        
        alpha = 1 - (epoch/self.epoch)*(1-self.min_alpha)
        for idx in range(self.pop_size):
            beta = self.generator.uniform(-alpha, alpha, size=self.g_best.solution.size) #Inside is better
            t = self.generator.random(size=self.g_best.solution.size) - alpha #New

            # gravity = self.generator.uniform(self.gravity_min, self.gravity_max) #New
            jump = 0.5 * 9.81 * t ** 2
            # jump = max(jump, 1e-6) #New (TO ensure no zero)
            
            dis = 0.5 * self.g_best.solution
            
            explore_sol = dis * beta * jump                     #Exploration
            exploit_sol = self.g_best.solution + beta * alpha  #Exploitation

            # pick any improvement over current g_best
            # 1. Compute fitness of each candidate
            f1 = self.get_target(explore_sol).fitness
            f2 = self.get_target(exploit_sol).fitness
            fg = self.g_best.target.fitness
            # 2. Pick the candidate with the lower fitness
            if f1 < f2:
                best_candidate = explore_sol
                f_current   = f1
            else:
                best_candidate = exploit_sol
                f_current   = f2
            
            #NEW compare further (Elitism)
            # 3. If that candidate is better than the current global best, accept it;
            #    otherwise do a small random move around the global best
            if f_current < fg:
                pos_new = best_candidate
            else:
                pos_new = self.g_best.solution + beta * 1e-3 # because the beta has negative values and the solutions may have it also the over all direction changed

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)


class IFOX6(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.decay_rate = 1e-3  
        self.gravity_min = 0.5 * 9.81
        self.gravity_max = 9.81
        self.min_alpha = 1/(self.epoch*2)
    def evolve(self, epoch: int):
        pop_new = []
        # alpha = self.min_alpha + (1 - self.min_alpha) * (1 - epoch / (self.epoch))
        # slower decay: only hits min_alpha at final epoch
        
        alpha = 1 - (epoch/self.epoch)*(1-self.min_alpha)
        for idx in range(self.pop_size):
            beta = self.generator.uniform(-alpha, alpha, size=self.g_best.solution.size) #Inside is better
            t = self.generator.random(size=self.g_best.solution.size) #New

            # gravity = self.generator.uniform(self.gravity_min, self.gravity_max) #New
            jump = 0.5 * 9.81 * t ** 2
            # jump = max(jump, 1e-6) #New (TO ensure no zero)
            
            dis = 0.5 * self.g_best.solution
            
            explore_sol = dis * beta * jump                     #Exploration
            exploit_sol = self.g_best.solution + beta * alpha  #Exploitation

            # pick any improvement over current g_best
            # 1. Compute fitness of each candidate
            f1 = self.get_target(explore_sol).fitness
            f2 = self.get_target(exploit_sol).fitness
            fg = self.g_best.target.fitness
            # 2. Pick the candidate with the lower fitness
            if f1 < f2:
                best_candidate = explore_sol
                f_current   = f1
            else:
                best_candidate = exploit_sol
                f_current   = f2
            
            #NEW compare further (Elitism)
            # 3. If that candidate is better than the current global best, accept it;
            #    otherwise do a small random move around the global best
            if f_current < fg:
                pos_new = best_candidate
            else:
                pos_new = self.g_best.solution + beta # because the beta has negative values and the solutions may have it also the over all direction changed

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)

class IFOX7(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.decay_rate = 1e-3  
        self.gravity_min = 0.5 * 9.81
        self.gravity_max = 9.81
        self.min_alpha = 1/(self.epoch*2)
    def evolve(self, epoch: int):
        pop_new = []
        # alpha = self.min_alpha + (1 - self.min_alpha) * (1 - epoch / (self.epoch))
        # slower decay: only hits min_alpha at final epoch
        
        alpha = 1 - (epoch/self.epoch)*(1-self.min_alpha)
        for idx in range(self.pop_size):
            beta = self.generator.uniform(-alpha, alpha, size=self.g_best.solution.size) #Inside is better
            t = self.generator.random(size=self.g_best.solution.size) #New

            # gravity = self.generator.uniform(self.gravity_min, self.gravity_max) #New
            jump = 0.5 * 9.81 * t ** 4 #New
            # jump = max(jump, 1e-6) #New (TO ensure no zero)
            
            dis = 0.5 * self.g_best.solution
            
            explore_sol = dis * beta * jump                     #Exploration
            exploit_sol = self.g_best.solution + beta * alpha  #Exploitation

            # pick any improvement over current g_best
            # 1. Compute fitness of each candidate
            f1 = self.get_target(explore_sol).fitness
            f2 = self.get_target(exploit_sol).fitness
            # fg = self.g_best.target.fitness
            # 2. Pick the candidate with the lower fitness
            if f1 < f2:
                pos_new = explore_sol
                f_current   = f1
            else:
                pos_new = exploit_sol
                f_current   = f2
            
            # #NEW compare further (Elitism)
            # # 3. If that candidate is better than the current global best, accept it;
            # #    otherwise do a small random move around the global best
            # if f_current < fg:
            #     pos_new = best_candidate
            # else:
            #     pos_new = self.g_best.solution + beta # because the beta has negative values and the solutions may have it also the over all direction changed

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)


class IFOX8(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.half_G = 0.5 * 9.81
        self.min_alpha = 1/(self.epoch*2)
    
    def evolve(self, epoch: int):
        pop_new = []

        #OLD ALPHA
        #alpha = self.min_alpha + (1 - self.min_alpha) * (1 - epoch / (self.epoch))

        alpha = 1 - (epoch/self.epoch)*(1-self.min_alpha)
        for idx in range(self.pop_size):
            beta = self.generator.uniform(-0.5*alpha, alpha, size=self.g_best.solution.size) #Inside is better
            t = self.generator.random(size=self.g_best.solution.size) #New

            # gravity = self.generator.uniform(self.gravity_min, self.gravity_max) #New
            jump = self.half_G * t ** 2 
            dis = 0.5 * self.g_best.solution
            
            explore_sol = dis * beta * jump                     #Exploration
            exploit_sol = self.g_best.solution * beta * alpha  #Exploitation

            # pick any improvement over current g_best
            # 1. Compute fitness of each candidate
            f1 = self.get_target(explore_sol).fitness
            f2 = self.get_target(exploit_sol).fitness
            fg = self.g_best.target.fitness
            # 2. Pick the candidate with the lower fitness
            if f1 < f2:
                best_candidate = explore_sol
                f_current   = f1
            else:
                best_candidate = exploit_sol
                f_current   = f2
            
            #NEW compare further (Elitism)
            # 3. If that candidate is better than the current global best, accept it;
            #    otherwise do a small random move around the global best
            if f_current < fg:
                pos_new = best_candidate
            else:
                pos_new = self.g_best.solution + self.generator.random(size=self.g_best.solution.size) # because the beta has negative values and the solutions may have it also the over all direction changed

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)


import math
class IFOX9(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, mu=3.8, p_levy=0.2, **kwargs):
        super().__init__(**kwargs)
        self.epoch      = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size   = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mu         = mu                  # chaotic map parameter
        self.p_levy     = p_levy              # prob of Levy exploration
        self.alpha_max  = 1.0
        self.alpha_min  = 1.0/(2*self.epoch)  # small floor
        self.alpha      = self.alpha_max      # will be updated chaotically
        self.set_parameters(["epoch","pop_size","mu","p_levy"])

    def levy_flight(self, size, beta=1.5):
        # compute sigma for Levy
        num = math.gamma(1+beta) * math.sin(math.pi*beta/2)
        den = beta * math.gamma((1+beta)/2) * 2**((beta-1)/2)
        sigma = (num/den)**(1/beta)
        u = self.generator.normal(0, sigma, size)
        v = self.generator.normal(0, 1.0, size)
        return u / np.abs(v)**(1/beta)

    def evolve(self, epoch):
        pop_solutions = [agent.solution for agent in self.pop]
        # Update chaotic alpha
        decay = 1 - epoch/self.epoch
        self.alpha = self.alpha_min + (self.alpha_max-self.alpha_min)*decay
        self.alpha = self.mu * self.alpha * (1 - self.alpha)  # logistic map

        new_pop = []
        for x in pop_solutions:
            dim = x.size
            if self.generator.random() < self.p_levy:
                # Lévy-flight exploration around g_best
                step = self.levy_flight(dim)
                candidate = self.g_best.solution + self.alpha * step * (self.g_best.solution - x)
            else:
                # Gaussian exploitation around current
                candidate = x + self.generator.normal(0, 1, dim) * self.alpha

            # Opposition-based probe for g_best occasionally
            if self.generator.random() < 0.1:
                candidate = self.problem.lb + self.problem.ub - self.g_best.solution

            # # boundary check
            # candidate = np.clip(candidate, self.problem.lb, self.problem.ub)

            pos_new = candidate
            # Compare with global best (elitism)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            new_pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[x] = agent

        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(new_pop)

class IFOX10(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, mu=3.8, p_levy=0.2, **kwargs):
        super().__init__(**kwargs)
        self.epoch      = self.validator.check_int("epoch",    epoch,    [1, 100000])
        self.pop_size   = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mu         = mu                  # chaotic map parameter
        self.p_levy     = p_levy              # prob of injecting Lévy-flight
        self.alpha_max  = 1.0
        self.alpha_min  = 1.0/(2*self.epoch)  # floor for alpha
        self.alpha      = self.alpha_max
        self.half_G     = 0.5 * 9.81          # gravity constant
        self.set_parameters(["epoch","pop_size","mu","p_levy"])

    def levy_flight(self, size, beta=1.5):
        num   = math.gamma(1+beta)*math.sin(math.pi*beta/2)
        den   = beta*math.gamma((1+beta)/2)*2**((beta-1)/2)
        sigma = (num/den)**(1/beta)
        u = self.generator.normal(0, sigma, size)
        v = self.generator.normal(0, 1,     size)
        return u / np.abs(v)**(1/beta)

    def evolve(self, epoch):
        dim = self.g_best.solution.size

        # 1) update α with a logistic (chaotic) map
        decay      = 1 - epoch/self.epoch
        self.alpha = self.alpha_min + (self.alpha_max-self.alpha_min)*decay
        self.alpha = self.mu * self.alpha * (1 - self.alpha)

        pop_new = []
        for agent in self.pop:
            x = agent.solution

            # 2) draw β either by uniform (original IFOX) or Lévy
            if self.generator.random() < self.p_levy:
                beta = self.levy_flight(dim) * self.alpha
            else:
                beta = self.generator.uniform(-0.5*self.alpha, self.alpha, size=dim)

            # 3) compute original IFOX jump and dis
            t     = self.generator.random(size=dim)
            jump  = self.half_G * t**2
            dis   = 0.5 * self.g_best.solution

            # 4) original IFOX formulas
            explore = dis * beta * jump
            exploit = self.g_best.solution + beta * self.alpha

            # 5) pick the better of the two
            f1 = self.get_target(explore).fitness
            f2 = self.get_target(exploit).fitness
            if f1 < f2:
                cand, f_cand = explore, f1
            else:
                cand, f_cand = exploit, f2

            # 6) elitism vs g_best
            if f_cand < self.g_best.target.fitness:
                pos_new = cand
            else:
                # small random move around g_best to escape stagnation
                pos_new = self.g_best.solution + self.generator.random(size=dim)

            # 7) occasionally try an opposition move on g_best
            if self.generator.random() < 0.05:
                pos_new = self.problem.lb + self.problem.ub - self.g_best.solution


                    # Compare with global best (elitism)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[x] = agent

        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)



class IFOX11(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, mu=3.8, p_levy=0.2, **kwargs):
        super().__init__(**kwargs)
        self.epoch      = self.validator.check_int("epoch",    epoch,    [1, 100000])
        self.pop_size   = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mu         = mu                  # chaotic map parameter
        self.p_levy     = p_levy              # prob of injecting Lévy-flight
        self.alpha_max  = 1.0
        self.alpha_min  = 1.0/(2*self.epoch)  # floor for alpha
        self.alpha      = self.alpha_max
        self.half_G     = 0.5 * 9.81          # gravity constant
        self.set_parameters(["epoch","pop_size","mu","p_levy"])

    def levy_flight(self, size, beta=1.5):
        num   = math.gamma(1+beta)*math.sin(math.pi*beta/2)
        den   = beta*math.gamma((1+beta)/2)*2**((beta-1)/2)
        sigma = (num/den)**(1/beta)
        u = self.generator.normal(0, sigma, size)
        v = self.generator.normal(0, 1,     size)
        return u / np.abs(v)**(1/beta)

    def evolve(self, epoch):
        dim = self.g_best.solution.size

        # 1) update α with a logistic (chaotic) map
        decay      = 1 - epoch/self.epoch
        self.alpha = self.alpha_min + (self.alpha_max-self.alpha_min)*decay
        self.alpha = self.mu * self.alpha * (1 - self.alpha)

        pop_new = []
        for agent in self.pop:
            x = agent.solution

            # 2) draw β either by uniform (original IFOX) or Lévy
            if self.generator.random() < self.p_levy:
                beta = self.levy_flight(dim) * self.alpha
            else:
                beta = self.generator.uniform(-self.alpha, self.alpha, size=dim)

            # 3) compute original IFOX jump and dis
            t     = self.generator.random(size=dim)
            jump  = self.half_G * t**2
            dis   = 0.5 * self.g_best.solution

            # 4) original IFOX formulas
            explore = dis * beta * jump
            exploit = self.g_best.solution + beta * self.alpha

            # 5) pick the better of the two
            f1 = self.get_target(explore).fitness
            f2 = self.get_target(exploit).fitness
            if f1 < f2:
                cand, f_cand = explore, f1
            else:
                cand, f_cand = exploit, f2

            # 6) elitism vs g_best
            if f_cand < self.g_best.target.fitness:
                pos_new = cand
            else:
                # small random move around g_best to escape stagnation
                pos_new = self.g_best.solution + self.generator.random(size=dim)

            # 7) occasionally try an opposition move on g_best
            if self.generator.random() < 0.01:
                pos_new = self.problem.lb + self.problem.ub - self.g_best.solution


                    # Compare with global best (elitism)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[x] = agent

        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)
            

class IFOX12(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, mu=3.8, p_levy=0.2, **kwargs):
        super().__init__(**kwargs)
        self.epoch      = self.validator.check_int("epoch",    epoch,    [1, 100000])
        self.pop_size   = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mu         = mu                  # chaotic map parameter
        self.p_levy     = p_levy              # prob of injecting Lévy-flight
        self.alpha_min  = 1.0/(2*self.epoch)  # floor for alpha
        self.half_G     = 0.5 * 9.81          # gravity constant
        self.set_parameters(["epoch","pop_size","mu","p_levy"])

    def levy_flight(self, size, beta=1.5):
        num   = math.gamma(1+beta)*math.sin(math.pi*beta/2)
        den   = beta*math.gamma((1+beta)/2)*2**((beta-1)/2)
        sigma = (num/den)**(1/beta)
        u = self.generator.normal(0, sigma, size)
        v = self.generator.normal(0, 1,     size)
        return u / np.abs(v)**(1/beta)

    def evolve(self, epoch):
        dim = self.g_best.solution.size

        # 1) update α with a logistic (chaotic) map
        self.alpha = self.alpha_min + (1 - self.alpha_min)* (1 - epoch/self.epoch)
        self.alpha = self.mu * self.alpha * (1 - self.alpha)

        pop_new = []
        for agent in self.pop:
            x = agent.solution

            # 2) draw β either by Alpha or Lévy
            if self.generator.random() < self.alpha:
                beta = self.levy_flight(dim) * self.alpha
            else:
                beta = self.generator.uniform(-self.alpha, self.alpha, size=dim)

            # 3) compute original IFOX jump and dis
            t     = self.generator.random(size=dim)
            jump  = self.half_G * t**2
            dis   = 0.5 * self.g_best.solution

            # 4) original IFOX formulas
            explore = dis * beta * jump
            exploit = self.g_best.solution + beta * self.alpha

            # 5) pick the better of the two
            f1 = self.get_target(explore).fitness
            f2 = self.get_target(exploit).fitness
            if f1 < f2:
                cand, f_cand = explore, f1
            else:
                cand, f_cand = exploit, f2

            # 6) elitism vs g_best
            if f_cand < self.g_best.target.fitness:
                pos_new = cand
            else:
                # small random move around g_best to escape stagnation
                pos_new = self.g_best.solution + self.generator.random(size=dim)

            # 7) occasionally try an opposition move on g_best
            if self.generator.random() < min(beta):
                pos_new = self.problem.lb + self.problem.ub - self.g_best.solution


                    # Compare with global best (elitism)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[x] = agent

        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)



class IFOX13(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, mu=3.8, p_levy=0.2, **kwargs):
        super().__init__(**kwargs)
        self.epoch      = self.validator.check_int("epoch",    epoch,    [1, 100000])
        self.pop_size   = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mu         = mu                  # chaotic map parameter
        self.p_levy     = p_levy              # prob of injecting Lévy-flight
        self.alpha_min  = 1.0/(0.5*self.epoch)  # floor for alpha
        self.half_G     = 0.5 * 9.81          # gravity constant
        self.set_parameters(["epoch","pop_size","mu","p_levy"])
        
    def levy_flight(self, size, beta=1.5):
        num   = math.gamma(1+beta)*math.sin(math.pi*beta/2)
        den   = beta*math.gamma((1+beta)/2)*2**((beta-1)/2)
        sigma = (num/den)**(1/beta)
        u = self.generator.normal(0, sigma, size)
        v = self.generator.normal(0, 1,     size)
        return u / np.abs(v)**(1/beta)

    def evolve(self, epoch):
        dim = self.g_best.solution.size

        # 1) update α with a logistic (chaotic) map
        self.alpha = self.alpha_min + (1 - self.alpha_min)* (1 - epoch/self.epoch)
  

        pop_new = []
        for idx,agent in enumerate(self.pop):

            # 2) draw β either by Alpha or Lévy
            if self.generator.random() < self.alpha:
                beta = self.levy_flight(dim) * self.alpha
            else:
                beta = self.generator.uniform(-self.alpha, self.alpha, size=dim)

            # 3) compute original IFOX jump and dis
            t     = self.generator.random(size=dim)
            jump  = self.half_G * t**2
            dis   = 0.5 * self.g_best.solution

            # 4) original IFOX formulas
            explore = dis * beta * jump
            exploit = self.g_best.solution + beta * self.alpha

            # 5) pick the better of the two
            f1 = self.get_target(explore).fitness
            f2 = self.get_target(exploit).fitness
            if f1 < f2:
                cand, f_cand = explore, f1
            else:
                cand, f_cand = exploit, f2

            # 6) elitism vs g_best
            if f_cand < self.g_best.target.fitness:
                pos_new = cand
            else:
                # small random move around g_best to escape stagnation
                pos_new = self.g_best.solution + self.generator.random(size=dim)

            # 7) occasionally try an opposition move on g_best
            if self.generator.random() < min(beta):
                pos_new = self.problem.lb + self.problem.ub - self.g_best.solution


                    # Compare with global best (elitism)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent

        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)