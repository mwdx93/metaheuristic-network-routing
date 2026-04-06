# 🚀 Metaheuristic Network Routing Optimization

A Python-based framework for solving the packet routing problem in computer networks using metaheuristic optimization algorithms. The problem is formulated as a multi-objective shortest path problem considering cost, delay, and bandwidth under QoS constraints.

---

## 📌 Features

- Multi-objective routing optimization (cost, delay, bandwidth)  
- Supports multiple network topologies:
  - Grid  
  - Random  
  - Scale-free  
- Implements and compares 10 metaheuristic algorithms:
  - GWO, ABC, DO, BA, PSO  
  - FOX, IFOX, LSHADE  
  - GA, BBO  
- Constraint handling (max delay, min bandwidth)  
- Statistical evaluation (multi-trial experiments)  
- Visualization:
  - Network graphs  
  - Delay & bandwidth matrices  
  - Convergence plots  
  - Ranking analysis  

---

## 🧠 Problem Formulation

The routing problem is modeled as a graph:

- Nodes represent routers  
- Edges represent communication links with:
  - Delay  
  - Bandwidth  
  - Cost  

### Objective Function

A weighted fitness function is used:

f = w1 * (Norm_Cost)^2 + w2 * (Norm_Delay)^2 + w3 * (Norm_BW)^2

Where:
- w1 = 0.4, w2 = 0.3, w3 = 0.3  
- QoS constraints:
  - Maximum delay  
  - Minimum bandwidth  

---

## ⚙️ Installation

git clone https://github.com/your-username/metaheuristic-network-routing.git  
cd metaheuristic-network-routing  
pip install -r requirements.txt  

---

## ▶️ Usage

Run the main experiment:

python main.py

### Configuration (inside script)

NODES = 100  
topology = "scale_free"  
epoch = 500  
n_agents = 50  
n_trials = 30  

---

## 📊 Output

results/  
├── best_fit/  
├── convergence/  
├── Analysis/  
│   ├── TABLES/  
│   ├── PLOTS/  

---

## 📈 Key Findings

- LSHADE consistently achieved the best performance  
- Metaheuristic methods outperform traditional routing approaches  
- The proposed weighted objective improves QoS-aware routing  

---

## 📚 Citation

If you use this work, please cite:

[1] Jumaah, M. A., Ali, Y. H., Rashid, T. A., & Vimal, S. (2024). Foxann: A method for boosting neural network performance. Journal of Soft Computing and Computer Applications, 1(1), 1001.

[2] Jumaah, M. A., Shihab, A. I., & Farhan, A. A. (2020). Epileptic seizures detection using DCT-II and KNN classifier in long-term EEG signals. Iraqi Journal of Science, 2687–2694.

[3] Jumaah, M. A., Ali, Y. H., & Rashid, T. A. (2025). Efficient Q-learning hyperparameter tuning using FOX optimization algorithm. Results in Engineering, 25, 104341.

[4] Jumaah, M. A., Ali, Y. H., & Rashid, T. A. (2025). Artificial Liver Classifier: A New Alternative to Conventional Machine Learning Models. Frontiers in Artificial Intelligence, 8.

[5] Abbod, A. A., Hassan, A. K. A., & Jumaah, M. A. (2025). Analyzing user behavior for targeted commercial advertisements using Apriori and K-means algorithms. AIP Conference Proceedings, 3169(1).

[6] Jumaah, M. A., Ali, Y. H., & Rashid, T. A. (2025). An improved FOX optimization algorithm using adaptive exploration and exploitation for global optimization. PLOS ONE, 20(9), e0331965.

[7] Jumaah, M. A., Ali, Y. H., & Rashid, T. A. (2024). Q-FOX learning: breaking tradition in reinforcement learning. arXiv.

[8] Jumaah, M. A. (2024). QF-tuner: Breaking Tradition in Reinforcement Learning. arXiv.

---

## 📄 License

MIT License
