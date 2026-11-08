# Project 7: Self-Driving Lab Simulator — Problem Formulation (1 Page)

**Student Project — Reinforcement Learning (Unit 5) | Algorithm: Tabular Q-Learning**

## 1. Problem
Finding the best temperature and concentration for a chemical reaction normally
needs many slow, expensive lab experiments. A *self-driving lab* lets an RL agent
choose the next experiment itself, observe the yield, and improve its choices —
exactly the loop used in modern automated chemistry and materials labs.

## 2. Environment
A simulated reaction on a discrete grid (file: `lab_env.py`).
Temperature ∈ {20,30,…,100} °C (9 values) × Concentration ∈ {0.2,0.5,…,2.0} mol/L
(7 values) = **63 possible settings**. The hidden true yield is a Gaussian peak
(95% max) centred at 65 °C / 1.15 M, minus a side-reaction penalty above 85 °C.
Each measurement adds Gaussian noise (σ = 1.5%) so the agent cannot just memorise
one lucky reading. The agent never sees this formula — it only sees measured yield.

## 3. MDP Formulation
| Element | Definition |
|---|---|
| **State** `s` | Current setting `(temp_index, conc_index)` — 63 states |
| **Actions** `a` | 0: Temp +10 °C · 1: Temp −10 °C · 2: Conc +0.3 M · 3: Conc −0.3 M |
| **Reward** `r` | `measured_yield/100 − 0.01` (high yield good; −0.01 = cost per experiment) |
| **Episode** | Random starting setting, 30 experiments, then reset |
| **Goal** | Maximise total return ⇒ reach and stay at the highest-yield setting fast |

## 4. Method
Tabular **Q-learning** (Watkins, 1989): α = 0.1, γ = 0.95, ε-greedy exploration
decaying 1.0 → 0.05 (×0.995/episode), 800 episodes, seed 42.
Update: `Q(s,a) ← Q(s,a) + α [ r + γ max Q(s′,a′) − Q(s,a) ]`.
Compared against **(a) Random** agent and **(b) Heuristic** hill-climbing chemist
that greedily moves to the best neighbouring setting every step.

## 5. Results (seed 42, reproducible)
| Method | Avg return (last 50 eps) | Avg best yield / episode |
|---|---|---|
| **Q-Learning** | **24.05** | **91.9%** (measured, incl. noise) |
| Random baseline | 4.02 | 58.6% |
| Heuristic hill-climbing | 24.84 | 90.1% |
| True grid optimum (hidden) | — | 89.0% true yield at **60 °C, 1.1 M** |

The greedy learned policy finishes at **60 °C / 1.1 M** — exactly the true grid
optimum (see `results/yield_heatmap.png`, red X on the star). On the learning
curve (`results/learning_curve.png`) Q-learning climbs from ≈6 to ≈24 return,
while Random stays flat at ≈4. The heuristic starts high because it "probes"
all 4 neighbours every step (4 trial evaluations per move — expensive in a real
lab); Q-learning matches it by the end with a reusable policy that needs no
probing. Measured yields slightly exceed the 89.0% true optimum because of the
+noise on individual readings; averages over 50 episodes remove that effect.

## 6. Limitations & Ethics
*Limitations:* discrete 2-variable grid only; real reactions have continuous
controls, more reactants, delays and safety constraints; tabular Q-learning
will not scale to many variables (DQN/PPO would be needed).
*Ethics/Safety:* an autonomous lab must have hard safety limits (max temp/
pressure, toxic reagents) outside the reward function — an agent maximising
yield alone could discover dangerous conditions. Human approval should remain
required before scaling any agent-found recipe to production.

**Files:** `lab_env.py` (environment) · `train.py` (agent + baselines + plots) ·
`results/learning_curve.png` · `results/yield_heatmap.png` · `results/results_summary.csv`
