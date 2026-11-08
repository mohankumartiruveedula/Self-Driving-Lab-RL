# Project 7: Self-Driving Lab — 5-Slide Presentation

> Present from `slides.pptx` (generated). This file is your speaker script.
> Total speaking time: ~5 minutes (1 minute per slide).

---

## Slide 1 — Motivation: Why a Self-Driving Lab?
- Finding the best reaction conditions = hundreds of manual experiments.
- Each experiment costs time, chemicals and money.
- Real self-driving labs (e.g. automated chemistry/materials platforms) let an
  algorithm pick the next experiment.
- **Our question:** can a simple RL agent learn the best temperature +
  concentration by itself, faster than random trial-and-error?

*Speaker note: "Chemists normally guess, test, adjust. We replaced the guessing with Q-learning."*

## Slide 2 — Method: The MDP + Q-Learning
- **State:** current (temperature, concentration) — 63 grid settings.
- **Actions:** Temp ±10 °C, Concentration ±0.3 M (4 actions).
- **Reward:** measured yield/100 − 0.01 per experiment.
- **Hidden simulator:** yield peaks at 65 °C / 1.15 M, penalty above 85 °C, + noise.
- **Algorithm:** tabular Q-learning, α=0.1, γ=0.95, ε: 1.0→0.05, 800 episodes.
- Baselines: Random agent + greedy hill-climbing "chemist".

*Show: `lab_env.py` grid diagram / code snippet if asked.*

## Slide 3 — Results: Learning Curves
- **Show `results/learning_curve.png`**
- Q-learning: return climbs ≈6 → ≈24 over 800 episodes.
- Random: flat at ≈4 — never learns.
- Heuristic: high from the start (≈25) but does not learn; it probes all
  4 neighbours every step, which would cost 4× experiments in a real lab.
- Q-learning catches up and learns a reusable policy.

## Slide 4 — Results: Did It Find the Optimum?
- **Show `results/yield_heatmap.png`** (star = true optimum, X = agent)
- Agent's favourite final setting: **60 °C, 1.1 M = exact grid optimum**.
- Best yield: Q-learning 91.9% vs Random 58.6% vs Heuristic 90.1% (measured).
- True optimum is 89.0%; measured values vary ±1.5% due to lab noise.
- Conclusion: agent discovered the peak without ever seeing the formula.

## Slide 5 — Limitations & Ethics + Conclusion
- **Limitations:** only 2 discrete variables; tabular Q-learning won't scale
  to continuous/high-dimensional labs — next step DQN/PPO + more reactants.
- **Ethics/Safety:** yield-only reward could push toward unsafe temperatures;
  real labs need hard safety limits + human approval before scale-up.
- **Conclusion:** even the simplest RL (a 63×4 Q-table) turns random search
  into directed discovery — the core idea behind real self-driving labs.
- Thank you — questions welcome.

*Backup answer if asked "why not deep learning?": state space is only 63, so a Q-table is exact, faster and fully interpretable — deep RL would be overkill here.*
