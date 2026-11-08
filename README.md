# Project 7 — Self-Driving Lab Simulator (Reinforcement Learning)

A toy self-driving lab: a Q-learning agent learns to choose **temperature**
and **concentration** to maximise the yield of a simulated chemical reaction.
Kept deliberately simple for a university project — no GPU, runs in seconds.

## Quick Start (Windows)

```powershell
cd Project7_SelfDrivingLab
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe make_slides.py
.venv\Scripts\python.exe -m pytest tests/ -v
```

First-time setup only (if `.venv` is missing):

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe numpy matplotlib pytest python-pptx
```

## What you get

| File | What it is |
|---|---|
| `lab_env.py` | The simulated lab (state/action/reward defined here) |
| `train.py` | Q-learning agent + Random & Heuristic baselines + plots |
| `results/learning_curve.png` | Learning curves — main result graph |
| `results/yield_heatmap.png` | Yield map; agent found the exact optimum (60C, 1.1M) |
| `results/results_summary.csv` | Numbers for your report table |
| `REPORT.md` | 1-page problem formulation (state/action/reward/environment) |
| `PRESENTATION.md` | 5-slide script + speaker notes |
| `slides.pptx` | Ready-to-present 5-slide deck |
| `tests/test_env.py` | 4 unit tests for the environment |

## Headline Results (seed 42)

- Q-Learning: avg return **24.05**, best yield **91.9%**
- Random baseline: **4.02**, **58.6%**
- Heuristic hill-climbing: **24.84**, **90.1%**
- True optimum: **89.0% at 60C / 1.1M** — agent's favourite setting matches it exactly.

Present `REPORT.md` as your 1-page formulation and `slides.pptx` as your deck.
See `PRESENTATION.md` for what to say on each slide.
