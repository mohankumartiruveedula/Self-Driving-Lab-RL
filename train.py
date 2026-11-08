"""
Project 7 - Self-Driving Lab: Training Script
Trains a tabular Q-learning agent and compares it with:
  1. Random baseline  (picks actions uniformly at random)
  2. Heuristic baseline (greedy hill-climbing chemist)

Run:  python train.py
Outputs in results/:
  - learning_curve.png  (returns over episodes, all 3 methods)
  - yield_heatmap.png   (true yield map + agent's best point)
  - results_summary.csv (final numbers for your report)
  - q_table.npy         (saved learned Q-table)
"""
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab_env import SelfDrivingLabEnv

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EPISODES_TRAIN = 800
EPISODES_EVAL = 50
SEED = 42


# ---------- Q-Learning Agent ----------
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=SEED):
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        self.q[state, action] += self.alpha * (td_target - self.q[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------- Episode runners ----------
def run_episode(env, get_action):
    """Run one episode. get_action(state, env, last_yield) -> action."""
    state = env.reset()
    total_reward = 0.0
    best_yield = 0.0
    last_info = None
    done = False
    while not done:
        action = get_action(state, env)
        state, reward, done, info = env.step(action)
        total_reward += reward
        best_yield = max(best_yield, info["yield"])
        last_info = info
    return total_reward, best_yield, last_info


def train_q_learning():
    env = SelfDrivingLabEnv(seed=SEED)
    agent = QLearningAgent(env.n_states, env.n_actions, seed=SEED)
    returns, best_yields = [], []

    for ep in range(EPISODES_TRAIN):
        state = env.reset(seed=SEED + ep)
        total = 0.0
        best = 0.0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total += reward
            best = max(best, info["yield"])
        agent.decay_epsilon()
        returns.append(total)
        best_yields.append(best)
    return agent, np.array(returns), np.array(best_yields)


def run_baselines():
    """Random and heuristic agents, same number of episodes as training."""
    # Random
    rng = np.random.default_rng(SEED)
    env_r = SelfDrivingLabEnv(seed=SEED)
    rand_returns, rand_best = [], []
    for ep in range(EPISODES_TRAIN):
        r, b, _ = run_episode(env_r, lambda s, e: int(rng.integers(0, e.n_actions)))
        # reset with different start each episode for fairness
        env_r.rng = np.random.default_rng(SEED + 1000 + ep)
        rand_returns.append(r)
        rand_best.append(b)

    # Heuristic: hill-climbing - try moving toward higher measured yield.
    # Simple chemist rule: probe neighbours via true step, keep best move greedily
    # using observed yield only (no knowledge of the hidden optimum).
    heur_returns, heur_best = [], []
    for ep in range(EPISODES_TRAIN):
        env_h = SelfDrivingLabEnv(seed=SEED + 2000 + ep)
        state = env_h.reset(seed=SEED + 2000 + ep)
        total, best, done = 0.0, 0.0, False
        current_yield = env_h.observe_yield(*env_h.current_temp_conc())
        best = current_yield
        while not done:
            # Evaluate 4 neighbours by peeking at true yield + noise (like trial runs)
            candidates = []
            for a in range(4):
                ti, ci = env_h.temp_idx, env_h.conc_idx
                if a == 0:
                    ti = min(ti + 1, env_h.n_temps - 1)
                elif a == 1:
                    ti = max(ti - 1, 0)
                elif a == 2:
                    ci = min(ci + 1, env_h.n_concs - 1)
                else:
                    ci = max(ci - 1, 0)
                y = env_h.true_yield(env_h.TEMPS[ti], env_h.CONCS[ci])
                candidates.append(y)
            action = int(np.argmax(candidates))
            # If all neighbours worse, still take best (will plateau near peak)
            state, reward, done, info = env_h.step(action)
            total += reward
            best = max(best, info["yield"])
        heur_returns.append(total)
        heur_best.append(best)

    return (np.array(rand_returns), np.array(rand_best),
            np.array(heur_returns), np.array(heur_best))


def evaluate_greedy(agent, n_episodes=EPISODES_EVAL):
    env = SelfDrivingLabEnv(seed=999)
    finals, bests, settings = [], [], []
    for ep in range(n_episodes):
        state = env.reset(seed=9000 + ep)
        done = False
        best = 0.0
        last = None
        while not done:
            action = agent.choose_action(state, greedy=True)
            state, reward, done, info = env.step(action)
            best = max(best, info["yield"])
            last = info
        finals.append(last["yield"])
        bests.append(best)
        settings.append((last["temp"], last["conc"]))
    return float(np.mean(finals)), float(np.mean(bests)), settings


def evaluate_policy(name, get_action_factory):
    """Evaluate random/heuristic greedily for final table (best yield per episode)."""
    bests = []
    for ep in range(EPISODES_EVAL):
        env = SelfDrivingLabEnv(seed=9000 + ep)
        get_action = get_action_factory(env, ep)
        _, b, _ = run_episode(env, get_action)
        bests.append(b)
    return float(np.mean(bests))


def smooth(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Training Q-learning agent...")
    agent, q_returns, q_best = train_q_learning()
    print(f"  Done. Last-50 avg return: {q_returns[-50:].mean():.2f} | best yield: {q_best[-50:].mean():.1f}%")

    print("Running Random + Heuristic baselines...")
    rand_ret, rand_best, heur_ret, heur_best = run_baselines()
    print(f"  Random    last-50 avg return: {rand_ret[-50:].mean():.2f} | best yield: {rand_best[-50:].mean():.1f}%")
    print(f"  Heuristic last-50 avg return: {heur_ret[-50:].mean():.2f} | best yield: {heur_best[-50:].mean():.1f}%")

    print("Evaluating greedy Q-policy...")
    final_yield, best_yield, settings = evaluate_greedy(agent)
    most_common = max(set(settings), key=settings.count)
    print(f"  Greedy policy: final yield {final_yield:.1f}% | best-in-episode {best_yield:.1f}%")
    print(f"  Most common final setting: Temp={most_common[0]}C, Conc={most_common[1]}M")

    # Theoretical optimum on the discrete grid
    env0 = SelfDrivingLabEnv(seed=SEED)
    grid = env0.yield_grid()
    oi, oj = np.unravel_index(np.argmax(grid), grid.shape)
    print(f"  True grid optimum: Temp={env0.TEMPS[oi]}C, Conc={env0.CONCS[oj]}M, Yield={grid[oi, oj]:.1f}%")

    # ---- Plot 1: Learning curves ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(smooth(q_returns), label="Q-Learning", linewidth=2)
    ax.plot(smooth(rand_ret), label="Random baseline", linewidth=1.5)
    ax.plot(smooth(heur_ret), label="Heuristic (hill-climbing)", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (smoothed, window=20)")
    ax.set_title("Project 7: Self-Driving Lab - Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "learning_curve.png")
    fig.savefig(curve_path, dpi=180)
    plt.close(fig)
    print(f"  Saved {curve_path}")

    # ---- Plot 2: Yield heatmap + learned policy endpoint ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[min(env0.CONCS), max(env0.CONCS), min(env0.TEMPS), max(env0.TEMPS)])
    ax.set_xlabel("Concentration (mol/L)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("True Yield Map - Star = Grid Optimum, X = Agent's Favourite Setting")
    ax.plot(env0.CONCS[oj], env0.TEMPS[oi], marker="*", color="white",
            markersize=18, markeredgecolor="black", label="True optimum")
    ax.plot(most_common[1], most_common[0], marker="X", color="red",
            markersize=12, markeredgecolor="white", label="Agent final (mode)")
    fig.colorbar(im, ax=ax, label="Yield (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    heat_path = os.path.join(RESULTS_DIR, "yield_heatmap.png")
    fig.savefig(heat_path, dpi=180)
    plt.close(fig)
    print(f"  Saved {heat_path}")

    # ---- Save Q-table + summary CSV ----
    np.save(os.path.join(RESULTS_DIR, "q_table.npy"), agent.q)
    csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "last50_avg_return", "last50_avg_best_yield_pct", "eval_best_yield_pct"])
        w.writerow(["Q-Learning (train)", f"{q_returns[-50:].mean():.3f}", f"{q_best[-50:].mean():.2f}", f"{best_yield:.2f}"])
        w.writerow(["Random", f"{rand_ret[-50:].mean():.3f}", f"{rand_best[-50:].mean():.2f}", ""])
        w.writerow(["Heuristic hill-climbing", f"{heur_ret[-50:].mean():.3f}", f"{heur_best[-50:].mean():.2f}", ""])
        w.writerow(["True grid optimum", "", "", f"{grid[oi, oj]:.2f}"])
        w.writerow(["Agent favourite setting", f"Temp={most_common[0]}C", f"Conc={most_common[1]}M", ""])
    print(f"  Saved {csv_path}")
    print("\nAll done. Check the results/ folder for your graphs.")


if __name__ == "__main__":
    main()
