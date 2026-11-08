"""
Self-Driving Lab Simulator - Project 7
A toy chemistry lab where an RL agent chooses temperature and
concentration to maximise the yield of a simulated reaction.

"""
import numpy as np


class SelfDrivingLabEnv:
    """
    State:  (temp_index, conc_index)
    Action: 0 = Temp +10C, 1 = Temp -10C, 2 = Conc +0.3M, 3 = Conc -0.3M
    Reward: yield/100 - 0.01  (high yield is good, each experiment costs a little)
    Episode: starts at a random setting, ends after max_steps
    """

    # Grid definition
    TEMPS = [20, 30, 40, 50, 60, 70, 80, 90, 100]          # degrees C
    CONCS = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]           # mol/L

    # True (hidden) optimum - agent does not know this
    OPT_TEMP = 65.0
    OPT_CONC = 1.15

    def __init__(self, max_steps=30, noise_std=1.5, seed=None):
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.n_temps = len(self.TEMPS)
        self.n_concs = len(self.CONCS)
        self.n_states = self.n_temps * self.n_concs
        self.n_actions = 4
        self.action_names = ["Temp +10", "Temp -10", "Conc +0.3", "Conc -0.3"]
        self.temp_idx = 0
        self.conc_idx = 0
        self.steps = 0

    # ---- helpers ----
    def state_id(self, temp_idx=None, conc_idx=None):
        if temp_idx is None:
            temp_idx = self.temp_idx
        if conc_idx is None:
            conc_idx = self.conc_idx
        return temp_idx * self.n_concs + conc_idx

    def true_yield(self, temp, conc):
        """Hidden reaction model. Returns yield in % (0-100) without noise."""
        # Main reaction: Gaussian peak at the optimum
        temp_term = ((temp - self.OPT_TEMP) ** 2) / (2 * 15.0 ** 2)
        conc_term = ((conc - self.OPT_CONC) ** 2) / (2 * 0.35 ** 2)
        y = 95.0 * np.exp(-temp_term - conc_term)
        # Side reaction above 85C wastes product
        if temp > 85:
            y -= (temp - 85) * 0.8
        return float(np.clip(y, 0, 100))

    def observe_yield(self, temp, conc):
        """What the lab would actually measure (true yield + small noise)."""
        noise = self.rng.normal(0, self.noise_std)
        return float(np.clip(self.true_yield(temp, conc) + noise, 0, 100))

    def current_temp_conc(self):
        return self.TEMPS[self.temp_idx], self.CONCS[self.conc_idx]

    # ---- Gym-style API (no gym dependency needed) ----
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.temp_idx = int(self.rng.integers(0, self.n_temps))
        self.conc_idx = int(self.rng.integers(0, self.n_concs))
        self.steps = 0
        return self.state_id()

    def step(self, action):
        # Move on the grid, stay inside bounds
        if action == 0:
            self.temp_idx = min(self.temp_idx + 1, self.n_temps - 1)
        elif action == 1:
            self.temp_idx = max(self.temp_idx - 1, 0)
        elif action == 2:
            self.conc_idx = min(self.conc_idx + 1, self.n_concs - 1)
        elif action == 3:
            self.conc_idx = max(self.conc_idx - 1, 0)
        else:
            raise ValueError(f"Invalid action {action}, expected 0-3")

        self.steps += 1
        temp, conc = self.current_temp_conc()
        measured_yield = self.observe_yield(temp, conc)
        # Reward encourages high yield and discourages wasting experiments
        reward = measured_yield / 100.0 - 0.01
        done = self.steps >= self.max_steps
        info = {"temp": temp, "conc": conc, "yield": measured_yield}
        return self.state_id(), reward, done, info

    def yield_grid(self):
        """Full true-yield table for plotting (rows=temp, cols=conc)."""
        grid = np.zeros((self.n_temps, self.n_concs))
        for i, t in enumerate(self.TEMPS):
            for j, c in enumerate(self.CONCS):
                grid[i, j] = self.true_yield(t, c)
        return grid
