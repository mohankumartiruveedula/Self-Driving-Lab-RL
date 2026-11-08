"""Simple tests for the Self-Driving Lab environment. Run: python -m pytest tests/ -v"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lab_env import SelfDrivingLabEnv


def test_optimum_is_highest():
    env = SelfDrivingLabEnv(seed=0)
    grid = env.yield_grid()
    assert grid.max() > 85  # peak should be high yield
    # Optimum grid point should be near 60-70C and 1.1M
    import numpy as np
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    assert env.TEMPS[i] in (60, 70)
    assert env.CONCS[j] in (1.1, 1.4)


def test_step_stays_in_bounds():
    env = SelfDrivingLabEnv(seed=0)
    env.reset(seed=0)
    env.temp_idx, env.conc_idx = 0, 0
    # Temp -10 and Conc -0.3 at corner should not go negative
    state, reward, done, info = env.step(1)
    assert env.temp_idx == 0
    state, reward, done, info = env.step(3)
    assert env.conc_idx == 0


def test_episode_ends():
    env = SelfDrivingLabEnv(max_steps=5, seed=0)
    env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
    assert steps == 5


def test_yield_range():
    env = SelfDrivingLabEnv(seed=0)
    for t in env.TEMPS:
        for c in env.CONCS:
            assert 0 <= env.true_yield(t, c) <= 100
