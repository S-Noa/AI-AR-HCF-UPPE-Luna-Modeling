"""Gymnasium environment for surrogate-based raw4 UV-fraction optimization."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SurrogateUVEnv(gym.Env):
    """Continuous parameter-refinement environment backed by a frozen surrogate objective."""

    def __init__(self, predictor, lower, upper, max_steps=20, action_scale=0.10, seed=42):
        super().__init__()
        self.predictor = predictor
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.max_steps = int(max_steps)
        self.action_scale = float(action_scale)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Constrained objectives can include a penalty and therefore be negative.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.params = None
        self.score = None
        self.step_index = 0

    def _observe(self):
        return np.concatenate([self.params, [self.score, self.step_index / self.max_steps]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.params = self.rng.uniform(0.0, 1.0, size=4).astype(np.float32)
        self.score = float(self.predictor(self.params[None, :])[0])
        self.step_index = 0
        return self._observe(), {"objective_score": self.score}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        previous = self.score
        self.params = np.clip(self.params + self.action_scale * action, 0.0, 1.0)
        self.score = float(self.predictor(self.params[None, :])[0])
        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        reward = 100.0 * (self.score - previous) - 0.01 * float(np.dot(action, action))
        if terminated:
            reward += 10.0 * self.score
        return self._observe(), reward, terminated, False, {"objective_score": self.score}

    def decoded_parameters(self):
        return self.lower + self.params * (self.upper - self.lower)
