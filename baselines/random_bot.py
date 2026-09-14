import numpy as np


class Agent:
    def __init__(self):
        """Random baseline — no model to load."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
