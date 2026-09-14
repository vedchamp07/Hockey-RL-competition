import numpy as np
from hockey.hockey_env import BasicOpponent


class Agent:
    def __init__(self):
        self._opponent = BasicOpponent(weak=True, keep_mode=True)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self._opponent.act(observation), dtype=np.float32)
