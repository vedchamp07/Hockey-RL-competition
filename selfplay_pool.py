"""Opponent pool: BasicOpponent plus historical deterministic snapshots."""

from __future__ import annotations

import copy
import random
from typing import Any, Protocol

import numpy as np
import torch

from policy import DeterministicPolicy


class Actable(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray: ...


class SnapshotOpponent:
    """Wraps a DeterministicPolicy state dict for .act(obs)."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.net = DeterministicPolicy()
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = self.net(x).squeeze(0).numpy()
        return np.asarray(a, dtype=np.float32)


class OpponentPool:
    """Samples BasicOpponent or past deterministic policy snapshots."""

    def __init__(self, basic_opponent: Actable, max_size: int = 8) -> None:
        self.basic_opponent = basic_opponent
        self.max_size = max_size
        self._snapshots: list[dict[str, torch.Tensor]] = []
        self._best_score = float("-inf")

    def maybe_add(
        self,
        actor_state_dict: dict[str, Any],
        eval_score: float,
        episode: int,
        every: int = 500,
    ) -> bool:
        """Add a CPU deepcopy when episode % every == 0 and score beats best."""
        if episode <= 0 or episode % every != 0:
            return False
        if eval_score <= self._best_score:
            return False
        snap = {k: v.detach().cpu().clone() for k, v in actor_state_dict.items()}
        self._snapshots.append(snap)
        self._best_score = eval_score
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)
        return True

    def sample(self) -> Actable:
        if not self._snapshots or random.random() < 0.3:
            return self.basic_opponent
        return SnapshotOpponent(random.choice(self._snapshots))
