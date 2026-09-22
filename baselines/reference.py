"""Five public-ladder reference agents with distinct styles.

Organizers should replace ``baselines/mirror.pt`` and ``baselines/apex.pt``
after a real self-play run. Saturday calibration requires apex to be beatable
by nobody and bot to be beatable by everyone. Until those checkpoints exist,
Mirror and Apex fall back to ``BasicOpponent(weak=False)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from hockey.hockey_env import BasicOpponent

_BASELINES = Path(__file__).resolve().parent


def _clip(action: Any) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:4], -1.0, 1.0).astype(
        np.float32
    )


def _pd(
    target: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    kp: float = 4.0,
    kd: float = 1.0,
) -> np.ndarray:
    return kp * (target - pos) - kd * vel


def _load_jit(path: Path):
    import torch

    net = torch.jit.load(str(path), map_location="cpu")
    net.eval()
    return net


class Bot:
    """Scripted weak baseline. Ladder weight 1."""

    def __init__(self) -> None:
        self._opp = BasicOpponent(weak=True, keep_mode=True)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return _clip(self._opp.act(obs))


class Rusher:
    """Aggressive attacker: chase puck with lead, shoot on possession, weak defense."""

    def __init__(self) -> None:
        self.kp = 4.0
        self.kd = 1.0
        self.lead = 0.15

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        pos = obs[0:2]
        vel = obs[3:5]
        puck = obs[12:14]
        puck_vel = obs[14:16]
        target = puck + self.lead * puck_vel
        xy = _pd(target, pos, vel, kp=self.kp, kd=self.kd)
        torque = float(np.clip(0.5 * (0.0 - obs[2]) - 0.1 * obs[5], -1.0, 1.0))
        shoot = 1.0 if obs[16] > 0 else 0.0
        return _clip([xy[0], xy[1], torque, shoot])


class Wall:
    """Pure defender camping the own goal line. Rarely scores. Shoot always 0."""

    def __init__(self) -> None:
        self.kp = 2.5
        self.kd = 1.2
        self.target_x = -3.5

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        pos = obs[0:2]
        vel = obs[3:5]
        puck_y = float(np.clip(obs[13], -1.2, 1.2))
        target = np.array([self.target_x, puck_y], dtype=np.float64)
        xy = _pd(target, pos, vel, kp=self.kp, kd=self.kd)
        torque = float(np.clip(0.3 * (0.0 - obs[2]) - 0.1 * obs[5], -1.0, 1.0))
        return _clip([xy[0], xy[1], torque, 0.0])


class Mirror:
    """Mid-strength SAC when ``mirror.pt`` exists; else strong BasicOpponent.

    Organizers should replace ``baselines/mirror.pt`` after a real self-play run.
    """

    def __init__(self) -> None:
        path = _BASELINES / "mirror.pt"
        self._net = None
        self._opp = None
        if path.is_file():
            self._net = _load_jit(path)
        else:
            self._opp = BasicOpponent(weak=False, keep_mode=True)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._net is not None:
            import torch

            with torch.no_grad():
                x = torch.tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
                a = self._net(x).squeeze(0).numpy()
            return _clip(a)
        return _clip(self._opp.act(obs))


class Apex:
    """Strongest reference when ``apex.pt`` exists; else strong BasicOpponent.

    Organizers should replace ``baselines/apex.pt`` after a real self-play run.
    Saturday calibration: apex should be beatable by nobody; bot by everyone.
    """

    def __init__(self) -> None:
        path = _BASELINES / "apex.pt"
        self._net = None
        self._opp = None
        if path.is_file():
            self._net = _load_jit(path)
        else:
            self._opp = BasicOpponent(weak=False, keep_mode=True)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._net is not None:
            import torch

            with torch.no_grad():
                x = torch.tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
                a = self._net(x).squeeze(0).numpy()
            return _clip(a)
        return _clip(self._opp.act(obs))


LADDER: tuple[tuple[str, type, int], ...] = (
    ("bot", Bot, 1),
    ("rusher", Rusher, 2),
    ("wall", Wall, 2),
    ("mirror", Mirror, 3),
    ("apex", Apex, 5),
)

MAX_LADDER_SCORE = 13.0  # 1+2+2+3+5

_REFERENCE_CLASSES: dict[str, type] = {name: cls for name, cls, _ in LADDER}


def load_reference(name: str):
    """Return a fresh instance. ``name`` in bot, rusher, wall, mirror, apex."""

    key = name.strip().lower()
    if key not in _REFERENCE_CLASSES:
        raise ValueError(f"unknown reference agent: {name!r}; expected one of {sorted(_REFERENCE_CLASSES)}")
    return _REFERENCE_CLASSES[key]()
