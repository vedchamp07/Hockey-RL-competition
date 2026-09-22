"""Read-only competition wrapper around the installed ``hockey-env`` package.

Students should not modify this module. Train against the public API only.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from hockey.hockey_env import BasicOpponent, HockeyEnv as UpstreamHockeyEnv, Mode

from domain_random import apply_physics, sample_physics

OBS_DIM = 18
ACTION_DIM = 4
MAX_STEPS = 250

MODE_MAP: dict[str, Mode] = {
    "normal": Mode.NORMAL,
    "shooting": Mode.TRAIN_SHOOTING,
    "defense": Mode.TRAIN_DEFENSE,
}


def _resolve_mode(mode: str | Mode) -> Mode:
    if isinstance(mode, Mode):
        return mode
    key = str(mode).lower()
    if key not in MODE_MAP:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODE_MAP)}")
    return MODE_MAP[key]


def clip_action(action: Any) -> np.ndarray:
    """Return a ``(4,)`` float32 action clipped to ``[-1, 1]``."""
    try:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action must be 4 numbers, got {action!r}") from exc
    if arr.size != ACTION_DIM:
        raise ValueError(f"action must have {ACTION_DIM} elements, got shape {np.shape(action)}")
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


class HockeyEnv:
    """Two-player hockey match; caller supplies the joint 8-d action."""

    def __init__(
        self,
        mode: str | Mode = "normal",
        physics: dict[str, float] | None = None,
        verbose: bool = False,
        keep_mode: bool = True,
    ) -> None:
        resolved = _resolve_mode(mode)
        self._env = UpstreamHockeyEnv(keep_mode=keep_mode, mode=resolved, verbose=verbose)
        self.physics = physics
        self._info: dict[str, Any] = {}

    def set_physics(self, physics: dict[str, float] | None) -> None:
        """Store physics scales; applied on every subsequent :meth:`reset`."""
        self.physics = physics

    def reset(
        self,
        seed: int | None = None,
        one_starting: bool | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, one_starting=one_starting)
        if self.physics is not None:
            apply_physics(self, self.physics)
        self._info = dict(info)
        return np.asarray(obs, dtype=np.float32), self._info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        joint = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0).astype(
            np.float32
        )
        obs, reward, done, _truncated, info = self._env.step(joint)
        self._info = dict(info)
        winner = int(self._info.get("winner", getattr(self._env, "winner", 0)))
        done = bool(done)

        if winner != 0:
            terminated, truncated = True, False
        elif done and winner == 0 and self._env.time >= self._env.max_timesteps:
            terminated, truncated = False, True
        else:
            terminated, truncated = False, False

        return np.asarray(obs, dtype=np.float32), float(reward), terminated, truncated, self._info

    def obs_agent_two(self) -> np.ndarray:
        return np.asarray(self._env.obs_agent_two(), dtype=np.float32)

    def render(self, mode: str = "human") -> Any:
        return self._env.render(mode=mode)

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped(self) -> UpstreamHockeyEnv:
        return self._env

    @property
    def winner(self) -> int:
        return int(self._info.get("winner", getattr(self._env, "winner", 0)))


class SingleAgentHockey(gym.Env):
    """Player 1 learns; the opponent is stepped internally."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        opponent: Any | None = None,
        mode: str | Mode = "normal",
        physics: dict[str, float] | None = None,
        physics_range: tuple[float, float] | None = None,
        verbose: bool = False,
        keep_mode: bool = True,
    ) -> None:
        super().__init__()
        self.env = HockeyEnv(mode=mode, physics=physics, verbose=verbose, keep_mode=keep_mode)
        self.opponent = (
            BasicOpponent(weak=True, keep_mode=True) if opponent is None else opponent
        )
        self.physics_range = physics_range
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.physics_range is not None:
            self.env.set_physics(sample_physics(self.physics_range, seed=seed))
        one_starting = (options or {}).get("one_starting")
        return self.env.reset(seed=seed, one_starting=one_starting)

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a1 = clip_action(action)
        a2 = clip_action(self.opponent.act(self.env.obs_agent_two()))
        return self.env.step(np.hstack([a1, a2]))

    def render(self) -> Any:
        return self.env.render(mode="human")

    def close(self) -> None:
        self.env.close()
