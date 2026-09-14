"""Thin wrappers around ``hockey-env`` for consistent evaluation and training.

Two entry points are provided:

* :class:`HockeyGame` -- explicit two-player interface used by the evaluation
  pipeline. Both player actions are supplied by the caller.
* :class:`HockeySingleAgentEnv` -- a ``gymnasium.Env`` where the opponent is
  stepped internally, so Stable-Baselines3 and friends can train on it.

Observation layout (18 floats, exactly what ``HockeyEnv`` returns)::

    [0:2]   player1 position (x, y) relative to centre
    [2]     player1 angle
    [3:5]   player1 velocity (vx, vy)
    [5]     player1 angular velocity
    [6:8]   player2 position (x, y)
    [8]     player2 angle
    [9:11]  player2 velocity (vx, vy)
    [11]    player2 angular velocity
    [12:14] puck position (x, y)
    [14:16] puck velocity (vx, vy)
    [16]    player1 has puck (timer)
    [17]    player2 has puck (timer)

Player 2 must always be fed :meth:`HockeyGame.obs_agent_two`, which mirrors the
view so the opponent sees itself as "player 1".

Action layout (4 floats, clipped to ``[-1, 1]``)::

    [0] force x, [1] force y, [2] torque, [3] shoot (fires when > 0.5)
"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from hockey.hockey_env import BasicOpponent, HockeyEnv, Mode

__all__ = [
    "ACTION_DIM",
    "EVAL_SEED",
    "OBS_DIM",
    "HockeyGame",
    "HockeySingleAgentEnv",
    "clip_action",
]

EVAL_SEED = 42
OBS_DIM = 18
ACTION_DIM = 4


class Opponent(Protocol):
    """Anything that can play a side: ``BasicOpponent`` or a submitted ``Agent``."""

    def act(self, observation: np.ndarray) -> np.ndarray: ...


def clip_action(action: Any) -> np.ndarray:
    """Coerce ``action`` to a ``float32`` array of shape ``(4,)`` clipped to ``[-1, 1]``.

    Accepts lists, tuples and numpy arrays of any shape whose total size is 4.

    Raises:
        ValueError: if ``action`` cannot be interpreted as 4 floats.
    """
    try:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action is not a numeric array of {ACTION_DIM} floats: {action!r}") from exc
    if arr.size != ACTION_DIM:
        raise ValueError(f"action must have {ACTION_DIM} elements, got shape {np.shape(action)}")
    return np.clip(arr, -1.0, 1.0)


class HockeyGame:
    """Two-player hockey match with a fixed seed for reproducible evaluation.

    The underlying ``HockeyEnv`` takes a single 8-d action (both players
    concatenated); this class keeps the two sides separate, clips each one, and
    normalises the episode-end flags.

    Note on episode end: upstream never sets ``truncated`` -- a timeout is
    reported as ``terminated=True`` with ``info["winner"] == 0``. We translate
    that case into ``terminated=False, truncated=True`` so callers can tell a
    goal apart from a timeout. Callers should still treat *either* flag as
    "episode over" and read ``info["winner"]`` for the result.
    """

    def __init__(
        self,
        seed: int | None = EVAL_SEED,
        mode: Mode | str | int | None = None,
        keep_mode: bool = True,
    ) -> None:
        """
        Args:
            seed: default seed applied on every :meth:`reset` that does not
                override it. Pass ``None`` for unseeded (non-deterministic) play.
            mode: ``hockey.hockey_env.Mode``; defaults to ``Mode.NORMAL``.
            keep_mode: keep the puck-possession timer, making actions 4-d.
        """
        self.seed = seed
        self.mode = Mode.NORMAL if mode is None else mode
        self.keep_mode = keep_mode
        self._env = HockeyEnv(keep_mode=keep_mode, mode=self.mode, verbose=False)
        self._info: dict[str, Any] = {}

    @property
    def env(self) -> HockeyEnv:
        """The underlying ``HockeyEnv``."""
        return self._env

    @property
    def unwrapped(self) -> HockeyEnv:
        """Alias of :attr:`env`, for gymnasium-style access."""
        return self._env

    @property
    def winner(self) -> int:
        """``1`` player 1 scored, ``-1`` player 2 scored, ``0`` draw/timeout."""
        return int(self._info.get("winner", getattr(self._env, "winner", 0)))

    def reset(
        self,
        seed: int | None = None,
        one_starting: bool | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: overrides the seed given to ``__init__`` for this episode.
            one_starting: ``True`` gives player 1 the opening puck, ``False``
                player 2. ``None`` lets the env alternate sides.
        """
        self._info = {}
        obs, info = self._env.reset(
            one_starting=one_starting,
            seed=self.seed if seed is None else seed,
        )
        self._info = dict(info)
        return obs, self._info

    def obs_agent_two(self) -> np.ndarray:
        """Mirrored observation for player 2. Never pass the raw obs to player 2."""
        return self._env.obs_agent_two()

    def step(
        self,
        action1: Any,
        action2: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one frame with both players' actions.

        Each action is clipped to ``[-1, 1]`` and reshaped to ``(4,)``
        independently, then concatenated into the 8-d action the env expects.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` where ``reward`` is
            from player 1's perspective and ``truncated`` marks a timeout.
        """
        joint = np.concatenate([clip_action(action1), clip_action(action2)])
        obs, reward, terminated, truncated, info = self._env.step(joint)
        self._info = dict(info)

        done = bool(terminated) or bool(truncated) or bool(getattr(self._env, "done", False))
        if done and self._is_timeout():
            terminated, truncated = False, True
        else:
            terminated, truncated = done, False
        return obs, float(reward), terminated, truncated, self._info

    def _is_timeout(self) -> bool:
        """True when the episode ended on the clock rather than on a goal."""
        if self.winner != 0:
            return False
        time = getattr(self._env, "time", None)
        limit = getattr(self._env, "max_timesteps", None)
        if time is None or limit is None:
            return False
        return time >= limit

    def close(self) -> None:
        self._env.close()


class HockeySingleAgentEnv(gym.Env):
    """Single-agent view of the game: the opponent is stepped internally.

    The learner always plays player 1 and receives the upstream player-1 reward.
    The opponent is fed the mirrored observation from
    :meth:`HockeyGame.obs_agent_two`.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        opponent: Opponent | None = None,
        weak_opponent: bool = True,
        seed: int | None = None,
        mode: Mode | str | int | None = None,
    ) -> None:
        """
        Args:
            opponent: object exposing ``act(obs) -> (4,)``. Defaults to
                ``BasicOpponent(weak=weak_opponent, keep_mode=True)``.
            weak_opponent: strength of the default built-in opponent.
            seed: seed for the first :meth:`reset` only; later episodes vary so
                training does not replay one fixed game.
            mode: ``hockey.hockey_env.Mode``; defaults to ``Mode.NORMAL``.
        """
        self.game = HockeyGame(seed=None, mode=mode, keep_mode=True)
        self.opponent: Opponent = (
            BasicOpponent(weak=weak_opponent, keep_mode=True) if opponent is None else opponent
        )
        self._pending_seed = seed
        self.render_mode: str | None = None

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            seed, self._pending_seed = self._pending_seed, None
        super().reset(seed=seed)
        one_starting = (options or {}).get("one_starting")
        return self.game.reset(seed=seed, one_starting=one_starting)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the learner's action against the internally sampled opponent action."""
        opponent_action = self.opponent.act(self.game.obs_agent_two())
        return self.game.step(action, opponent_action)

    def render(self, mode: str | None = None) -> Any:
        return self.game.env.render(mode or self.render_mode or "human")

    def close(self) -> None:
        self.game.close()
