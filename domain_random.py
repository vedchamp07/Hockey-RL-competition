"""Physics domain randomization for the hockey environment."""

from __future__ import annotations

from typing import Any

import numpy as np

PUBLIC_RANGE = (0.9, 1.1)
TRAIN_RANGE = (0.8, 1.2)
BRAWL_RANGE = (0.7, 1.3)

# Upstream defaults (RACKETFACTOR = 1.2)
_DEFAULT_PLAYER_DENSITY = 200.0 / 1.2
_DEFAULT_PLAYER_FRICTION = 1.0
_DEFAULT_PUCK_DENSITY = 7.0
_DEFAULT_PUCK_FRICTION = 0.1
_DEFAULT_PUCK_LINEAR_DAMPING = 0.05


def sample_physics(
    phys_range: tuple[float, float] = PUBLIC_RANGE,
    seed: int | None = None,
) -> dict[str, float]:
    """Return ``{'friction': scale, 'mass': scale}``, each uniform in ``phys_range``.

    Same seed yields the same scales. Uses ``numpy.random.Generator`` and does
    not touch the global ``np.random`` state.
    """
    lo, hi = phys_range
    rng = np.random.default_rng(seed)
    return {
        "friction": float(rng.uniform(lo, hi)),
        "mass": float(rng.uniform(lo, hi)),
    }


def _resolve_upstream(env: Any) -> Any:
    """Locate the Box2D hockey env (has ``.puck``)."""
    if hasattr(env, "puck"):
        return env
    if hasattr(env, "_env"):
        return _resolve_upstream(env._env)
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and unwrapped is not env:
        return _resolve_upstream(unwrapped)
    return env


def _scale_body(body: Any, density: float, friction: float) -> None:
    if body is None:
        return
    fixtures = getattr(body, "fixtures", None)
    if not fixtures:
        return
    for fixture in fixtures:
        fixture.density = density
        fixture.friction = friction
    body.ResetMassData()


def apply_physics(env: Any, physics: dict[str, float]) -> None:
    """Scale friction and mass of both players and the puck from defaults.

    ``env`` may be our wrapper or the upstream ``HockeyEnv``. Scales are always
    applied relative to upstream defaults (idempotent). Missing bodies are a no-op.
    """
    upstream = _resolve_upstream(env)
    friction_scale = float(physics["friction"])
    mass_scale = float(physics["mass"])

    player_density = _DEFAULT_PLAYER_DENSITY * mass_scale
    player_friction = _DEFAULT_PLAYER_FRICTION * friction_scale
    puck_density = _DEFAULT_PUCK_DENSITY * mass_scale
    puck_friction = _DEFAULT_PUCK_FRICTION * friction_scale

    for attr in ("player1", "player2"):
        body = getattr(upstream, attr, None)
        _scale_body(body, player_density, player_friction)

    puck = getattr(upstream, "puck", None)
    if puck is None:
        return
    _scale_body(puck, puck_density, puck_friction)
    puck.linearDamping = _DEFAULT_PUCK_LINEAR_DAMPING
