"""Reproducible head-to-head matches between hockey agents."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain_random import PUBLIC_RANGE, sample_physics
from env.hockey_env import HockeyEnv

ZERO_ACTION = np.zeros(4, dtype=np.float32)
MAX_STEPS = 250
FAIL_FORFEIT = 5


def _safe_act(agent: Any, obs: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (action, failed). On exception, substitute zeros."""

    try:
        action = agent.act(np.asarray(obs, dtype=np.float32))
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
            return ZERO_ACTION.copy(), True
        return np.clip(arr[:4], -1.0, 1.0).astype(np.float32), False
    except BaseException:
        return ZERO_ACTION.copy(), True


def play_match(
    agent_a_factory: Callable[[], Any],
    agent_b_factory: Callable[[], Any],
    n_games: int = 20,
    seed_base: int = 0,
    phys_range: tuple[float, float] = PUBLIC_RANGE,
) -> dict:
    """Factories are callables returning a fresh agent (side swap needs new instances).

    Same physics seed for both sides: sample_physics(phys_range, seed=seed_base+g)
    once per game. Alternate sides: game g even, A is left; odd, B is left.
    Max 250 steps. Stop on terminated or truncated.
    """

    score = {"A": 0, "B": 0, "draw": 0, "games": int(n_games)}
    for g in range(n_games):
        a_is_left = g % 2 == 0
        left = agent_a_factory() if a_is_left else agent_b_factory()
        right = agent_b_factory() if a_is_left else agent_a_factory()

        seed = seed_base + g
        physics = sample_physics(phys_range, seed=seed)
        # Physics applied once via constructor → reset(); do not re-apply.
        env = HockeyEnv(mode="normal", physics=physics)

        try:
            obs, info = env.reset(seed=seed)
            obs2 = env.obs_agent_two()
            fails_left = 0
            fails_right = 0
            winner = 0

            for _ in range(MAX_STEPS):
                a1, fail1 = _safe_act(left, obs)
                a2, fail2 = _safe_act(right, obs2)
                if fail1:
                    fails_left += 1
                if fail2:
                    fails_right += 1
                if fails_left >= FAIL_FORFEIT or fails_right >= FAIL_FORFEIT:
                    if fails_left >= FAIL_FORFEIT and fails_right >= FAIL_FORFEIT:
                        winner = 0
                    elif fails_left >= FAIL_FORFEIT:
                        winner = -1  # right wins
                    else:
                        winner = 1  # left wins
                    break

                obs, _reward, terminated, truncated, info = env.step(np.hstack([a1, a2]))
                obs2 = env.obs_agent_two()
                if terminated or truncated:
                    winner = int(info.get("winner", 0))
                    break
            else:
                winner = int(info.get("winner", 0))

            if winner == 0:
                score["draw"] += 1
            elif (winner == 1) == a_is_left:
                score["A"] += 1
            else:
                score["B"] += 1
        finally:
            try:
                env.close()
            except BaseException:
                pass

    return score


def _import_agent_class(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"agent file not found: {path}")
    path_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_hockey_agent_{path_hash}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    parent = str(path.parent)
    inserted = parent not in sys.path
    if inserted:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
        return getattr(module, "Agent")
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


class _JitAgent:
    def __init__(self, weights_path: Path):
        import torch

        self.net = torch.jit.load(str(weights_path), map_location="cpu")
        self.net.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = torch.tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
            a = self.net(x).squeeze(0).numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)


class _RandomAgent:
    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return np.random.uniform(-1.0, 1.0, size=4).astype(np.float32)


REFERENCE_NAMES = frozenset({"bot", "rusher", "wall", "mirror", "apex"})
BUILTIN_NAMES = frozenset({"weak", "medium", "random"})


def load_agent_factory(spec: str) -> Callable[[], Any]:
    """Return a zero-arg factory.

    ``spec`` is a .py file, a directory containing agent.py, or a reference /
    builtin name (bot, rusher, wall, mirror, apex, weak, medium, random).
    """

    key = spec.strip()
    lower = key.lower()

    if lower in REFERENCE_NAMES:
        from baselines.reference import load_reference

        return lambda name=lower: load_reference(name)

    if lower == "random":
        return _RandomAgent

    if lower in {"weak", "medium"}:
        weights = REPO_ROOT / "baselines" / f"{lower}.pt"

        def factory(path=weights):
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint: {path}")
            return _JitAgent(path)

        return factory

    path = Path(key).expanduser()
    if path.is_dir():
        agent_path = path / "agent.py"
        cls = _import_agent_class(agent_path)
        return cls
    if path.is_file() and path.suffix == ".py":
        cls = _import_agent_class(path)
        return cls
    if path.suffix == ".py" or "/" in key or key.endswith("agent.py"):
        cls = _import_agent_class(path)
        return cls

    raise ValueError(
        f"unknown agent spec {spec!r}; use a path or one of "
        f"{sorted(REFERENCE_NAMES | BUILTIN_NAMES)}"
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Play a hockey match between two agents")
    parser.add_argument("--a", required=True, help="agent A: path or reference name")
    parser.add_argument("--b", required=True, help="agent B: path or reference name")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    factory_a = load_agent_factory(args.a)
    factory_b = load_agent_factory(args.b)
    result = play_match(factory_a, factory_b, n_games=args.games, seed_base=args.seed)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
