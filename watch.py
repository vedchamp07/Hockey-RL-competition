"""Render (or headlessly step) one or more hockey games."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from hockey.hockey_env import BasicOpponent

from domain_random import sample_physics
from env.hockey_env import MAX_STEPS, HockeyEnv


class _Actable(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray: ...


class RandomAgent:
    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)


class TorchScriptAgent:
    """Left-side policy loaded from a TorchScript ``.pt`` file."""

    def __init__(self, path: str) -> None:
        import torch

        self._torch = torch
        self.net = torch.jit.load(path, map_location="cpu")
        self.net.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.tensor(obs, dtype=self._torch.float32).unsqueeze(0)
            a = self.net(x).squeeze(0).numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)


def _load_agent_from_file(path: str) -> _Actable:
    path_obj = Path(path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"agent file not found: {path}")
    spec = importlib.util.spec_from_file_location("watch_user_agent", path_obj)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Agent"):
        raise AttributeError(f"{path} does not define class Agent")
    return module.Agent()


def build_agent(name: str) -> _Actable:
    if name == "random":
        return RandomAgent()
    if name == "basic":
        return BasicOpponent(weak=True, keep_mode=True)
    if name == "strong":
        return BasicOpponent(weak=False, keep_mode=True)
    return _load_agent_from_file(name)


def build_opponent(name: str) -> _Actable:
    if name == "random":
        return RandomAgent()
    if name == "weak":
        return BasicOpponent(weak=True, keep_mode=True)
    if name == "basic":
        return BasicOpponent(weak=False, keep_mode=True)
    raise ValueError(f"unknown opponent {name!r}; expected basic, weak, or random")


def play_game(
    left: _Actable,
    right: _Actable,
    *,
    mode: str = "normal",
    seed: int = 0,
    physics: dict[str, float] | None = None,
    render: bool = True,
) -> int:
    env = HockeyEnv(mode=mode, physics=physics)
    obs, info = env.reset(seed=seed)
    if render:
        env.render(mode="human")

    winner = 0
    for _ in range(MAX_STEPS):
        a1 = left.act(obs)
        a2 = right.act(env.obs_agent_two())
        obs, _reward, terminated, truncated, info = env.step(np.hstack([a1, a2]))
        if render:
            env.render(mode="human")
        if terminated or truncated:
            winner = int(info.get("winner", 0))
            break
    else:
        winner = int(info.get("winner", 0))

    env.close()
    return winner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a hockey game")
    parser.add_argument(
        "--agent",
        default="random",
        help="random | basic | strong | path/to/agent.py",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="TorchScript .pt file; if set, plays as the left agent",
    )
    parser.add_argument(
        "--opponent",
        default="basic",
        choices=("basic", "weak", "random"),
        help="right-side opponent (default: strong BasicOpponent)",
    )
    parser.add_argument("--mode", default="normal", choices=("normal", "shooting", "defense"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument(
        "--physics",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=None,
        help="optional physics scale range; sample once per game",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="step without opening a display (for headless smoke tests)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.weights is not None:
        left: _Actable = TorchScriptAgent(args.weights)
    else:
        left = build_agent(args.agent)
    right = build_opponent(args.opponent)

    render = not args.no_render
    try:
        for g in range(args.games):
            if args.physics is not None:
                physics = sample_physics((args.physics[0], args.physics[1]), seed=args.seed + g)
            else:
                physics = {"friction": 1.0, "mass": 1.0}

            winner = play_game(
                left,
                right,
                mode=args.mode,
                seed=args.seed + g,
                physics=physics,
                render=render,
            )
            print(f"game {g}: winner={winner}")
    except Exception as exc:  # noqa: BLE001 — surface display failures cleanly
        msg = str(exc).lower()
        display_markers = (
            "display",
            "pygame",
            "sdl",
            "video",
            "no available video device",
            "couldn't find matching render driver",
        )
        if render and any(m in msg for m in display_markers):
            print(
                "Display unavailable. Re-run on a machine with a graphical display, "
                "or pass --no-render for a headless smoke test.",
                file=sys.stderr,
            )
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
