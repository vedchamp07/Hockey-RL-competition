"""Evaluation APIs for matches and round-robin tournaments."""

from __future__ import annotations

from typing import Any


__all__ = ["run_match", "run_tournament", "load_agent_from_dir"]


def __getattr__(name: str) -> Any:
    """Load public APIs lazily so ``python -m eval.<module>`` stays warning-free."""

    if name in {"run_match", "load_agent_from_dir"}:
        from eval.match import load_agent_from_dir, run_match

        value = {"run_match": run_match, "load_agent_from_dir": load_agent_from_dir}[name]
    elif name == "run_tournament":
        from eval.tournament import run_tournament

        value = run_tournament
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
