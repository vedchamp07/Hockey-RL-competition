"""Evaluation package: matches, ladder, brawl, leaderboard blend."""

from __future__ import annotations

from typing import Any

__all__ = ["play_match", "ladder_score", "run_brawl", "blend"]


def __getattr__(name: str) -> Any:
    if name == "play_match":
        from eval.match import play_match

        return play_match
    if name == "ladder_score":
        from eval.ladder import ladder_score

        return ladder_score
    if name == "run_brawl":
        from eval.brawl import run_brawl

        return run_brawl
    if name == "blend":
        from eval.leaderboard import blend

        return blend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
