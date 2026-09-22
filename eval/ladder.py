"""Public ladder: submission vs five weighted reference agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.reference import LADDER, MAX_LADDER_SCORE, load_reference
from domain_random import PUBLIC_RANGE
from eval.match import load_agent_factory, play_match


def ladder_score(
    submission_factory: Callable[[], Any],
    n_games: int = 50,
    seed_base: int = 0,
    phys_range: tuple[float, float] = PUBLIC_RANGE,
) -> dict:
    """Score a submission against the five ladder references.

    Per opponent: points = wins + 0.5 * draws, rate = points / n_games,
    weighted = weight * rate. Total is the sum of weighted rates (max 13.0).
    """

    results: dict[str, dict] = {}
    total = 0.0
    for name, _cls, weight in LADDER:
        match = play_match(
            submission_factory,
            lambda n=name: load_reference(n),
            n_games=n_games,
            seed_base=seed_base,
            phys_range=phys_range,
        )
        wins = int(match["A"])
        losses = int(match["B"])
        draws = int(match["draw"])
        points = wins + 0.5 * draws
        rate = points / n_games if n_games > 0 else 0.0
        weighted = weight * rate
        total += weighted
        results[name] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "rate": rate,
            "weight": weight,
            "weighted": weighted,
        }

    return {
        "score": float(total),
        "max_score": float(MAX_LADDER_SCORE),
        "results": results,
    }


def _print_table(payload: dict) -> None:
    print(f"ladder score: {payload['score']:.4f} / {payload['max_score']:.1f}")
    header = f"{'OPP':<10}{'W':>5}{'D':>5}{'L':>5}{'RATE':>8}{'WGT':>5}{'WEIGHTED':>10}"
    print(header)
    print("-" * len(header))
    for name, _cls, weight in LADDER:
        row = payload["results"][name]
        print(
            f"{name:<10}{row['wins']:>5}{row['draws']:>5}{row['losses']:>5}"
            f"{row['rate']:>8.3f}{weight:>5}{row['weighted']:>10.4f}"
        )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Score a submission on the public ladder")
    parser.add_argument("submission", help="path to agent.py or directory containing it")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    factory = load_agent_factory(args.submission)
    payload = ladder_score(factory, n_games=args.games, seed_base=args.seed)
    print(json.dumps(payload))
    _print_table(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
