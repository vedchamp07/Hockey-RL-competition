"""Final brawl: round-robin matches with Elo ranking."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.reference import LADDER, load_reference
from domain_random import BRAWL_RANGE
from eval.match import load_agent_factory, play_match

DEFAULT_ELO = 1000.0
K_FACTOR = 32.0


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def run_brawl(
    factories: dict[str, Callable[[], Any]],
    n_games: int = 20,
    seed_base: int = 0,
) -> dict:
    """Round-robin. Every pair plays ``n_games`` with ``BRAWL_RANGE`` physics.

    Elo starts at 1000, K=32. One update per pairing using
    score = (wins + 0.5 * draws) / games as the actual result.
    """

    names = list(factories.keys())
    elo = {name: DEFAULT_ELO for name in names}
    matches: list[dict] = []
    pair_index = 0

    for a, b in combinations(names, 2):
        seed = seed_base + pair_index * max(n_games, 1)
        result = play_match(
            factories[a],
            factories[b],
            n_games=n_games,
            seed_base=seed,
            phys_range=BRAWL_RANGE,
        )
        wins_a = int(result["A"])
        wins_b = int(result["B"])
        draws = int(result["draw"])
        games = int(result["games"])
        score_a = (wins_a + 0.5 * draws) / games if games > 0 else 0.5
        score_b = 1.0 - score_a

        ea = _expected(elo[a], elo[b])
        eb = 1.0 - ea
        elo[a] = elo[a] + K_FACTOR * (score_a - ea)
        elo[b] = elo[b] + K_FACTOR * (score_b - eb)

        matches.append(
            {
                "a": a,
                "b": b,
                "A": wins_a,
                "B": wins_b,
                "draw": draws,
                "games": games,
                "score_a": score_a,
            }
        )
        pair_index += 1

    return {"matches": matches, "elo": elo}


def _discover_submissions(directory: Path) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {}
    if not directory.is_dir():
        return factories
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / "agent.py").is_file():
            factories[child.name] = load_agent_factory(str(child))
    return factories


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run the final brawl round-robin")
    parser.add_argument(
        "--submissions",
        type=str,
        default="submissions",
        help="directory of submission subdirs each containing agent.py",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="also include the five ladder reference agents",
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    factories = _discover_submissions(Path(args.submissions))
    if args.include_reference:
        for name, _cls, _w in LADDER:
            factories[name] = lambda n=name: load_reference(n)

    if len(factories) < 2:
        print(
            json.dumps({"error": "need at least two agents", "found": list(factories)}),
            file=sys.stderr,
        )
        return 1

    payload = run_brawl(factories, n_games=args.games, seed_base=args.seed)
    print(json.dumps(payload, indent=2))
    print("\nElo standings:")
    for name, rating in sorted(payload["elo"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:<20} {rating:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
