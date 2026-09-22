"""Blend ladder score and brawl Elo into a final ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def final_score(ladder_normalized: float, brawl_elo_normalized: float) -> float:
    """0.30 * ladder + 0.70 * elo, both already normalized to [0, 1] by the caller."""

    return 0.30 * float(ladder_normalized) + 0.70 * float(brawl_elo_normalized)


def blend(entries: list[dict]) -> list[dict]:
    """Normalize ladder by /13 and Elo by min-max, then rank by final desc.

    Each entry needs ``name``, ``ladder_score`` (0..13), and ``elo`` (raw).
    If all Elo values are equal, every ``norm_elo`` is 1.0.
    """

    if not entries:
        return []

    elos = [float(e["elo"]) for e in entries]
    elo_min = min(elos)
    elo_max = max(elos)
    elo_span = elo_max - elo_min

    ranked: list[dict[str, Any]] = []
    for entry in entries:
        norm_ladder = float(entry["ladder_score"]) / 13.0
        if elo_span <= 0:
            norm_elo = 1.0
        else:
            norm_elo = (float(entry["elo"]) - elo_min) / elo_span
        final = final_score(norm_ladder, norm_elo)
        ranked.append(
            {
                "name": entry["name"],
                "ladder_score": float(entry["ladder_score"]),
                "elo": float(entry["elo"]),
                "norm_ladder": norm_ladder,
                "norm_elo": norm_elo,
                "final": final,
            }
        )

    ranked.sort(key=lambda r: r["final"], reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def _print_table(rows: list[dict]) -> None:
    header = f"{'RANK':<6}{'NAME':<20}{'LADDER':>8}{'ELO':>10}{'FINAL':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['rank']:<6}{row['name']:<20}"
            f"{row['ladder_score']:>8.2f}{row['elo']:>10.1f}{row['final']:>10.4f}"
        )


_DEMO_ENTRIES = [
    {"name": "alpha", "ladder_score": 11.5, "elo": 1180.0},
    {"name": "beta", "ladder_score": 8.0, "elo": 1050.0},
    {"name": "gamma", "ladder_score": 4.2, "elo": 920.0},
]


def _main() -> int:
    parser = argparse.ArgumentParser(description="Blend ladder and brawl Elo into a final ranking")
    parser.add_argument(
        "json_file",
        nargs="?",
        help='JSON with {"entries": [{"name", "ladder_score", "elo"}, ...]}',
    )
    args = parser.parse_args()

    if args.json_file:
        path = Path(args.json_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["entries"] if isinstance(data, dict) else data
    else:
        print("No input file; demonstrating with built-in example entries.\n")
        entries = _DEMO_ENTRIES

    rows = blend(entries)
    print(json.dumps(rows, indent=2))
    print()
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
