#!/usr/bin/env python3
"""Build the hockey-rl leaderboard from tournament result files.

Reads every ``results/tournament_*.json`` (and any other ``*.json`` in the
results directory that has a ``matches`` or ``standings`` key), picks the
most recent one, and produces:

    results/leaderboard.csv
    results/leaderboard.json

A formatted table is also printed to stdout.

Usage:
    python eval/leaderboard.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone

# Guard sys.path so this works whether it's run as `python eval/leaderboard.py`
# from the repo root, `python leaderboard.py` from inside eval/, or imported
# as a module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_ELO = 1000.0
K_FACTOR = 32.0


def _load_json_candidates(results_dir: str) -> list[tuple[str, dict]]:
    """Return (path, data) pairs for every JSON file that looks like a
    tournament result file (i.e. has 'matches' or 'standings')."""
    candidates: list[tuple[str, dict]] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        # leaderboard.json is our own output -- never treat it as a source.
        if os.path.basename(path) == "leaderboard.json":
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and ("matches" in data or "standings" in data):
            candidates.append((path, data))
    return candidates


def _sort_key(item: tuple[str, dict]):
    """Sort key preferring the 'timestamp' field, falling back to the
    filename, then to file mtime."""
    path, data = item
    ts = data.get("timestamp") if isinstance(data, dict) else None
    if isinstance(ts, str) and ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return (2, dt.timestamp(), os.path.basename(path))
        except ValueError:
            pass
    # Fallback: filenames like tournament_20240102_030405.json sort lexically
    # in chronological order; ties broken by mtime.
    return (1, os.path.getmtime(path), os.path.basename(path))


def pick_latest_result_file(results_dir: str) -> tuple[str, dict]:
    candidates = _load_json_candidates(results_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No tournament result files found in '{results_dir}'. "
            "Expected files like 'tournament_<timestamp>.json' containing "
            "'matches' or 'standings'. Run the tournament first, e.g.:\n"
            "  python eval/tournament.py --submissions-dir submissions --results-dir results"
        )
    return max(candidates, key=_sort_key)


def _compute_standings_from_matches(matches: list[dict]) -> list[dict]:
    """Recompute per-agent wins/losses/draws/win_rate/elo from a list of
    pairwise match results, using the standard Elo update (K=32, starting at
    1000)."""
    elo: dict[str, float] = {}
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    draws: dict[str, int] = {}

    def ensure(agent: str) -> None:
        elo.setdefault(agent, DEFAULT_ELO)
        wins.setdefault(agent, 0)
        losses.setdefault(agent, 0)
        draws.setdefault(agent, 0)

    for m in matches:
        a1, a2 = m.get("agent1"), m.get("agent2")
        if not a1 or not a2:
            continue
        ensure(a1)
        ensure(a2)

        w1 = int(m.get("agent1_wins", 0) or 0)
        w2 = int(m.get("agent2_wins", 0) or 0)
        d = int(m.get("draws", 0) or 0)
        total = int(m.get("total_episodes", w1 + w2 + d) or (w1 + w2 + d))
        if total <= 0:
            continue

        score1 = (w1 + 0.5 * d) / total
        score2 = 1.0 - score1

        e1, e2 = elo[a1], elo[a2]
        expected1 = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))
        expected2 = 1.0 - expected1

        elo[a1] = e1 + K_FACTOR * (score1 - expected1)
        elo[a2] = e2 + K_FACTOR * (score2 - expected2)

        wins[a1] += w1
        losses[a1] += w2
        draws[a1] += d

        wins[a2] += w2
        losses[a2] += w1
        draws[a2] += d

    standings = []
    for agent in elo:
        w, l, dr = wins[agent], losses[agent], draws[agent]
        total_games = w + l + dr
        win_rate = (w + 0.5 * dr) / total_games if total_games > 0 else 0.0
        standings.append(
            {
                "agent": agent,
                "wins": w,
                "losses": l,
                "draws": dr,
                "win_rate": win_rate,
                "elo": elo[agent],
            }
        )
    return standings


def build_rankings(data: dict) -> list[dict]:
    """Return a ranked list of {rank, agent, elo, wins, losses, draws,
    win_rate} dicts, using the file's precomputed 'standings' if present,
    otherwise recomputing from 'matches'."""
    standings = data.get("standings")
    if not standings:
        standings = _compute_standings_from_matches(data.get("matches", []))

    rows = []
    for s in standings:
        agent = s.get("agent")
        if agent is None:
            continue
        wins = int(s.get("wins", 0) or 0)
        losses = int(s.get("losses", 0) or 0)
        draws = int(s.get("draws", 0) or 0)
        elo = float(s.get("elo", DEFAULT_ELO))
        win_rate = s.get("win_rate")
        if win_rate is None:
            total = wins + losses + draws
            win_rate = ((wins + 0.5 * draws) / total) if total > 0 else 0.0
        rows.append(
            {
                "agent": agent,
                "elo": elo,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": float(win_rate),
            }
        )

    rows.sort(key=lambda r: r["elo"], reverse=True)

    ranked = []
    for i, r in enumerate(rows, start=1):
        ranked.append(
            {
                "rank": i,
                "agent": r["agent"],
                "elo": r["elo"],
                "wins": r["wins"],
                "losses": r["losses"],
                "draws": r["draws"],
                "win_rate": r["win_rate"],
            }
        )
    return ranked


def write_csv(rankings: list[dict], path: str) -> None:
    fieldnames = ["rank", "agent", "elo", "wins", "losses", "draws", "win_rate"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rankings:
            writer.writerow(row)


def write_json(rankings: list[dict], source_file: str, path: str) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": os.path.basename(source_file),
        "leaderboard": rankings,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def print_table(rankings: list[dict]) -> None:
    header = f"{'RANK':<6}{'AGENT':<24}{'ELO':>8}{'W-L-D':>14}{'WIN%':>10}"
    print(header)
    print("-" * len(header))
    for row in rankings:
        wld = f"{row['wins']}-{row['losses']}-{row['draws']}"
        win_pct = f"{row['win_rate'] * 100:.1f}%"
        print(
            f"{row['rank']:<6}{row['agent']:<24}{row['elo']:>8.1f}{wld:>14}{win_pct:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the hockey-rl leaderboard.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing tournament_*.json result files (default: results).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(
            f"ERROR: results directory '{results_dir}' does not exist. "
            "Run the tournament first.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        source_file, data = pick_latest_result_file(results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    rankings = build_rankings(data)
    if not rankings:
        print(
            f"ERROR: '{source_file}' contained no matches or standings to rank.",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_path = os.path.join(results_dir, "leaderboard.csv")
    json_path = os.path.join(results_dir, "leaderboard.json")

    write_csv(rankings, csv_path)
    write_json(rankings, source_file, json_path)

    print(f"Leaderboard computed from: {source_file}\n")
    print_table(rankings)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
