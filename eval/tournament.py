"""Round-robin tournament runner with episode statistics and Elo ratings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.match import run_match_isolated


DEFAULT_EPISODES = 20
DEFAULT_TIMEOUT_SECONDS = 120
ELO_INITIAL = 1000.0
ELO_K = 32.0
PHASE1_BASELINES = ("weak_bot", "strong_bot")
PHASE2_BASELINES = ("weak_bot", "strong_bot", "random_bot")
BASELINE_NAMES = set(PHASE2_BASELINES)


@dataclass(frozen=True)
class Competitor:
    name: str
    agent_file: Path


def _submission_competitors(agent_dirs: Sequence[str]) -> list[Competitor]:
    competitors: list[Competitor] = []
    seen_paths: set[Path] = set()
    for raw_directory in agent_dirs:
        directory = Path(raw_directory).expanduser().resolve()
        agent_file = (directory / "agent.py").resolve()
        if agent_file in seen_paths:
            continue
        seen_paths.add(agent_file)
        competitors.append(Competitor(directory.name, agent_file))
    return competitors


def _all_competitors(agent_dirs: Sequence[str], phase: int = 2) -> list[Competitor]:
    competitors = _submission_competitors(agent_dirs)
    baseline_dir = REPO_ROOT / "baselines"
    baselines = PHASE1_BASELINES if phase == 1 else PHASE2_BASELINES
    competitors.extend(Competitor(name, baseline_dir / f"{name}.py") for name in baselines)
    return competitors


def _match_pairs(competitors: Sequence[Competitor], phase: int) -> list[tuple[int, int]]:
    """Phase 1: each submission vs weak/strong (plus a weak-vs-strong calibration).

    Phase 2: full round-robin among submissions and all baselines.
    """
    if phase == 1:
        baseline_idx = [i for i, c in enumerate(competitors) if c.name in PHASE1_BASELINES]
        submission_idx = [i for i, c in enumerate(competitors) if c.name not in BASELINE_NAMES]
        pairs = [(i, j) for i in submission_idx for j in baseline_idx]
        if len(baseline_idx) == 2:
            pairs.append((baseline_idx[0], baseline_idx[1]))
        return pairs
    return [
        (i, j)
        for i in range(len(competitors))
        for j in range(i + 1, len(competitors))
    ]


def _match_record(
    agent1: str,
    agent2: str,
    result: dict[str, Any],
    n_episodes: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "agent1": agent1,
        "agent2": agent2,
        "agent1_wins": int(result.get("agent1_wins", 0)),
        "agent2_wins": int(result.get("agent2_wins", 0)),
        "draws": int(result.get("draws", 0)),
        "total_episodes": int(result.get("total_episodes", n_episodes)),
        "timeout": bool(result.get("timeout", False)),
    }
    for key in ("agent1_forfeits", "agent2_forfeits", "worker_error", "errors"):
        if key in result:
            record[key] = result[key]
    return record


def _standings_frame(
    competitors: Sequence[Competitor],
    statistics: Sequence[dict[str, int]],
    ratings: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for competitor, stats, rating in zip(competitors, statistics, ratings):
        games = stats["wins"] + stats["losses"] + stats["draws"]
        win_rate = (stats["wins"] + 0.5 * stats["draws"]) / games if games else 0.0
        rows.append(
            {
                "agent": competitor.name,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "draws": stats["draws"],
                "win_rate": win_rate,
                "elo": rating,
            }
        )

    columns = ["agent", "wins", "losses", "draws", "win_rate", "elo"]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("elo", ascending=False, kind="stable").reset_index(drop=True)


def _write_results(
    standings: pd.DataFrame,
    matches: list[dict[str, Any]],
    *,
    phase: int,
    n_episodes: int,
    timeout_seconds: int,
    results_dir: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc)
    payload = {
        "timestamp": timestamp.isoformat(),
        "phase": phase,
        "n_episodes_per_match": n_episodes,
        "timeout_seconds": timeout_seconds,
        "matches": matches,
        "standings": standings.to_dict(orient="records"),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    filename_timestamp = timestamp.strftime("%Y%m%dT%H%M%S_%fZ")
    destination = results_dir / f"tournament_{filename_timestamp}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def run_tournament(
    agent_dirs: list[str],
    n_episodes: int = DEFAULT_EPISODES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    results_dir: str | Path | None = None,
    phase: int = 2,
) -> pd.DataFrame:
    """Run the evaluation for a competition phase and return Elo standings.

    Phase 1 plays each submission against the built-in weak and strong bots.
    Phase 2 is a full round-robin among submissions and baselines.
    Episode-level wins, losses, and draws are accumulated over all matches.
    """

    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    competitors = _all_competitors(agent_dirs, phase=phase)
    pairs = _match_pairs(competitors, phase)
    statistics = [{"wins": 0, "losses": 0, "draws": 0} for _ in competitors]
    ratings = [ELO_INITIAL for _ in competitors]
    matches: list[dict[str, Any]] = []
    match_count = len(pairs)

    for match_number, (first_index, second_index) in enumerate(pairs, start=1):
        first = competitors[first_index]
        second = competitors[second_index]
        print(
            f"[phase {phase} {match_number}/{match_count}] {first.name} vs {second.name}",
            flush=True,
        )

        raw_result = run_match_isolated(
            str(first.agent_file),
            str(second.agent_file),
            n_episodes=n_episodes,
            timeout_seconds=timeout_seconds,
        )
        result = _match_record(first.name, second.name, raw_result, n_episodes)
        matches.append(result)

        first_wins = result["agent1_wins"]
        second_wins = result["agent2_wins"]
        draws = result["draws"]
        statistics[first_index]["wins"] += first_wins
        statistics[first_index]["losses"] += second_wins
        statistics[first_index]["draws"] += draws
        statistics[second_index]["wins"] += second_wins
        statistics[second_index]["losses"] += first_wins
        statistics[second_index]["draws"] += draws

        total = result["total_episodes"]
        first_score = (first_wins + 0.5 * draws) / total if total else 0.5
        first_rating = ratings[first_index]
        second_rating = ratings[second_index]
        first_expected = 1.0 / (1.0 + 10.0 ** ((second_rating - first_rating) / 400.0))
        ratings[first_index] = first_rating + ELO_K * (first_score - first_expected)
        ratings[second_index] = second_rating + ELO_K * (
            (1.0 - first_score) - (1.0 - first_expected)
        )

    standings = _standings_frame(competitors, statistics, ratings)
    destination_dir = Path(results_dir) if results_dir is not None else REPO_ROOT / "results"
    _write_results(
        standings,
        matches,
        phase=phase,
        n_episodes=n_episodes,
        timeout_seconds=timeout_seconds,
        results_dir=destination_dir.expanduser().resolve(),
    )
    return standings


def _discover_submissions(submissions_dir: Path) -> list[str]:
    if not submissions_dir.is_dir():
        return []
    return [
        str(entry)
        for entry in sorted(submissions_dir.iterdir(), key=lambda path: path.name)
        if entry.is_dir() and (entry / "agent.py").is_file()
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hockey competition evaluation")
    parser.add_argument("--submissions-dir", default="submissions")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=2,
        help="1 = vs built-in bots only (public leaderboard); 2 = full round-robin",
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()
    agent_dirs = _discover_submissions(Path(args.submissions_dir).expanduser().resolve())
    standings = run_tournament(
        agent_dirs,
        n_episodes=args.episodes,
        timeout_seconds=args.timeout,
        results_dir=args.results_dir,
        phase=args.phase,
    )
    print(standings.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
