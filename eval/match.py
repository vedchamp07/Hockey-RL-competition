"""Run reproducible, failure-tolerant matches between hockey agents."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Protocol
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.hockey_wrapper import EVAL_SEED, HockeyGame


class Agent(Protocol):
    """Structural type implemented by every submitted agent."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return one four-dimensional action."""


MatchResult = dict[str, Any]
ProgressCallback = Callable[[MatchResult], None]


def _error_text(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    return message if len(message) <= 500 else f"{message[:497]}..."


def _new_result() -> MatchResult:
    return {
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws": 0,
        "total_episodes": 0,
        "timeout": False,
        "agent1_forfeits": 0,
        "agent2_forfeits": 0,
        "errors": [],
    }


def _safe_action(agent: Agent, observation: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    try:
        action = agent.act(np.asarray(observation).copy())
        if not isinstance(action, np.ndarray):
            return None, "act() must return a numpy.ndarray"
        if action.shape != (4,):
            return None, f"act() returned shape {action.shape}, expected (4,)"
        if not np.issubdtype(action.dtype, np.number):
            return None, f"act() returned non-numeric dtype {action.dtype}"
        action = np.asarray(action, dtype=np.float32)
        if not np.all(np.isfinite(action)):
            return None, "act() returned a non-finite action"
        return np.clip(action, -1.0, 1.0), None
    except BaseException as error:
        return None, _error_text(error)


def _record_physical_winner(
    result: MatchResult,
    winner: int,
    agent1_is_player1: bool,
) -> None:
    if winner == 0:
        result["draws"] += 1
    elif (winner == 1) == agent1_is_player1:
        result["agent1_wins"] += 1
    else:
        result["agent2_wins"] += 1
    result["total_episodes"] += 1


def _record_forfeit(
    result: MatchResult,
    *,
    agent1_failed: bool,
    agent2_failed: bool,
    episode_index: int,
    agent1_error: str | None,
    agent2_error: str | None,
) -> None:
    if agent1_failed:
        result["agent1_forfeits"] += 1
        result["errors"].append(
            {"episode": episode_index, "agent": "agent1", "error": agent1_error}
        )
    if agent2_failed:
        result["agent2_forfeits"] += 1
        result["errors"].append(
            {"episode": episode_index, "agent": "agent2", "error": agent2_error}
        )

    if agent1_failed and agent2_failed:
        result["draws"] += 1
    elif agent1_failed:
        result["agent2_wins"] += 1
    else:
        result["agent1_wins"] += 1
    result["total_episodes"] += 1


def _fill_remaining_draws(result: MatchResult, n_episodes: int) -> MatchResult:
    completed = int(result.get("total_episodes", 0))
    remaining = max(0, n_episodes - completed)
    result["draws"] = int(result.get("draws", 0)) + remaining
    result["total_episodes"] = completed + remaining
    return result


def _run_match(
    agent1: Agent,
    agent2: Agent,
    n_episodes: int,
    timeout_seconds: int,
    progress_callback: ProgressCallback | None = None,
) -> MatchResult:
    if n_episodes < 0:
        raise ValueError("n_episodes must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    result = _new_result()
    deadline = time.monotonic() + timeout_seconds
    game: HockeyGame | None = None

    try:
        try:
            game = HockeyGame(seed=EVAL_SEED)
        except BaseException as error:
            result["errors"].append({"episode": None, "agent": "environment", "error": _error_text(error)})
            _fill_remaining_draws(result, n_episodes)
            if progress_callback:
                progress_callback(result)
            return result

        for episode_index in range(n_episodes):
            if time.monotonic() >= deadline:
                result["timeout"] = True
                _fill_remaining_draws(result, n_episodes)
                if progress_callback:
                    progress_callback(result)
                break

            agent1_is_player1 = episode_index % 2 == 0
            player1 = agent1 if agent1_is_player1 else agent2
            player2 = agent2 if agent1_is_player1 else agent1

            try:
                observation1, _ = game.reset(seed=EVAL_SEED + episode_index)
                observation2 = game.obs_agent_two()
            except BaseException as error:
                result["errors"].append(
                    {"episode": episode_index, "agent": "environment", "error": _error_text(error)}
                )
                result["draws"] += 1
                result["total_episodes"] += 1
                if progress_callback:
                    progress_callback(result)
                continue

            episode_finished = False
            while not episode_finished:
                if time.monotonic() >= deadline:
                    result["timeout"] = True
                    _fill_remaining_draws(result, n_episodes)
                    if progress_callback:
                        progress_callback(result)
                    episode_finished = True
                    break

                action1, player1_error = _safe_action(player1, observation1)
                action2, player2_error = _safe_action(player2, observation2)

                if player1_error is not None or player2_error is not None:
                    if agent1_is_player1:
                        agent1_error, agent2_error = player1_error, player2_error
                    else:
                        agent1_error, agent2_error = player2_error, player1_error
                    _record_forfeit(
                        result,
                        agent1_failed=agent1_error is not None,
                        agent2_failed=agent2_error is not None,
                        episode_index=episode_index,
                        agent1_error=agent1_error,
                        agent2_error=agent2_error,
                    )
                    if progress_callback:
                        progress_callback(result)
                    episode_finished = True
                    continue

                try:
                    observation1, _, terminated, truncated, info = game.step(action1, action2)
                    if not (terminated or truncated):
                        observation2 = game.obs_agent_two()
                        continue

                    raw_winner = info.get("winner", 0)
                    winner = int(raw_winner)
                    if winner not in (-1, 0, 1):
                        raise ValueError(f"invalid winner value {raw_winner!r}")
                    _record_physical_winner(result, winner, agent1_is_player1)
                except BaseException as error:
                    result["errors"].append(
                        {"episode": episode_index, "agent": "environment", "error": _error_text(error)}
                    )
                    result["draws"] += 1
                    result["total_episodes"] += 1

                if progress_callback:
                    progress_callback(result)
                episode_finished = True

            if result["timeout"]:
                break
    finally:
        if game is not None:
            try:
                game.close()
            except BaseException:
                pass

    return result


def run_match(
    agent1: Agent,
    agent2: Agent,
    n_episodes: int = 20,
    timeout_seconds: int = 30,
) -> MatchResult:
    """
    Run a match in the current process.

    Call :func:`run_match_isolated` for untrusted submissions so a hanging
    ``act`` call can be terminated by the parent process.
    """

    return _run_match(agent1, agent2, n_episodes, timeout_seconds)


def load_agent_from_file(agent_file: str):
    """Import a Python file that defines class ``Agent`` and return ``Agent()``."""

    path = Path(agent_file).expanduser().resolve()
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
    inserted_path = parent not in sys.path
    if inserted_path:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
        agent_class = getattr(module, "Agent")
        return agent_class()
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if inserted_path:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def load_agent_from_dir(agent_dir: str):
    """Import ``agent.py`` from a directory and return ``Agent()``."""

    directory = Path(agent_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"agent directory not found: {directory}")
    return load_agent_from_file(str(directory / "agent.py"))


def _load_agent_from_spec(agent_spec: str):
    path = Path(agent_spec).expanduser()
    if path.is_dir():
        return load_agent_from_dir(str(path))
    if path.is_file():
        return load_agent_from_file(str(path))

    looks_like_path = (
        path.suffix == ".py"
        or os.sep in agent_spec
        or (os.altsep is not None and os.altsep in agent_spec)
    )
    if looks_like_path:
        candidate = path if path.suffix == ".py" else path / "agent.py"
        return load_agent_from_file(str(candidate))

    module_name, separator, class_name = agent_spec.partition(":")
    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name if separator else "Agent")
    return agent_class()


def _initialization_failure_result(
    agent1_error: BaseException | None,
    agent2_error: BaseException | None,
    n_episodes: int,
) -> MatchResult:
    result = _new_result()
    agent1_failed = agent1_error is not None
    agent2_failed = agent2_error is not None
    if agent1_failed:
        result["agent1_forfeits"] = n_episodes
        result["errors"].append(
            {"episode": None, "agent": "agent1", "error": _error_text(agent1_error)}
        )
    if agent2_failed:
        result["agent2_forfeits"] = n_episodes
        result["errors"].append(
            {"episode": None, "agent": "agent2", "error": _error_text(agent2_error)}
        )

    if agent1_failed and agent2_failed:
        result["draws"] = n_episodes
    elif agent1_failed:
        result["agent2_wins"] = n_episodes
    else:
        result["agent1_wins"] = n_episodes
    result["total_episodes"] = n_episodes
    return result


def _write_progress(path: str | None, result: MatchResult) -> None:
    if not path:
        return
    destination = Path(path)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    os.replace(temporary, destination)


def _worker_run(
    agent1_spec: str,
    agent2_spec: str,
    n_episodes: int,
    timeout_seconds: int,
    progress_file: str | None,
) -> MatchResult:
    agent1 = None
    agent2 = None
    agent1_error: BaseException | None = None
    agent2_error: BaseException | None = None

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                agent1 = _load_agent_from_spec(agent1_spec)
            except BaseException as error:
                agent1_error = error
            try:
                agent2 = _load_agent_from_spec(agent2_spec)
            except BaseException as error:
                agent2_error = error

            if agent1_error is not None or agent2_error is not None:
                result = _initialization_failure_result(agent1_error, agent2_error, n_episodes)
                _write_progress(progress_file, result)
                return result

            assert agent1 is not None and agent2 is not None
            return _run_match(
                agent1,
                agent2,
                n_episodes,
                timeout_seconds,
                progress_callback=lambda current: _write_progress(progress_file, current),
            )


def _read_result_text(output: str | bytes | None) -> MatchResult | None:
    if output is None:
        return None
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "total_episodes" in value:
            return value
    return None


def _read_progress(path: Path) -> MatchResult | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _kill_worker(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def run_match_isolated(
    agent1_spec: str,
    agent2_spec: str,
    n_episodes: int = 20,
    timeout_seconds: int = 30,
) -> MatchResult:
    """Load two agents in a subprocess and enforce a hard wall-clock timeout."""

    if n_episodes < 0:
        raise ValueError("n_episodes must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    with tempfile.TemporaryDirectory(prefix="hockey_match_") as temporary_dir:
        progress_path = Path(temporary_dir) / "progress.json"
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--agent1",
            os.fspath(agent1_spec),
            "--agent2",
            os.fspath(agent2_spec),
            "--episodes",
            str(n_episodes),
            "--timeout",
            str(timeout_seconds),
            "--progress-file",
            str(progress_path),
        ]
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(REPO_ROOT)
            if not existing_pythonpath
            else os.pathsep.join((str(REPO_ROOT), existing_pythonpath))
        )

        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            _kill_worker(process)
            stdout, stderr = process.communicate()
            partial_stdout = stdout or error.output
            result = _read_progress(progress_path) or _read_result_text(partial_stdout) or _new_result()
            _fill_remaining_draws(result, n_episodes)
            result["timeout"] = True
            return result

        result = _read_result_text(stdout) or _read_progress(progress_path)
        if result is None:
            result = _new_result()
            stderr = stderr.strip()
            result["errors"].append(
                {
                    "episode": None,
                    "agent": "worker",
                    "error": stderr[-500:] if stderr else f"worker exited with code {process.returncode}",
                }
            )
            _fill_remaining_draws(result, n_episodes)
            result["worker_error"] = True
        elif process.returncode != 0:
            _fill_remaining_draws(result, n_episodes)
            result["worker_error"] = True
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated hockey match")
    parser.add_argument("agent_specs", nargs="*", help="two agent files, directories, or import specs")
    parser.add_argument("--agent1")
    parser.add_argument("--agent2")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--progress-file", help=argparse.SUPPRESS)
    return parser


def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    agent1_spec = args.agent1
    agent2_spec = args.agent2
    if agent1_spec is None or agent2_spec is None:
        if len(args.agent_specs) == 2:
            agent1_spec, agent2_spec = args.agent_specs
        elif not sys.stdin.isatty():
            payload = json.load(sys.stdin)
            agent1_spec = payload.get("agent1") or payload.get("agent1_spec")
            agent2_spec = payload.get("agent2") or payload.get("agent2_spec")
            args.episodes = int(payload.get("n_episodes", args.episodes))
            args.timeout = int(payload.get("timeout_seconds", args.timeout))
        else:
            parser.error("provide two agent specs or a JSON object on stdin")

    if not agent1_spec or not agent2_spec:
        parser.error("both agent specs are required")

    if args.worker:
        result = _worker_run(
            agent1_spec,
            agent2_spec,
            args.episodes,
            args.timeout,
            args.progress_file,
        )
    else:
        result = run_match_isolated(
            agent1_spec,
            agent2_spec,
            args.episodes,
            args.timeout,
        )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
