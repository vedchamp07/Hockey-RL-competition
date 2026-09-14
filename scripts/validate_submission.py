#!/usr/bin/env python3
"""Validate a hockey-rl submission before accepting it into the tournament.

Checks performed:
  - The path argument is a directory
  - agent.py exists and imports without error
  - A class named Agent exists with __init__ and act methods
  - Agent() can be instantiated without raising
  - act() on a dummy (18,) observation returns a numpy array of shape (4,)
  - act() completes within a HARD 100ms limit (no slow inference in eval)
  - requirements.txt exists
  - (warning only) writeup.md exists

Exit code 0 = valid submission.
Exit code 1 = invalid submission or usage error (see stderr for details).

Usage:
    python scripts/validate_submission.py <submission_dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback

import numpy as np

ACT_TIME_LIMIT_S = 0.100  # 100ms HARD limit, no exceptions
N_UNTIMED_WARMUP = 2
N_TIMED_CALLS = 10

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fail(message: str) -> None:
    print(f"INVALID: {message}", file=sys.stderr)
    sys.exit(1)


def check_action(action, where: str) -> None:
    if not isinstance(action, np.ndarray):
        fail(f"act() must return a numpy array, but {where} returned {type(action)!r}")
    if action.shape != (4,):
        fail(
            f"act() must return an array of shape (4,), but {where} returned "
            f"shape {action.shape}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a hockey-rl agent submission before adding it to the tournament."
    )
    parser.add_argument(
        "submission_dir",
        nargs="?",
        help="Path to the submission directory (must contain agent.py).",
    )
    args = parser.parse_args()

    if not args.submission_dir:
        parser.print_usage(sys.stderr)
        fail("missing required argument: submission_dir")

    submission_dir = os.path.abspath(args.submission_dir)

    if not os.path.isdir(submission_dir):
        fail(f"'{submission_dir}' is not a directory")

    agent_path = os.path.join(submission_dir, "agent.py")
    if not os.path.isfile(agent_path):
        fail(f"agent.py not found in '{submission_dir}'")

    # Make both submission-local imports (e.g. helper modules shipped
    # alongside agent.py) and repo-root imports (e.g. `env.hockey_wrapper`)
    # work, regardless of where this script is invoked from.
    for p in (submission_dir, REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Many submissions load model weights via relative paths in __init__, so
    # chdir into the submission dir while it imports/instantiates.
    original_cwd = os.getcwd()
    os.chdir(submission_dir)

    try:
        spec = importlib.util.spec_from_file_location("submission_agent", agent_path)
        module = importlib.util.module_from_spec(spec)
        try:
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except ModuleNotFoundError as e:
            if "hockey" in str(e).lower():
                fail(
                    "agent.py failed to import because the 'hockey' package "
                    "(hockey-env) is not installed in this environment. "
                    "Install it first, e.g.:\n"
                    "  pip install 'hockey @ git+https://github.com/martius-lab/hockey-env.git'\n"
                    f"Original error: {e}"
                )
            fail(
                f"agent.py raised an exception on import: {e}\n"
                f"{traceback.format_exc()}"
            )
        except Exception as e:
            fail(
                f"agent.py raised an exception on import: {e}\n"
                f"{traceback.format_exc()}"
            )

        if not hasattr(module, "Agent"):
            fail("agent.py does not define a class named 'Agent'")

        Agent = module.Agent

        if not hasattr(Agent, "__init__"):
            fail("Agent class has no __init__ method")
        if not hasattr(Agent, "act") or not callable(getattr(Agent, "act")):
            fail("Agent class has no callable 'act' method")

        try:
            agent = Agent()
        except Exception as e:
            fail(f"Agent() failed to instantiate: {e}\n{traceback.format_exc()}")

        dummy_obs = np.zeros(18, dtype=np.float32)

        # Untimed warmup calls -- lazy model loading / JIT warmup on the first
        # call(s) shouldn't count against the inference time limit.
        warmup_action = None
        for _ in range(N_UNTIMED_WARMUP):
            try:
                warmup_action = agent.act(dummy_obs)
            except Exception as e:
                fail(f"agent.act() raised an exception: {e}\n{traceback.format_exc()}")
        check_action(warmup_action, "a warmup call")

        # Timed calls -- ANY single call over the limit fails validation.
        for i in range(N_TIMED_CALLS):
            start = time.perf_counter()
            try:
                action = agent.act(dummy_obs)
            except Exception as e:
                fail(f"agent.act() raised an exception: {e}\n{traceback.format_exc()}")
            elapsed = time.perf_counter() - start

            if elapsed > ACT_TIME_LIMIT_S:
                fail(
                    f"act() exceeded 100ms (took {elapsed * 1000:.2f}ms); "
                    "slow inference is not allowed in eval"
                )
            check_action(action, f"timed call #{i + 1}")

        req_path = os.path.join(submission_dir, "requirements.txt")
        if not os.path.isfile(req_path):
            fail("requirements.txt not found in submission directory")
        if os.path.getsize(req_path) == 0:
            print(
                "WARNING: requirements.txt is empty.",
                file=sys.stderr,
            )

        writeup_path = os.path.join(submission_dir, "writeup.md")
        if not os.path.isfile(writeup_path):
            print(
                "WARNING: writeup.md not found. A writeup is MANDATORY for the "
                "competition and its absence disqualifies the submission -- "
                "but this script only checks technical validity, so the "
                "submission is otherwise valid.",
                file=sys.stderr,
            )

        print(f"VALID: '{submission_dir}' passed all checks.")
    finally:
        os.chdir(original_cwd)

    sys.exit(0)


if __name__ == "__main__":
    main()
