"""Reference training script for the hockey-rl competition.

This trains a Soft Actor-Critic (SAC) agent from Stable-Baselines3 against
the built-in *weak* opponent. It is intentionally simple and is NOT a
competitive agent -- it exists only to prove that the training pipeline
(env wrapper -> Stable-Baselines3 -> saved model -> Agent.act()) works
end to end. Participants are expected to build something meaningfully
better than this: a stronger algorithm, better reward shaping, curriculum
training against the strong opponent, self-play, etc.

Usage:
    python submission_template/train.py --timesteps 500000 --output my_agent
"""
from __future__ import annotations

import argparse
import os
import sys

# Add the repo root to sys.path so `env.hockey_wrapper` can be imported
# regardless of the current working directory this script is run from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import SAC  # noqa: E402

from env.hockey_wrapper import HockeySingleAgentEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reference SAC agent for the hockey-rl competition."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total number of training timesteps (default: 500000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="my_agent",
        help=(
            "Filename (no extension needed) to save the trained model to. "
            "Saved relative to the current working directory (default: my_agent)."
        ),
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=None,
        help="Optional directory to write TensorBoard logs to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for the environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Single-agent view of the two-player hockey env: the opponent (the
    # built-in weak bot, since weak_opponent=True) is stepped internally.
    env = HockeySingleAgentEnv(weak_opponent=True, seed=args.seed)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
    )

    print(
        f"Training SAC for {args.timesteps} timesteps against the built-in "
        "weak opponent...\n"
        "(This is a proof-of-concept baseline, not a competitive agent.)"
    )
    model.learn(total_timesteps=args.timesteps)

    model.save(args.output)
    print(f"\nSaved model to '{args.output}.zip'")
    print(
        "\nNext steps:\n"
        f"  1. Copy '{args.output}.zip' into your submission folder, next to agent.py\n"
        "  2. In Agent.__init__, load it with:\n"
        "         from stable_baselines3 import SAC\n"
        f"         self.model = SAC.load('{args.output}.zip')\n"
        "  3. In act(), return your model's prediction, e.g.:\n"
        "         action, _ = self.model.predict(observation, deterministic=True)\n"
        "         return action\n"
        "  4. This SAC-vs-weak-opponent baseline is intentionally weak -- go beyond it!"
    )


if __name__ == "__main__":
    main()
