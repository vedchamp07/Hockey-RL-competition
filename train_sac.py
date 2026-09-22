"""Soft Actor-Critic trainer for laser hockey (no stable-baselines3)."""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from policy import (
    ACT_DIM,
    OBS_DIM,
    DeterministicPolicy,
    SquashedGaussianActor,
    export_policy,
    policy_param_count,
)
from selfplay_pool import OpponentPool

from hockey.hockey_env import BasicOpponent

from domain_random import TRAIN_RANGE, sample_physics
from env.hockey_env import HockeyEnv


DEVICE = torch.device("cpu")
ALPHA = 0.2
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
BATCH_SIZE = 256
REPLAY_CAPACITY = 200_000
GRAD_CLIP = 1.0


class TwinQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q1 = self._build()
        self.q2 = self._build()

    @staticmethod
    def _build() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(OBS_DIM + ACT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.act = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)  # 1 = goal (stop bootstrap)
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        bootstrap_done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = bootstrap_done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=DEVICE),
            torch.as_tensor(self.act[idx], device=DEVICE),
            torch.as_tensor(self.rew[idx], device=DEVICE),
            torch.as_tensor(self.next_obs[idx], device=DEVICE),
            torch.as_tensor(self.done[idx], device=DEVICE),
        )


def make_env(mode: str, physics: dict[str, float] | None = None) -> HockeyEnv:
    return HockeyEnv(mode=mode, physics=physics, verbose=False)


def soft_update(target: nn.Module, source: nn.Module, tau: float = TAU) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def update(
    actor: SquashedGaussianActor,
    critics: TwinQ,
    critics_targ: TwinQ,
    actor_opt: Adam,
    critic_opt: Adam,
    buffer: ReplayBuffer,
    updates: int = 1,
) -> None:
    if buffer.size < BATCH_SIZE:
        return
    for _ in range(updates):
        obs, act, rew, next_obs, done = buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            next_act, next_logp = actor.sample(next_obs)
            q1_t, q2_t = critics_targ(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t) - ALPHA * next_logp
            y = rew + GAMMA * (1.0 - done) * q_t

        q1, q2 = critics(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critics.parameters(), GRAD_CLIP)
        critic_opt.step()

        new_act, logp = actor.sample(obs)
        q1_pi, q2_pi = critics(obs, new_act)
        actor_loss = (ALPHA * logp - torch.min(q1_pi, q2_pi)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        actor_opt.step()

        soft_update(critics_targ, critics)


@torch.no_grad()
def act_deterministic(actor: SquashedGaussianActor, obs: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    a = actor.deterministic(x).squeeze(0).cpu().numpy()
    return np.clip(a, -1.0, 1.0).astype(np.float32)


@torch.no_grad()
def act_stochastic(actor: SquashedGaussianActor, obs: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    a, _ = actor.sample(x)
    return np.clip(a.squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)


def evaluate(
    actor: SquashedGaussianActor,
    mode: str,
    opponent: Any,
    n_games: int,
    seed: int,
) -> dict[str, float]:
    wins = draws = losses = 0
    returns: list[float] = []
    env = make_env(mode, physics={"friction": 1.0, "mass": 1.0})
    for g in range(n_games):
        obs, _ = env.reset(seed=seed + 10_000 + g)
        ep_ret = 0.0
        done = False
        while not done:
            a1 = act_deterministic(actor, obs)
            a2 = opponent.act(env.obs_agent_two())
            obs, reward, terminated, truncated, info = env.step(np.hstack([a1, a2]))
            ep_ret += float(reward)
            done = bool(terminated) or bool(truncated)
        returns.append(ep_ret)
        winner = int(info.get("winner", 0))
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    env.close()
    n = max(n_games, 1)
    return {
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "win_rate": wins / n,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    actor = SquashedGaussianActor().to(DEVICE)
    critics = TwinQ().to(DEVICE)
    critics_targ = copy.deepcopy(critics)
    for p in critics_targ.parameters():
        p.requires_grad_(False)

    actor_opt = Adam(actor.parameters(), lr=LR)
    critic_opt = Adam(critics.parameters(), lr=LR)
    buffer = ReplayBuffer(REPLAY_CAPACITY)

    weak = args.opponent != "strong"
    basic = BasicOpponent(weak=weak, keep_mode=True)
    pool = OpponentPool(basic, max_size=8) if args.self_play else None

    det_params = policy_param_count(DeterministicPolicy())
    print(
        f"obs={OBS_DIM} act={ACT_DIM} policy_params={det_params} mode={args.mode}"
    )

    env = make_env(args.mode)
    recent_returns: deque[float] = deque(maxlen=args.log_every)
    total_steps = 0

    for ep in range(1, args.episodes + 1):
        if not args.no_domain_random:
            env.set_physics(sample_physics(TRAIN_RANGE, seed=args.seed + ep))
        obs, _ = env.reset(seed=args.seed + ep)

        opponent: Any = pool.sample() if pool is not None else basic
        ep_ret = 0.0
        done = False

        while not done:
            if total_steps < args.warmup:
                a1 = np.random.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
            else:
                a1 = act_stochastic(actor, obs)

            a2 = np.asarray(opponent.act(env.obs_agent_two()), dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(np.hstack([a1, a2]))

            if args.shape:
                reward = float(reward) + 0.1 * float(info.get("reward_puck_direction", 0.0))
            else:
                reward = float(reward)

            # Bootstrap unless a true goal (winner != 0). Timeouts still bootstrap.
            bootstrap_done = 1.0 if int(info.get("winner", 0)) != 0 else 0.0
            buffer.add(obs, a1, reward, next_obs, bootstrap_done)

            obs = next_obs
            ep_ret += reward
            total_steps += 1
            done = bool(terminated) or bool(truncated)

            if total_steps >= args.warmup:
                update(
                    actor,
                    critics,
                    critics_targ,
                    actor_opt,
                    critic_opt,
                    buffer,
                    updates=args.updates_per_step,
                )

        recent_returns.append(ep_ret)

        should_log = ep % args.log_every == 0 or ep == args.episodes
        should_snapshot = pool is not None and ep % 500 == 0
        if should_log or should_snapshot:
            stats = evaluate(actor, args.mode, basic, args.eval_games, args.seed)
            if should_log:
                mean_train = float(np.mean(recent_returns)) if recent_returns else 0.0
                print(
                    f"ep={ep} train_ret={mean_train:.2f} "
                    f"W/D/L={int(stats['wins'])}/{int(stats['draws'])}/{int(stats['losses'])} "
                    f"win_rate={stats['win_rate']:.2f}"
                )
            if should_snapshot and pool is not None:
                det = DeterministicPolicy()
                det.fc1.load_state_dict(actor.fc1.state_dict())
                det.fc2.load_state_dict(actor.fc2.state_dict())
                det.mean.load_state_dict(actor.mean.state_dict())
                pool.maybe_add(det.state_dict(), stats["win_rate"], ep, every=500)

        if ep % args.save_every == 0:
            export_policy(actor, args.out)

    export_policy(actor, args.out)
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC trainer for hockey-rl")
    p.add_argument("--mode", choices=("normal", "shooting", "defense"), default="normal")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--out", type=str, default="weights.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opponent", choices=("weak", "strong"), default="weak")
    p.add_argument("--self-play", action="store_true")
    p.add_argument("--no-domain-random", action="store_true")
    p.add_argument("--shape", action="store_true", help="Add puck-direction shaping")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
