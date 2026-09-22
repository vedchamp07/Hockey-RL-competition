"""Deterministic competition policy and squashed-Gaussian SAC actor."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.distributions import Normal

OBS_DIM = 18
ACT_DIM = 4
HIDDEN = 128

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class DeterministicPolicy(nn.Module):
    """pi(s) = tanh(W3 relu(W2 relu(W1 s + b1) + b2) + b3). Scripted into weights.pt."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.mean = nn.Linear(HIDDEN, ACT_DIM)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mean(x))


class SquashedGaussianActor(nn.Module):
    """SAC actor: shared trunk + mean/log_std heads with tanh squashing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.mean = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(HIDDEN, ACT_DIM)

    def _trunk(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs))
        return torch.relu(self.fc2(x))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._trunk(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


def policy_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def export_policy(actor: SquashedGaussianActor, path: str) -> None:
    """Copy fc1, fc2, mean weights into DeterministicPolicy and torch.jit.script it."""
    policy = DeterministicPolicy()
    policy.fc1.load_state_dict(actor.fc1.state_dict())
    policy.fc2.load_state_dict(actor.fc2.state_dict())
    policy.mean.load_state_dict(actor.mean.state_dict())
    policy.eval()
    scripted = torch.jit.script(copy.deepcopy(policy).cpu())
    scripted.save(path)
