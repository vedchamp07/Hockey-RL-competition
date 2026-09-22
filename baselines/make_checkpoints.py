"""Create template and baseline torchscript checkpoints."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# Allow running as script from repo root or baselines/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from policy import DeterministicPolicy  # noqa: E402


class LinearTanhPolicy(nn.Module):
    """a = tanh(W obs + b). Used for hand-crafted PD baselines."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(18, 4, bias=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(obs))


def _encode_pd(fx_gain: float, fy_gain: float, vx_damp: float, vy_damp: float,
               torque_angle: float, torque_fy: float = 0.0,
               shoot_w16: float = 0.0, shoot_bias: float = 0.0) -> LinearTanhPolicy:
    """Encode linear PD rules into Linear(18,4) then tanh.

    fx = fx_gain*(obs[12]-obs[0]) - vx_damp*obs[3]
    fy = fy_gain*(obs[13]-obs[1]) - vy_damp*obs[4]
    torque = torque_angle*obs[2] + torque_fy*(obs[13]-obs[1])
    shoot = shoot_w16*obs[16] + shoot_bias
    """
    net = LinearTanhPolicy()
    with torch.no_grad():
        W = torch.zeros(4, 18)
        b = torch.zeros(4)
        # fx
        W[0, 12] = fx_gain
        W[0, 0] = -fx_gain
        W[0, 3] = -vx_damp
        # fy
        W[1, 13] = fy_gain
        W[1, 1] = -fy_gain
        W[1, 4] = -vy_damp
        # torque
        W[2, 2] = torque_angle
        W[2, 13] = torque_fy
        W[2, 1] = -torque_fy
        # shoot
        W[3, 16] = shoot_w16
        b[3] = shoot_bias
        net.linear.weight.copy_(W)
        net.linear.bias.copy_(b)
    net.eval()
    return net


def main() -> None:
    torch.manual_seed(0)
    root = ROOT
    baselines = os.path.join(root, "baselines")
    os.makedirs(baselines, exist_ok=True)

    # 1) Random DeterministicPolicy template
    weights_path = os.path.join(root, "weights.pt")
    policy = DeterministicPolicy()
    policy.eval()
    torch.jit.script(policy).save(weights_path)
    print(f"wrote {weights_path}")

    # 2) Weak gentle PD
    weak = _encode_pd(
        fx_gain=0.8, fy_gain=0.8, vx_damp=0.3, vy_damp=0.3,
        torque_angle=-0.5, torque_fy=0.0, shoot_w16=0.0, shoot_bias=0.0,
    )
    weak_path = os.path.join(baselines, "weak.pt")
    torch.jit.script(weak).save(weak_path)
    print(f"wrote {weak_path}")

    # 3) Medium stronger PD that shoots
    medium = _encode_pd(
        fx_gain=2.5, fy_gain=2.5, vx_damp=0.6, vy_damp=0.6,
        torque_angle=-1.0, torque_fy=0.4, shoot_w16=1.5, shoot_bias=-0.5,
    )
    medium_path = os.path.join(baselines, "medium.pt")
    torch.jit.script(medium).save(medium_path)
    print(f"wrote {medium_path}")

    for path in (weights_path, weak_path, medium_path):
        m = torch.jit.load(path, map_location="cpu")
        out = m(torch.zeros(1, 18))
        assert out.shape == (1, 4), path
        print(f"load ok {path} -> {tuple(out.shape)}")


if __name__ == "__main__":
    main()
