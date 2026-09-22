"""Competition submission agent: loads scripted DeterministicPolicy weights."""

import os

import numpy as np
import torch


class Agent:
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "weights.pt")
        self.net = torch.jit.load(path, map_location="cpu")
        self.net.eval()

    def act(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = self.net(x).squeeze(0).numpy()
        return np.clip(a, -1, 1).astype(np.float32)


if __name__ == "__main__":
    agent = Agent()
    action = agent.act(np.zeros(18, dtype=np.float32))
    print(f"action shape: {action.shape}")
