# Laser Hockey RL Championship

A 48-hour Kaggle-style reinforcement learning competition. Teams train an agent to play 2D air hockey, climb a public ladder against five organizer reference agents all weekend, then face every other team in a final round-robin brawl after the deadline. Final rank is **30% normalized ladder score + 70% normalized brawl Elo**. Teams of 2–3.

## First hour

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python watch.py --agent random
python train_sac.py --mode shooting
python agent.py
```

On macOS you may need `brew install swig` before `box2d-py` builds. Use Python 3.10–3.12.

| Path | Role |
|---|---|
| `README.md` | this document |
| `requirements.txt` | gymnasium, numpy, torch, box2d-py, pygame, hockey-env |
| `env/hockey_env.py` | environment wrapper (read-only at eval) |
| `agent.py` | submission template |
| `policy.py` | actor network (this is what becomes θ) |
| `train_sac.py` | working SAC trainer |
| `selfplay_pool.py` | opponent-pool sampler |
| `domain_random.py` | physics randomization |
| `watch.py` | renders a game |
| `validate.py` | checks your zip |
| `baselines/weak.pt` | weak checkpoint |
| `baselines/medium.pt` | medium checkpoint |

`HOCKEY_RL_HACKATHON.md` is the full brief for organizers and curious students.

## The game

The environment gives you 18 floats and takes 4 forces in `[-1, 1]`. You never write `if puck_left` — you train a policy that maps observations to actions.

**Observation (18):** paddle xy, angle, vel, ang vel (0–5); opponent (6–11); puck xy/vel (12–15); own / opp possession timers (16–17).

**Action (4):** force x, force y, torque, shoot.

An episode ends on a goal or at 250 steps (draw). Training reward is +10/−10 on the terminal step plus a small shaping term; the ladder ignores reward and counts wins, draws, and losses only. The match server feeds the right player `obs_agent_two()` (mirrored coordinates). Students never write side-handling code.

## Submission

Zip contains only `agent.py` and `weights.pt`. Contract:

- class `Agent` with `act(obs) -> (4,)` float32 in `[-1, 1]`
- weights loaded with `torch.jit.load`
- imports: `numpy`, `torch`, and `os` only
- ≤ 50,000 policy parameters (critic stays on your machine)
- `act()` under 10 ms on CPU
- zip ≤ 5 MB
- no exploration noise at inference, no network, no filesystem writes
- 5 submissions per team per day

```bash
python validate.py submission.zip
```

```python
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
```

## Training

`train_sac.py` is SAC and works out of the box. Useful flags: `--mode shooting|defense|normal`, `--episodes`, `--self-play`, `--no-domain-random`, `--out`.

Run curriculum **shooting → defense → normal**. Domain randomization samples friction and mass scales wider than the public ladder (0.9–1.1); the training default is 0.8–1.2. `selfplay_pool.py` manages the opponent pool. Only the actor parameters θ are submitted.

Common mistakes:

- Leaving exploration noise on at submission
- Forgetting `net.eval()`
- Skipping an inference-speed check and timing out mid-match
- Training only against the weak baseline
- Trusting reward curves instead of win rate
- Exceeding 50k parameters and discovering it at the deadline

## Scoring

Five reference agents and ladder weights: **bot 1**, **rusher 2**, **wall 2**, **mirror 3**, **apex 5**.

```
ladder_score = Σ_i  w_i × (wins_i + 0.5 × draws_i) / n_games_i
```

Maximum is **13.0**. Wall camps the goal and forces you to learn to score; drawing every game earns half credit at best.

The final brawl is a full round-robin (20 games per pairing, sides swapped, wider physics). Ranked by Elo. Official score:

```
final = 0.30 × normalized_ladder + 0.70 × normalized_brawl_elo
```

The public ladder freezes 3 hours before the deadline.

## Rules

- Teams of 2–3
- No pretrained public checkpoints for this environment
- No reading or modifying env internals at eval time
- No external control at inference — the agent runs alone
- Imitation of the provided baselines is allowed; downloading third-party agents is not
- LLM-assisted coding is allowed
- 5 ladder submissions per team per day
- Env bug: report it for a bounty. Exploit it silently and you are DQ'd from the affected games

## Organizers

```bash
python -m eval.ladder path/to/submission --games 50
python -m eval.brawl --submissions submissions --include-reference --games 20
python -m eval.match --a bot --b wall --games 4
```

Production eval runs one container per submission with no network, a read-only filesystem, an import whitelist, a wall-clock timeout, and separate processes for the two agents. The local match runner in this repo is for testing, not the production sandbox.

Reference agents live in `baselines/reference.py`. Replace `mirror.pt` and `apex.pt` with real self-play checkpoints before the event; until then they fall back to the built-in strong opponent. Calibrate so nobody beats apex on Saturday morning and everyone beats bot by Saturday evening.
