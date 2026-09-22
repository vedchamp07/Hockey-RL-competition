# Laser Hockey RL Championship

A 48-hour reinforcement learning competition. Teams train an agent to play 2D air hockey. Agents fight each other. Leaderboard decides everything.

---

## 1. What this competition is

**Format:** Kaggle-style. One environment, one leaderboard, one metric. No stages, no unlocking, no subjective judging.

**Two phases:**

1. **Public ladder** (all weekend) — your agent plays against five organizer-trained reference agents. Weighted score, live leaderboard, fast feedback.
2. **Final brawl** (after deadline) — every team's agent plays every other team's agent. Round-robin. Elo ranking.

**Final rank = 30% ladder score + 70% brawl Elo.**

A live top-8 single-elimination bracket runs on the projector after the brawl. It is exhibition only — the blended score is official.

**Scale:** ~100 students, teams of 2–3.

---

## 2. The environment

### 2.1 What it is

Laser Hockey, built by the Martius Lab (MPI-IS / University of Tübingen) for their RL course. Two-player 2D air hockey on a Box2D physics engine. They've run it as a course tournament for years — the 2023 edition had 89 competing agents.

```bash
pip install git+https://github.com/martius-lab/hockey-env.git
```

Repo: `github.com/martius-lab/hockey-env` (older references call it `laser-hockey-env`).

### 2.2 The game

Two paddles on a rink, one puck, two goals. You control the left paddle. The opponent controls the right. Score to win.

- Physics steps at ~50 Hz
- Episode ends on a goal, or at **250 steps → draw**
- A full match takes well under a second of wall clock, headless

### 2.3 Observation space — 18 floats

Origin is the centre of the rink. All values roughly normalized around zero.

| Index | Meaning |
|---|---|
| 0, 1 | your paddle x, y position |
| 2 | your paddle angle |
| 3, 4 | your paddle x, y velocity |
| 5 | your paddle angular velocity |
| 6, 7 | opponent paddle x, y position |
| 8 | opponent paddle angle |
| 9, 10 | opponent paddle x, y velocity |
| 11 | opponent paddle angular velocity |
| 12, 13 | puck x, y position |
| 14, 15 | puck x, y velocity |
| 16, 17 | puck possession timers (one per player) |

Useful derived quantities:
- `obs[12] - obs[0]` — how far right the puck is from you
- `obs[14]` — is the puck coming at you or moving away

**You never see the opponent's action.** Only its consequences in the next observation. This makes the setting partially observable — the physics is deterministic but from your side the dynamics look stochastic.

### 2.4 Action space — 4 floats in [-1, 1]

| Index | Meaning |
|---|---|
| 0 | force left / right |
| 1 | force up / down |
| 2 | torque — rotate the paddle |
| 3 | shoot — fires the puck if you're holding it |

There is no "move to position (x, y)" and no "aim at goal". Only forces. Your paddle has momentum — push right and you keep drifting right. The agent has to learn to decelerate *before* it arrives, or it overshoots the puck every time.

**Index 2 matters more than beginners think.** Paddle angle determines deflection angle on contact, so a good agent rotates to turn blocks into counterattacks.

### 2.5 Reward

- +10 to the winner on the final step
- −10 to the loser
- 0 for a draw
- Small dense shaping term for keeping your paddle near the puck

The shaping exists so early training isn't a sparse-reward wall. A randomly flailing agent almost never scores, but it does sometimes get near the puck — that's the seed signal it climbs from.

**Competition scoring ignores reward entirely.** Only wins, losses, and draws count.

### 2.6 Training modes

The env ships reduced modes. Use them as a curriculum:

- **shooting** — puck spawns, no defender. Learn to aim.
- **defense** — you only block. Learn to position.
- **normal** — full game.

Skipping the curriculum costs hours.

### 2.7 Side mirroring

`env.obs_agent_two()` returns a mirrored observation for the right player — flipped coordinates so it also believes it's attacking rightward. **You never write side-handling code.** The match server swaps sides between games; your agent is unaware.

---

## 3. Randomized physics

Physics parameters are sampled **per game** from a range. The same seed is used for both sides so neither agent gets luckier physics.

```
Public ladder:  friction ∈ [0.9, 1.1],  mass ∈ [0.9, 1.1]
Final brawl:    wider range — announced as wider, exact bounds not published
```

An agent tuned to one physics configuration will lose position in the brawl. An agent that trained across randomized configs will hold up.

**Domain-randomize during training.** Sample your own physics per episode, wider than the public range. This is the single highest-value thing you can do for your final rank.

---

## 4. What you submit

### 4.1 The object

You submit **θ** — a vector of real numbers. The weights of a neural network. Nothing else.

For a 18 → 256 → 256 → 4 network:

```
W1: 256×18 = 4,608     b1: 256
W2: 256×256 = 65,536   b2: 256
W3: 4×256 = 1,024      b3: 4
                       θ ∈ R^134,684
```

That's about 540 KB. Your entire weekend compresses into that file.

*(Note: that example exceeds this competition's 50k parameter cap — see §4.4. Size your network accordingly.)*

### 4.2 What θ defines

A deterministic policy — a function from states to actions:

```
π_θ : R^18 → [-1,1]^4

a_t = π_θ(s_t) = tanh( W3 · relu(W2 · relu(W1 s_t + b1) + b2) + b3 )
```

The `tanh` keeps actions in range. `agent.py` is just the code that evaluates this expression. It carries no strategy of its own — swap θ for random noise and the same file produces a paddle that vibrates in a corner.

### 4.3 The zip

```
submission.zip
├── agent.py      defines the architecture, loads θ, runs the forward pass
└── weights.pt    θ
```

Nothing else. No training code, no env code, no `main()`.

The server doesn't know or care about your architecture. It imports your `Agent` class and calls `act(obs)`. `agent.py` *is* the architecture spec — that's why weights alone aren't enough.

### 4.4 The contract

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
        # obs: np.ndarray shape (18,) float32
        # returns: np.ndarray shape (4,) float32, clipped to [-1, 1]
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = self.net(x).squeeze(0).numpy()
        return np.clip(a, -1, 1).astype(np.float32)
```

**Hard constraints:**

| Constraint | Limit |
|---|---|
| Policy network parameters | **50,000 max** |
| `act()` inference time | under 10 ms on CPU |
| Zip file size | 5 MB |
| Imports | `numpy`, `torch` only |
| Network access | none |
| Filesystem writes | none |
| Exploration noise at inference | must be off |

The parameter cap counts **the policy network only**. Your critic can be any size — it never leaves your machine.

Validate locally before submitting:

```bash
python validate.py submission.zip
```

This catches roughly 90% of submission failures.

### 4.5 Submission limits

**5 submissions per team per day.** Forces thinking over spraying.

---

## 5. How training works

### 5.1 There is no dataset

This is the core conceptual difference from supervised learning, and the thing most people get stuck on.

In supervised learning you're handed a fixed dataset and you fit to it. In RL **the environment is the data source**. Your agent walks into the simulator and generates its own training data. Every game produces 250 rows:

```
(s_t, a_t, r_t, s_{t+1}, done)
```

Run 2,000 games overnight and you've generated 500,000 transitions. Nobody shipped you that data — your agent made it.

**You need compute, not data.** No downloads, no labels, no preprocessing. Just CPU hours running Box2D.

### 5.2 The loop

```python
env = HockeyEnv()
opponent = BasicOpponent()
buffer = ReplayBuffer(1_000_000)
agent = SAC(obs_dim=18, act_dim=4)

for ep in range(20000):
    obs = env.reset()
    obs2 = env.obs_agent_two()

    # --- data generation ---
    for t in range(250):
        a = agent.act(obs, noise=True)
        a_opp = opponent.act(obs2)
        nxt, r, done, trunc, info = env.step(np.hstack([a, a_opp]))
        buffer.add(obs, a, r, nxt, done)
        obs, obs2 = nxt, env.obs_agent_two()
        if done or trunc:
            break

    # --- data consumption ---
    for _ in range(250):
        agent.update(buffer.sample(256))

    if ep % 100 == 0:
        print(ep, evaluate(agent, opponent, n=20))
```

Generation and consumption interleave. The agent plays, learns from what it played, plays slightly better, generates better data, learns more.

**How it bootstraps from nothing:** the first 10,000 transitions are a random paddle flailing. But among that flailing, a few times the puck accidentally goes in the net and the agent gets +10. Those rare rows are the seed. The critic learns "states that looked like that were worth something", the actor shifts toward them, and the flailing becomes marginally less random. Repeat a million times.

After ~2,000 episodes on a laptop CPU (roughly 40 minutes) you should beat the scripted bot most of the time. That's your first submission.

### 5.3 What you're optimizing

```
θ* = argmax_θ  E[ Σ_t γ^t r_t | π_θ, π_opp ]
```

with γ ≈ 0.99, horizon T = 250.

You can't compute this expectation or its gradient directly — Box2D isn't differentiable. RL estimates the gradient from sampled games. That's what the algorithms are for.

**PPO / policy gradient** estimates the gradient directly from trajectories:

```
∇_θ J ≈ E[ ∇_θ log π_θ(a_t|s_t) · Â_t ]
```

where Â_t is the advantage — how much better that action was than average.

**SAC / TD3** go through a critic. Learn Q by minimizing the Bellman residual:

```
L(φ) = E[ ( Q_φ(s,a) − (r + γ Q_φ̄(s', π_θ̄(s'))) )² ]
```

then improve the actor by climbing the critic:

```
∇_θ J ≈ E[ ∇_a Q_φ(s,a)|_{a=π_θ(s)} · ∇_θ π_θ(s) ]
```

The critic is scaffolding. It exists only to shape θ, and it doesn't get submitted.

**The credit assignment problem** is the whole difficulty. The goal happens at step 156, but the action that caused it was at step 89. These algorithms are different answers to "how do I propagate that +10 back to the move that deserved it."

### 5.4 The knobs you control

**Algorithm.** SAC and TD3 for continuous actions, PPO if you prefer on-policy. Different sample efficiency, different stability. Try more than one.

**Curriculum.** shooting → defense → normal. Carry weights forward between stages.

**Reward shaping.** Add your own terms during training:

```python
r_shaped = r + 0.3 * puck_velocity_toward_enemy_goal - 0.1 * distance_from_own_goal
```

Dangerous knob. Shape too hard and the agent maximizes your shaping term instead of winning. The classic failure: an agent that glues itself to the puck and never shoots, because proximity reward beats goal reward in expectation.

**Self-play.** The step that separates the top 10 from the middle.

```python
pool = [copy.deepcopy(agent)]

if ep % 500 == 0 and eval_score > best:
    pool.append(copy.deepcopy(agent))

opponent = random.choice(pool + [basic_bot])
```

If you only train against the scripted bot, you learn to exploit *its* quirks and lose to everyone on the leaderboard. Published implementations of opponent-pool self-play on this env reach around 90% win rates.

**Domain randomization.** Sample physics per episode, wider than the public ladder range.

**Architecture and hyperparameters.** Under the 50k cap, width vs depth is a real decision. Also: learning rate, replay buffer size, exploration noise, discount factor, batch size.

### 5.5 Mistakes that will eat your Saturday

- Submitting with exploration noise still on — plays ~20% worse than it trained
- Forgetting `net.eval()`
- Not checking inference speed, then timing out mid-match
- Training only against the weak baseline, then getting crushed by the field
- Trusting reward curves instead of win rate — a rising reward curve with a falling win rate means your shaping is lying to you
- Exceeding 50k parameters and discovering it at the deadline

---

## 6. Scoring

### 6.1 Public ladder — reference agents

Your agent plays 50 games against each of five organizer agents. They are deliberately different in *style*, not just strength — a single policy should not beat all five.

| Agent | Style | Weight |
|---|---|---|
| `bot` | scripted baseline | 1 |
| `rusher` | aggressive, always attacks, weak defense | 2 |
| `wall` | pure defender, camps the goal, rarely scores | 2 |
| `mirror` | mid-strength SAC, balanced | 3 |
| `apex` | strongest, self-play trained | 5 |

```
ladder_score = Σ_i  w_i × (wins_i + 0.5 × draws_i) / n_games_i
```

Maximum possible: **13.0**

`wall` is the important one. It forces you to learn to *score*, not just to not-lose. An agent that only blocks draws every game against it and earns half credit at best.

### 6.2 Final brawl

Full round-robin. Every team vs every team, 20 games per pairing, sides swapped every other game, wider physics randomization.

Ranked by Elo fit to match outcomes.

### 6.3 Final rank

```
final = 0.30 × normalized_ladder_score + 0.70 × normalized_brawl_elo
```

Ladder freezes 3 hours before the deadline so brawl results don't move under us.

---

## 7. Schedule

**Day 1**

| Time | |
|---|---|
| 09:00 | Opening briefing — env walkthrough, rules, submission spec (45 min) |
| 10:00 | Repo released. Ladder opens. |
| 10:00 | Optional crash course: MDPs → Q-learning → policy gradients (60 min) |
| 14:00 | Mentor office hours |
| 18:00 | Leaderboard checkpoint |
| 21:00 | Optional talk: "why your agent isn't learning" — debugging RL |

**Day 2**

| Time | |
|---|---|
| 09:00 | Mentor office hours |
| 12:00 | Ladder freezes |
| 15:00 | Final submissions close |
| 15:00 | Brawl round-robin begins |
| 16:30 | Results + live top-8 bracket on projector |
| 17:30 | Prizes |

---

## 8. Rules

- Teams of 2–3
- No pretrained RL checkpoints for this env from public repos. Many exist — top finishers will be spot-checked.
- No reading or modifying env internals at eval time
- No external control at inference — the agent runs alone
- Imitation learning from provided baselines is allowed; from downloaded third-party agents is not
- LLM-assisted coding is allowed
- 5 ladder submissions per team per day
- Exploiting an env bug: report it and get a bounty prize. Exploiting it silently: disqualified from the affected games.

**Prizes:** overall winner, 2nd, 3rd, best beginner team, best writeup, best bug report.

---

## 9. Organizer infrastructure

### 9.1 Match runner

```python
def play_match(AgentA, AgentB, n_games=20, seed_base=0, phys_range=(0.9, 1.1)):
    score = {"A": 0, "B": 0, "draw": 0}
    for g in range(n_games):
        left, right = (AgentA(), AgentB()) if g % 2 == 0 else (AgentB(), AgentA())
        env = HockeyEnv(physics=sample_physics(phys_range, seed=seed_base + g))
        obs, info = env.reset(seed=seed_base + g)
        obs2 = env.obs_agent_two()

        for t in range(250):
            a1 = left.act(obs)
            a2 = right.act(obs2)
            obs, r, done, trunc, info = env.step(np.hstack([a1, a2]))
            obs2 = env.obs_agent_two()
            if done or trunc:
                break

        w = info.get("winner", 0)   # 1 = left, -1 = right, 0 = draw
        if w == 0:
            score["draw"] += 1
        elif (w == 1) == (g % 2 == 0):
            score["A"] += 1
        else:
            score["B"] += 1
    return score
```

### 9.2 Pipeline

```
upload → validate → queue → sandboxed match workers → score → leaderboard
```

### 9.3 Sandboxing

You are executing arbitrary student Python. Required:

- One container per submission
- No network, read-only filesystem
- Import whitelist (`numpy`, `torch`)
- Wall-clock timeout per match
- Separate processes for the two agents so neither can reach the env object or the opponent's memory
- Timeout handling: a step that times out gets a zero action; repeated timeouts forfeit the game

Someone will try `import os`. Someone else will leave a `print()` in `act()` and emit 250 lines per game. Handle both before Saturday.

### 9.4 Compute budget

**Brawl:** 100 teams → 4,950 pairings × 20 games = ~99,000 games. At sub-second each across 8 parallel workers, roughly one hour.

**Test this on dummy submissions before the event.** Discovering it takes six hours while 100 students wait is the worst possible Sunday.

**Ladder:** 5 agents × 50 games = 250 games per submission. Seconds.

**Students:** design assumes CPU-only training. No GPU allocation required. If you can provide GPU credits, do — but the env is deliberately small enough not to need them.

### 9.5 Calibrating the reference agents

Two rules:

- **Nobody should beat `apex` on Saturday.** If a team does by noon, apex is too weak and the leaderboard flattens. Train it harder than you think you need to.
- **Everyone should beat `bot` by Saturday evening.** If teams sit at zero all day, morale dies.

Train all five in advance. Budget a week. `wall` and `rusher` can be scripted or lightly trained; `mirror` and `apex` need real self-play runs.

---

## 10. Starter repo

```
rl-hackathon/
├── README.md                 this document
├── requirements.txt          gymnasium, numpy, torch, box2d-py, pygame
├── env/
│   └── hockey_env.py         the environment (read-only)
├── agent.py                  submission template
├── train_sac.py              working SAC implementation
├── selfplay_pool.py          opponent-pool sampler example
├── domain_random.py          physics randomization wrapper
├── watch.py                  renders a game
├── validate.py               checks your zip
└── baselines/
    ├── weak.pt
    └── medium.pt
```

`train_sac.py` must **work out of the box**. Students should run one command in minute five and see a learning curve. Debugging someone else's broken RL scaffold is not the lesson.

### First hour for a student

```bash
pip install -r requirements.txt
python watch.py --agent random          # see the flailing
python train_sac.py --mode shooting     # start learning
```

`watch.py` opens a pygame window. The five-minute "oh, my agent is just vibrating in the corner" moment is what makes the observation vector click.

---

## 11. The one-line framing for the briefing

> The environment gives you 18 numbers and takes 4 numbers. You never write hockey strategy. You write a *learning setup*, run it for six hours, and whatever comes out is what plays for you.

Every instinct from normal programming — `if puck_is_left: move_left` — is exactly what students must stop doing. Their job is to build a system that discovers the strategy, then get out of the way.
