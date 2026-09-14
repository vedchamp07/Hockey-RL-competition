# ERG Hockey RL Competition

Recruitment tournament hosted by the **Electronics and Robotics Guild (ERG)** at IIT Madras. Train an RL agent for 2-player laser hockey. The public leaderboard is the entry ticket — the live viva is the filter.

**Start here:** [guide/getting-started.pdf](guide/getting-started.pdf)

Environment: [martius-lab/hockey-env](https://github.com/martius-lab/hockey-env).

---

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # 3.10–3.12; macOS may need `brew install swig`
pip install -r requirements.txt
python submission_template/agent.py                     # prints: action shape: (4,)
python scripts/validate_submission.py submission_template
```

Copy `submission_template/` and replace the random `act()` with your model. The SAC script in `submission_template/train.py` is a pipeline check, not a competitive agent.

Submit a folder with `agent.py`, weights, `requirements.txt`, and **`writeup.md`** (no writeup = disqualified). Checklist is in the PDF.

| Days | What happens |
|------|----------------|
| 1–10 | Phase 1 — public scores vs `weak_bot` and `strong_bot` |
| 11–14 | Phase 2 — blind round-robin; Phase 1 is wiped; Elo decides who advances |
| 15+ | Live viva for the top 5–8 |

`act()` must return a `(4,)` array in **under 100 ms**. Player 2 always sees the mirrored observation (`obs_agent_two()`).

---

## Organizers

```bash
python scripts/validate_submission.py submissions/<team>/
bash scripts/run_eval.sh --phase 1          # public leaderboard
bash scripts/run_eval.sh --phase 2          # final round-robin
# upload results/leaderboard.csv
```
