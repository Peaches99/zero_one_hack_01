# Submission

By Sunday 10:00, every team submits their work through a single Tally form. 

**Submission form:** {TALLY_FORM_URL}

---

## What you submit

| #   | Field              | Format                                 | Notes                                       |
| --- | ------------------ | -------------------------------------- | ------------------------------------------- |
| 1   | **Team name**      | Text                                   | The name you'll be called on stage          |
| 2   | **Repository URL** | Public GitHub link                     | Must be public and MIT licensed (see below) |
| 3   | **Slides**         | PDF upload                             | For your 3-minute pitch, max 10 slides      |
| 4   | **Demo video**     | File upload or link, **max 2 minutes** | Shows the system actually running           |

The Tally form timestamp is your official submission time.

---

## 1. Team name

The name we'll use to call you on stage. Keep it short, memorable, and the same name you used at registration so we can match you.

## 2. Repository

Your working code lives in a **public repository** (GitHub or GitLab). The repo must:

- Be **public** at the time of submission — no private repos, no time-limited access tokens
- Be **MIT licensed** — include a `LICENSE` file at the root with the standard MIT license text and the team / contributors as copyright holders
- Contain a **README.md** at the root with setup and run instructions
- Contain a **REPORT.md** at the root (see Section 3 below — this is required, not optional)
- Include a **`requirements.txt`** or equivalent dependency manifest
- Run from a clean checkout (test this on a colleague's machine before submitting)
- Contain **no secrets** — no API keys, tokens, or credentials in git or git history

The jury will clone and try to run your code, so make sure the README is honest about what's needed (Leonardo access, API keys, downloaded datasets, GPU requirements).

### Why MIT-licensed and public

This hackathon is about building real European AI infrastructure capability — that depends on the work being shareable, reusable, and verifiable. Public + MIT means partners, mentors, and other teams can learn from what you built, and you can reuse it freely after the event.


## 3. REPORT.md (in your repo)

The `REPORT.md` lives in the root of your repository and is part of what the jury reads. It's not a marketing document — it's where the jury sees your honest technical thinking.

Recommended sections (2–4 pages when rendered):

- **TL;DR** — 2–3 sentences: what you built, who it's for, what it achieved
- **Problem** — what specifically you decided to solve and why
- **Approach** — 3–5 bullets on architecture and key technical decisions
- **How to run it** — exact commands a stranger would need
- **Results** — headline metrics, baseline comparison, evidence
- **What worked / What didn't** — honest engineering reporting
- **What you'd do with another 36 hours** — concrete next steps
- **Credits & dependencies** — libraries, models, APIs, datasets, AI coding tools used

## 4. Slides (PDF, max 10 slides)

You get **3 minutes on stage** to pitch your work. Slides support the pitch — they're not the deliverable, the working artifact is. Keep them sharp:

- **1 slide**: team and one-sentence what you built
- **1 slide**: problem and why it matters
- **2–3 slides**: approach and key technical decisions
- **2–3 slides**: results, metrics, evidence
- **1 slide**: what you'd do next

## 5. Demo video (max 2 minutes, hard cutoff)

Two minutes. We mean it. The jury sees many submissions — clarity beats completeness.

A good demo video shows:
- **The problem in 15 seconds** — no setup, just the pain
- **The solution running** — live, not slideware
- **One concrete result** — with a number or comparison
- **The reasoning visible** — what your system decided and why

Format: MP4, 1080p, with audio. Upload directly to Tally, or paste an unlisted YouTube/Vimeo/Loom link in the form. 
---

## Pre-submission checklist

Before hitting submit on the Tally form, confirm:

- [ ] Team name matches your registration
- [ ] Repo is **public**
- [ ] Repo has an **MIT `LICENSE` file** at the root
- [ ] Repo has a **`README.md`** with setup and run instructions
- [ ] Repo has a **`REPORT.md`** at the root
- [ ] `requirements.txt` (or equivalent) is present
- [ ] No secrets in the repo — check twice
- [ ] Repo runs from clean checkout
- [ ] Slides are PDF and under 20 MB
- [ ] Demo video is **under 2 minutes**
- [ ] Track-specific deliverables are in your repo (see your track's briefing in `/tracks/`)

---

## What we judge

Each track has its own rubric in [`/judging/rubrics.md`](../judging/rubrics.md). Across all tracks we look for:

1. **Working artifact** that actually runs
2. **Honest, reproducible evaluation** with real numbers
3. **Visible technical choices** — what you decided and why
4. **Genuine use of infrastructure** — Leonardo, the partner API, the data
5. **No basic LLM wrappers** — there must be real engineering underneath

Polish does not beat substance. A rough demo with strong results wins over a slick demo with no measurement.

---

## Common mistakes that cost teams

- **Submitting at 09:58.** Tally can flake, uploads take a moment. Submit by 09:45 — you can re-submit until 10:00 if needed.
- **Repo accidentally still private.** We can't review what we can't see. Check the visibility setting before you submit.
- **Forgetting the LICENSE file.** MIT is one short text file at the root — easy to forget under deadline pressure.
- **REPORT.md missing or empty.** The jury reads this carefully. 
- **No `requirements.txt`.** "It works on my machine" is not reproducible. We will try to run your code.
---

## Track-specific deliverables (in your repo)

Each track expects additional outputs in the repo beyond the four submission fields. These don't go into the Tally form — they live in your repository and should be referenced from the REPORT.

### 🧾 Insurance AI (UNIQA)
- Working Conversion Coach prototype that runs
- Simulation across at least three personas
- Hypotheses document with 2–3 validated logics
- Demo video shows the prototype handling at least one persona from each segment

### ⚙️ Industrial AI (Infineon)
- Eval submission files:
  - `nextstep.csv` (Task 1 format)
  - `completion.csv` (Task 2 format)
  - `anomaly.csv` (Task 3 format)
- Training artifacts: checkpoint(s), training logs, loss curves
- Scores from `eval_metrics.py` on all three tasks, with per-family breakdown
- Demo shows baseline vs. trained output on identical inputs

### 📈 Forecasting AI (Sybilion)
- Working agent or application — not slideware
- Backtest results: at least one historical scenario validating the decision logic
- Driver-importance visualization included in the demo
- Agent ready to adapt to a mid-run assumption shift on Sunday
- Domain choice rationale in the repo README

---

## Questions?

`#{your-track}` on Discord, or find a Lumos team member at the front desk.

Good luck.
