# {Team Name} — {Track Name}

> Copy this file into `/submissions/{your-team-name}/REPORT.md` and fill in each section.
> Replace this quote block and all `{placeholders}` with your real content.
> Delete sections you genuinely don't have anything to say in — but most sections matter.

---

## Team

- **{Name}** — {role, e.g. ML / backend / design / domain}
- **{Name}** — {role}
- **{Name}** — {role}

**Track:** {Insurance AI (UNIQA) / Industrial AI (Infineon) / Forecasting AI (Sybilion)}

---

## TL;DR

Two or three sentences. What did you build, who is it for, and what did it achieve?

---

## Problem

What problem did you decide to solve, and why does it matter? Be specific. "Improving conversion" is not a problem — "reducing the 66% drop-off at the initial price screen for Segment 1 customers" is a problem.

Include the angle you chose if the track allowed multiple angles (which persona, which sub-domain, which use case).

---

## Approach

How does your solution work? 3–5 bullets is enough.

- {Key architectural decision 1 and why}
- {Key architectural decision 2 and why}
- {Which model / framework / API / data source you chose and why}
- {Where the system runs — local, Leonardo, the partner API}

A small diagram or architecture sketch in `extras/` helps but is not required.

---

## How to run it

The exact commands a stranger would need to reproduce your work:

```bash
# Setup
git clone {your-repo-url}
cd {repo-name}
pip install -r requirements.txt

# Run
python {entry-point}.py --config configs/final.yaml
```

If Leonardo access, specific API keys, or downloaded datasets are needed, say so explicitly. If your repo has a `README.md` that already covers this, you can just link to it.

---

## Results

The numbers. This is the section the jury reads most carefully.

- **Headline metric**: {what you measured, the number, and what it means}
- **Baseline comparison**: {what you compared against — without this, "we improved X" is meaningless}
- **Per-{persona / family / domain} breakdown** if relevant
- **Where the data came from** to produce these numbers (`extras/results/` is a good place for raw output files)

If you used the track's official scoring script (Infineon `eval_metrics.py`), paste the scores here. If you defined your own metrics, explain what they measure and why they are the right metrics for what you built.

---

## What worked

Two or three things you're genuinely proud of. Be specific — "the architecture was good" doesn't help anyone.

---

## What didn't work

Two or three things you tried that failed or got abandoned. This counts. Honest engineering reporting is part of what we judge.

---

## What you'd do with another 36 hours

Concrete next steps, not aspirations. "Train on more data" is aspiration. "Train two additional model sizes to extend the scaling curve" is concrete.

---

## Track-specific deliverables

Each track has additional required outputs beyond this report. Confirm yours are present:

### 🧾 Insurance AI (UNIQA)
- [ ] Working Conversion Coach prototype runs
- [ ] Simulation across at least three personas
- [ ] Hypotheses document in `extras/hypotheses.md` with 2–3 validated logics
- [ ] Demo video shows the prototype handling at least one persona from each segment

### ⚙️ Industrial AI (Infineon)
- [ ] Eval submission files in `extras/results/`:
  - `nextstep.csv` (Task 1 format)
  - `completion.csv` (Task 2 format)
  - `anomaly.csv` (Task 3 format)
- [ ] Training artifacts: checkpoint(s), training logs, loss curves
- [ ] Scores from `eval_metrics.py` on all three tasks, with per-family breakdown
- [ ] Demo shows baseline vs. trained output on identical inputs

### 📈 Forecasting AI (Sybilion)
- [ ] Working agent or application — not slideware
- [ ] Backtest results: at least one historical scenario validating the decision logic
- [ ] Driver-importance visualization included in demo
- [ ] Agent is ready to adapt to a mid-run assumption shift on Sunday
- [ ] Domain choice rationale stated above in "Problem"

---

## Credits & dependencies

- **Open-source libraries used** (with versions): {list}
- **Pre-trained models used**: {list, or "none"}
- **External APIs called**: {list}
- **AI coding assistants used during the hackathon**: {Claude Code / Cursor / Copilot / etc., or "none"}
- **Datasets**: {sources and license terms if applicable}

---

## A note on honesty

If something in your demo is partly mocked, hardcoded, or stubbed — say so here. We respect honest engineering. We notice when teams hide things, and the jury asks during Q&A. Being upfront is always the better play.

---

*Submitted by team {Team Name} for Zero One Hack_01, {date}.*
