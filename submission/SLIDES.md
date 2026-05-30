# Slide outline — "Learning vs. Memorizing Process Logic" (Infineon)

~10 slides for the submission deck. Each bullet ≈ one line on the slide; the
parenthetical is speaker/demo note. Figures live in
`process_lm/runs/figures/`.

---

### 1 — Title
- **Does the model learn process logic, or just memorize?**
- A from-scratch GPT over semiconductor process routes, benchmarked against the
  *exact information limit* of the data.
- One RTX 5090 · reproduces in < 1 hour.

### 2 — The idea that kept us honest
- The data is a *known stochastic grammar* → it has an **exact entropy floor** a
  perfect model can't beat.
- We measured it, so every "lower loss" is provably real progress or impossible.
- (This caught three traps other approaches fall into.)

### 3 — The exact floor (and why you can trust it)
- Instrumented the organizers' real generator → **byte-identical** to it (selftest).
- **ID floor = 0.328 nats/token.** The old ~0.34 plateau *was* the floor.
- Falsifiable: our model sits at 0.331 — **gap < 0.005 on fresh data**, never below.
- (Figure: loss curves with the floor line.)

### 4 — "Reach 0.01" — resolved honestly
- Overall 0.01 is impossible on honest data (it's below the floor).
- But split by position: on the **54% rule-forced transitions the model hits
  0.0002 nats** — it learned the logic perfectly.
- The residual 0.33 is **pure coin-flips** (46% of tokens, ~ln 2 each). No biased set.

### 5 — Scaling = memorizing
- More same-family data / bigger models **do not** improve the held-out family —
  they overfit faster (best epoch 40 → 1).
- (Figure: scaling.png — OOD flat while train hits the floor.)

### 6 — Three traps we caught (the honest part)
- **Hybrids**: help MOSFET (valid 0.76→0.96), *tank* IGBT (0.66→0.48) — a gamble.
- **Our v2 diversity leaked** held-out vocab (177→201) — "win" was leakage; top-1
  actually dropped. Added a leak guard.
- **Single-family/seed overfitting**: lever gaps (~0.03) are **within seed noise
  (±0.04)**. Test one family → confident, wrong answer.

### 7 — Word-level tokenization: a clean negative
- Tried to beat the vocab gap by composing unseen steps from words.
- **Backfired**: OOD top-1 0.41 (vs 0.635), valid 0.22 (vs 1.0) — compounding word
  errors outweigh the ~2% OOV rescue. Step-level is right.

### 8 — The one robust win: validity-guided decoding
- Model proposes each step; the validator vetoes any rule-breaking choice.
- Held-out **valid-completion 0.62 → 1.00 (IGBT)**, 0.73 → 0.82 (MOSFET); never hurts.
- Real full routes, ending in SHIP LOT. Model knowledge + rules guardrail.

### 9 — The three tasks (final model, at the floor)
- **Task 1 next-step:** top-3 0.997, **top-5 1.000**, MRR 0.838.
- **Task 2 completion:** **100% process-valid, 0% rule-breaking**; Block-level
  Accuracy 0.652, block-edit 0.021; 60/60 valid routes generated from scratch.
- **Task 3 anomaly:** LM-surprise detector **AUC 1.000 / F1 1.000** (all 10 rules);
  **rule attribution 0.82** (7/10 rules named; 3 confound an adjacent break — disclosed).
- (Demo: baseline vs trained, side by side — baseline 161 violations, trained valid.)

### 10 — What we claim / won't claim
- **Claim:** a small from-scratch model learns transferable process logic — at the
  information floor, valid routes for unseen families, flags violations from its own
  surprise.
- **Won't claim:** that any augmentation trick robustly helped. None did. We know,
  because we measured against the truth, not our hopes.

---

**Demo video (≤2 min) script:** run `python -m process_lm.demo --ckpt
process_lm/runs/final/best.pt` on camera — show the MOSFET/IGBT/IC before/after
(baseline garbage + invalid route vs trained correct next-step + valid route), then
`python -m process_lm.oracle` (floor + selftest), then `python -m process_lm.anomaly
--ckpt process_lm/runs/final/best.pt` (AUC/F1 1.0).
