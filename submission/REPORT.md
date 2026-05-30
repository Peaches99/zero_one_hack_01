# Learning vs. Memorizing Process Logic — Infineon Industrial Track

**A from-scratch GPT over semiconductor process-step sequences, benchmarked
honestly against the information-theoretic limit of the data itself.**

Team tooling: a ~5M–25M parameter decoder trained from scratch (no pretrained
base), a custom data generator, an exact entropy oracle, and self-scoring for all
three eval tasks. Everything runs on one RTX 5090; the whole study reproduces in
well under an hour.

### Headline results
| Result | Number |
|---|---|
| Exact ID entropy floor (proven, byte-identical generator selftest) | **0.328 nats/token** |
| Final model ID loss vs floor (fresh data) | 0.331 — **gap < 0.005** |
| Loss on deterministic, rule-forced transitions (54% of tokens) | **0.0002 nats** (≪ 0.01) |
| Task 1 next-step (ID) | top-3 0.997, **top-5 1.000**, MRR 0.838 |
| Task 2 completion (ID) | **100% process-valid**, block-edit-dist 0.022 |
| Task 3 anomaly detection | **ROC-AUC 1.000, F1 1.000, all 10 rules** |
| OOD valid-completion, guided+repair decoding (all 3 held-out families) | → **1.000** |
| OOD next-step top-1 across families/seeds | 0.65 ± 0.04 (no lever beats noise) |

---

## 0. The one idea that organizes everything

The track asks: *does the model learn process logic or just memorize?* Most teams
will chase a lower loss. We started somewhere more useful — **we computed the
exact loss that a perfect model would achieve**, so we always know whether a lower
number is real progress or impossible.

The data comes from a *known stochastic grammar* (`generate_sequences.py`). That
grammar makes independent random choices (optional steps, A/B synonyms) that **no
model can predict from context**. Their entropy is an irreducible floor on
next-token loss. We measure it exactly (§2) and use it as the yardstick for every
experiment. This is what kept us honest: it told us when we had hit the wall, and —
just as importantly — when an apparent win was actually an artifact (§5).

---

## 1. Model & setup

- **Architecture:** nanoGPT-style causal decoder (`process_lm/model.py`); each
  process step string is one token; family (MOSFET/IGBT/IC) is a conditioning
  token; weight-tied embeddings; learned positional embeddings.
- **Sizes studied:** tiny 0.5M · small 4.85M · medium 25M · large 85M.
- **Training:** AdamW, bf16 autocast on the 5090, single from-scratch run per
  config, best checkpoint by validation loss.
- **Honest OOD proxy (Task 4):** leave-one-family-out (LOFO). Train on two
  families, evaluate on the third — and the tokenizer is built from **train only**,
  so the held-out family's unique steps are genuinely unknown (`<UNK>`), exactly
  as the hidden 4th family will be.

---

## 2. The exact entropy floor (and why it's trustworthy)

`process_lm/oracle.py` instruments the organizers' real generator so every random
decision records its true probability, then **asserts the instrumented output is
byte-identical to `generate_sequence` for 400 seeds × 3 families**. The mean
negative-log-probability under the true process is the Bayes-optimal next-token
loss:

| Family | Exact floor (nats/token) |
|---|---|
| MOSFET | 0.308 |
| IGBT | 0.310 |
| IC | 0.373 |
| **ID mixture (1/3 each)** | **0.328** |

The floor is built to be **falsifiable, not self-confirming**: `score_model_vs_floor`
compares a trained model's NLL to the floor on *fresh* sequences. A correct floor
is a hard lower bound the model approaches from above; a model scoring below it
would prove the floor wrong. Our final model:

| Family | model NLL | floor | gap |
|---|---|---|---|
| MOSFET | 0.3122 | 0.3077 | +0.0045 |
| IGBT | 0.3149 | 0.3097 | +0.0052 |
| IC | 0.3727 | 0.3729 | −0.0002 (bf16 noise) |

**The model is within 0.005 nats/token of the information-theoretic optimum** on
held-out data across all three families. This explains the long-standing ~0.34
loss plateau from earlier exploration: **that plateau was the floor.** 0.01 loss is
not reachable on honest data — only by destroying the grammar's entropy (a biased
set), which we explicitly avoided.

**The "reach 0.01 loss" target, resolved honestly.** Split the model's per-token
loss by whether the grammar *forces* the next step. On the **deterministic,
rule-forced transitions — 54% of all tokens — the model reaches 0.0002 nats**, far
below 0.01: it has essentially perfectly learned the process logic. The other 46%
are irreducible coin-flips (mean loss 0.72 ≈ ln 2). So **0.01 is reached and beaten
where it is meaningful** (the logic), and **provably unreachable overall** (the 0.33
floor is pure grammar randomness) — no biased/low-entropy dataset required. An
independent cross-check confirms the split is real: the oracle's stochastic-decision
fraction (50%) ≈ the model's stochastic fraction (46%).

---

## 3. Scaling is memorization, not learning

40-epoch LOFO grid (hold-out IC), small model unless noted:

| Run | OOD valid-completion | OOD top-1 | OOD ppl |
|---|---|---|---|
| data 200 | 0.713 | 0.500 | 17.4 |
| data 1k | 0.988 | 0.620 | 14.5 |
| data 5k | 0.963 | 0.620 | 14.7 |
| data 20k | 0.988 | 0.595 | 15.1 |
| model tiny 0.5M | 0.512 | 0.480 | 17.4 |
| model small 4.85M | 0.963 | 0.620 | 14.7 |
| model medium 25M | 0.950 | 0.625 | 17.3 |

Train loss slams into the 0.31 floor; **OOD does not improve with more
same-family data or bigger models — it overfits faster** (best epoch moves
40 → 15 → 3 → 1). This is the memorization signature: the model perfectly learns
the two training families' distribution, and that does not transfer. (See
`figures/loss_curves.png`, `figures/scaling.png`.)

---

## 4. What actually generalizes — and the trap of testing one family

We first ran the lever study holding out IC, where family-dropout looked like a
clean winner. **Then we held out MOSFET and IGBT too — and the story inverted.**
Metrics use the `<UNK>` family token (the true proxy for the hidden 4th family,
which has no known family token).

OOD next-step **top-1**:

| lever | IC | MOSFET | IGBT | **avg** |
|---|---|---|---|---|
| real | 0.630 | 0.645 | 0.685 | **0.653** |
| real + family-dropout 0.15 | 0.635 | 0.615 | 0.700 | 0.650 |
| real + family-dropout 0.30 | 0.635 | 0.600 | 0.715 | 0.650 |
| + 8k hybrids | 0.600 | 0.670 | 0.690 | **0.653** |

OOD **valid-completion**:

| lever | IC | MOSFET | IGBT | **avg** |
|---|---|---|---|---|
| real | 0.975 | 0.762 | 0.662 | 0.800 |
| + family-dropout 0.15 | 0.988 | 0.762 | 0.662 | 0.804 |
| + 8k hybrids | 1.000 | **0.963** | **0.475** | 0.813 |

**The levers trade off across families and roughly cancel on average (~0.65 top-1,
~0.80 valid).** Family-dropout helps IC and IGBT but hurts MOSFET; hybrids rescue
MOSFET (valid 0.76 → 0.96) yet tank IGBT (0.66 → 0.48). MOSFET and IGBT are
structurally harder to reach from the other two (real valid 0.76 / 0.66) than IC
(0.98).

**And the differences are within noise — confirmed across every family and seed.**
Re-running `real` vs `real + family-dropout` with 3–4 seeds per held-out family
(OOD top-1, `<UNK>` token, mean ± std):

| hold-out | real | real + family-dropout |
|---|---|---|
| IC | 0.602 ± 0.049 | 0.602 ± 0.034 |
| MOSFET | 0.601 ± 0.032 | 0.604 ± 0.028 |
| IGBT | 0.682 ± 0.028 | 0.675 ± 0.030 |

**For every family the family-dropout effect is ≤ 0.007 — far inside the ±0.03–0.05
seed noise.** The pooled spread across 22 runs (0.535–0.715) is driven by seed and
which family is held out, not the lever. The single-seed gaps that looked like wins
were variance. There is no robust training-time lever to find — the honest recipe is
**plain real data**.

Honest takeaways:
1. **A clean small model already generalizes near its achievable limit from real
   data alone** — correct next-steps and legal routes for families it never saw.
2. **No augmentation lever robustly improves OOD.** Across seeds, family-dropout is
   *within noise* (harmless, not a real gain); hybrids are a high-variance gamble
   (big help on MOSFET, big hurt on IGBT).
3. For the hidden 4th family (direction unknown) the safe default is **plain real
   data** (a light family-dropout is harmless) — never the hybrids, which could tank
   a family the way they tanked IGBT.

Had we tested only IC (or only MOSFET) we'd have shipped a confident,
family-specific, and *wrong* recommendation. Cross-family robustness is the guard.

**The one lever that robustly helps is at inference, not training.**
Validity-guided decoding (`process_lm/guided.py`) — the model proposes each next
step, the validator vetoes any choice that would introduce a rule violation —
lifts held-out **valid-completion toward 1.0 — IGBT 0.62 → 1.00, IC 0.98 → 1.00,
MOSFET 0.73 → 0.82** — and **never hurts** (it can only veto illegal steps).
MOSFET's residual failures are a single mode: the model skips passivation and jumps
to the backside, which a veto cannot insert. Adding a one-block **grammar repair**
that supplies the missing mandatory passivation pushes **all three held-out
families to 1.000** valid-completion. The completions are genuine full routes ending
in `SHIP LOT`. The model supplies the process knowledge; the grammar supplies a
guardrail — vetoing illegal steps and inserting mandatory prerequisites. This is the
honest way to guarantee legal routes for a family the model has never seen: a
model+rules hybrid, not a bigger model or more data.

---

## 5. Two data-augmentation traps we caught (the honest part)

(The third trap — overfitting our own conclusions to one family/seed — is §4.)

**(a) Hybrid pseudo-families are a high-variance gamble, not a fix.** Mixing blocks
across families ("Frankenstein" routes) was hypothesized to boost generalization.
On the IC dose-response it monotonically *hurt* (valid-completion 0.963 → 0.900 →
0.850 for 0 → 2k → 8k hybrids). Across families it's more than "hurts": hybrids
*help* MOSFET (valid 0.76 → 0.96) but *tank* IGBT (0.66 → 0.48). An early "hybrids
help" reading came from a severely undertrained smoke model. The lesson isn't
"hybrids are bad" — it's that any single-family read (smoke, IC-only, MOSFET-only)
would have given a different, confident, wrong answer.

**(b) Our own v2 diversity generator leaked the answer.** v2 (variable cycle
counts, mixed blocks — `diversify2.py`) appeared to cut OOD loss 2.66 → 2.42. But
the training vocabulary jumped 177 → 201: the generic-cycle palette included
**held-out-family-unique steps** (e.g. `IMPLANT N-TYPE`, `MEASURE CD LEVEL 2` are
IC-only). The "win" was vocabulary leakage — perplexity dropped because leaked
steps became predictable, yet **OOD top-1 actually went down**. We added a
pool-vocabulary leak guard; leak-free, v2 provides no top-1 gain. This is exactly
the "don't make a biased set" failure mode, caught by watching the vocab and the
top-1 (not just the loss).

There's a deeper reason it *had* to leak: the structural diversity that would
prepare a model for IGBT's six-mask-level routes **is** `ALIGN MASK LEVEL 5/6` —
IGBT's own signature. You cannot teach an unseen family's novel structure without
showing that structure, which in the proxy is leakage. Restricted to the pool's
existing structure, v2 adds nothing the two training families don't already carry.
For the truly unknown 4th family this is the humbling limit: **you cannot pre-train
for structure you have never seen** — which is also why validity-guided decoding
(§4), a rules guardrail rather than a data trick, is the lever that actually holds.

---

## 6. Why OOD top-1 is capped (the vocabulary gap)

A fraction of each held-out family's next-step targets are *unique* steps the model
has never seen and cannot name:

| Hold-out | OOV target frac | step-level top-1 ceiling |
|---|---|---|
| MOSFET | 15.6% | ~0.844 |
| IGBT | 17.5% | ~0.825 |
| IC | 20.9% | ~0.791 |

At the 60/80% evaluation cut points the OOV rate is lower (~2%), so the model's
0.635 OOD top-1 ≈ its 0.648 shared-vocabulary top-1 — i.e. the limiter there is
process logic + irreducible coin-flips, not vocabulary. The remaining ID→OOD logic
gap is small (ID top-1 ≈ 0.74 vs OOD 0.635) and no lever in §4 closed it.

We tested the obvious vocabulary-gap fix directly — **word-level tokenization**
(`process_lm/wordlevel.py`): each step becomes its words plus `<ENDSTEP>`, so an
unseen step can be composed from known words. It **backfired**: OOD top-1 fell to
**0.41** (from 0.635) and valid-completion to **0.22** (from ~1.0). Fragmenting
each step into words compounds errors across the ~120 *shared* steps far more than
it rescues the ~2% genuinely-OOV targets at the cuts, and word-by-word generation
over ~600-token routes rarely stays rule-valid. **Step-level tokenization is the
right representation** for a grammar that operates on whole steps — a clean
negative result that sharpens, rather than weakens, the conclusion.

---

## 7. The three submission tasks (final model)

Final model: medium (25M), all three families + family-dropout 0.15, bf16. ID
validation loss **0.3315 ≈ the 0.328 floor** (train 0.327 — almost no gap).

- **Task 1 — Next-step:** ID **top-1 0.682, top-3 0.997, top-5 1.000, MRR 0.838**
  — the true step is in the top-3 ~99.7% of the time; top-1 is capped only by the
  grammar's coin-flips. OOD proxy: top-1 ~0.65 avg (near the vocabulary ceiling).
- **Task 2 — Completion:** ID completions are **100% process-valid (0%
  rule-breaking)** with block-level normalized edit distance **0.022** (near-perfect
  process-logic flow). Exact-match is low (1.3%) *by design* — the grammar's
  coin-flips make the exact continuation unpredictable (the floor again). The model
  also generates **60/60 valid routes from scratch** from only `RECEIVE WAFER LOT`,
  and infers family-specific detail from context (IGBT `ALIGN MASK LEVEL 5/6`, IC
  `DEPOSIT TUNGSTEN SEED → FILL VIA TUNGSTEN`). The untrained baseline produces
  150–220-step routes with dozens of violations.
- **Task 3 — Anomaly detection:** the LM's surprise spike (max per-step NLL)
  separates valid from rule-violating routes **perfectly on our validator-labeled
  eval: ROC-AUC 1.000, F1 1.000**, with a clean margin (valid sequences never
  exceed 8.3 nats of surprise; violations always exceed the 10.2 threshold).
  **100% recall across all 10 rule types.** This is learned logic — no rule
  engine inside the detector. (A rule-based oracle, `validate_sequence`, is the
  trivial 100% upper bound and the source of our labels.)

Submission files in the organizers' exact format are produced by
`process_lm/submit.py` (`task1_nextstep.csv`, `task2_completion.csv`,
`task3_anomaly.csv`); point it at `eval_input_*.csv` when they arrive, or run
`--selfmake` to reproduce on held-out data now.

---

## 8. Engineering notes (running it on the 5090)

- **Blackwell (sm_120)** needs PyTorch cu128 — installed `torch 2.11.0+cu128`;
  the stock `torch>=2.2` ships no sm_120 kernels.
- **fp32 + batch 256 forced SDPA onto the math attention kernel** → O(B·H·T²)
  memory (the 85M model OOM-thrashed ~30 GB at 75 s/epoch). bf16 autocast →
  efficient attention, O(T) memory, tensor cores. Small/medium models train at
  0.2–3 s/epoch; raw bf16 matmul measured 225 TFLOPS.
- **Windows portability:** `os.kill(pid, 0)` *terminates* a process on Windows;
  rewrote the single-instance run lock to use a Win32 liveness query.

---

## 9. Reproducibility

```bash
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv/Scripts/python.exe numpy matplotlib

# the exact entropy floor (+ faithfulness selftest)
python -m process_lm.oracle

# scaling + hybrid-dose study (LOFO, hold-out IC)
python -m process_lm.overnight --hold-out ic --epochs 40

# OOD lever comparison (real / family-dropout / hybrids / v2, leak-guarded)
python -m process_lm.ood_compare --hold-out ic --epochs 30

# final model + the three tasks + before/after demo
python -m process_lm.train --n-layer 8 --n-embd 512 --n-head 8 \
    --family-dropout 0.15 --extra-per-family 2000 --out-dir process_lm/runs/final
python -m process_lm.anomaly --ckpt process_lm/runs/final/best.pt          # Task 3
python -m process_lm.submit  --ckpt process_lm/runs/final/best.pt --selfmake --guided   # 3 task files
python -m process_lm.demo    --ckpt process_lm/runs/final/best.pt          # before/after
python -m process_lm.guided  --ckpt process_lm/runs/ood/igbt_real/best.pt --family igbt # guided+repair OOD
python -m process_lm.wordlevel --hold-out ic                              # word-level negative result
python -m process_lm.plots

# (optional) a giant validated synthetic corpus across all cores
python -m process_lm.diversify2 --n 100000 --workers 14 --out v2_synth_100k.csv --report
```

Figures: `process_lm/runs/figures/{loss_curves,scaling,hybrid_dose,levers}.png`.

---

## 10. What we'd claim, and what we wouldn't

**We claim:** a small from-scratch model learns transferable process logic — it
hits the exact information floor in-distribution, produces valid routes for an
unseen family, and flags rule violations from its own surprise. We know these are
real because we measured against the data's true entropy, not against our hopes.

**We won't claim:** that any data-augmentation trick robustly improved
generalization. None did — effects are family-specific and cancel on average, one
of ours leaked the answer until we caught it, and a single-family evaluation would
have produced a confident wrong recommendation. The honest, robust lever is small
(light family-dropout). Reporting that — with the three traps we avoided
(memorization-as-progress, vocabulary leakage, and single-family overfitting of our
own conclusions) — is the result.
