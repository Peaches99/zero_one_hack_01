# Learning vs. Memorizing Process Logic — Infineon Industrial Track

**A from-scratch GPT over semiconductor process-step sequences, benchmarked
honestly against the information-theoretic limit of the data itself.**

Team tooling: a ~5M–25M parameter decoder trained from scratch (no pretrained
base), a custom data generator, an exact entropy oracle, and self-scoring for all
three eval tasks. Everything runs on one RTX 5090; the whole study reproduces in
well under an hour.

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

A legitimate place where ~0.01 *is* reachable: loss restricted to the
**deterministic, rule-forced transitions** (the ~70% of tokens with no random
choice). On those, a good model is essentially perfect; the residual loss lives
entirely in the irreducible coin-flips.

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

## 4. What actually generalizes (OOD lever study, hold-out IC)

Same model, same epochs; only the training-data / regularization strategy varies.
`vUNK` / `t1UNK` use the `<UNK>` family token — the **true proxy for the hidden 4th
family**, which has no known family token.

| config | valid (UNK) | top-1 | shared-vocab top-1 | t1-UNK | ppl |
|---|---|---|---|---|---|
| **real + family-dropout 0.15** | **0.988** | 0.635 | 0.648 | **0.635** | 14.3 |
| real + family-dropout 0.30 | 0.988 | 0.640 | 0.653 | 0.635 | 14.3 |
| real | 0.975 | 0.635 | 0.648 | 0.630 | 14.5 |
| + 8k hybrid pseudo-families | 1.000 | 0.615 | 0.628 | 0.600 | 14.4 |
| + 8k v2 diversity *(leaky)* | 1.000 | 0.595 | 0.602 | 0.605 | 11.4 |

**Finding: the best OOD generalizer is plain real data + a small family-token
dropout.** Family-dropout gives a small, consistent gain on the unknown-family
proxy (it stops the model from binding logic to a family identity it won't have at
test time). Valid-completion is essentially saturated (≈1.0) — the model produces
*legal* routes for an unseen family almost always; the discriminating metric is
next-step top-1, capped near the vocabulary ceiling (§6).

---

## 5. Two traps we caught (the honest part)

**(a) Hybrid pseudo-families hurt OOD.** Mixing blocks across families to make
"Frankenstein" routes was hypothesized to boost generalization. Dose-response says
otherwise: OOD valid-completion fell 0.963 → 0.900 → 0.850 as we added 0 → 2k → 8k
hybrids. An earlier "hybrids help" reading came from a *severely undertrained*
smoke model; properly trained, the clean baseline is already strong and the
hybrids only add distribution-shift noise.

**(b) Our own v2 diversity generator leaked the answer.** v2 (variable cycle
counts, mixed blocks — `diversify2.py`) appeared to cut OOD loss 2.66 → 2.42. But
the training vocabulary jumped 177 → 201: the generic-cycle palette included
**held-out-family-unique steps** (e.g. `IMPLANT N-TYPE`, `MEASURE CD LEVEL 2` are
IC-only). The "win" was vocabulary leakage — perplexity dropped because leaked
steps became predictable, yet **OOD top-1 actually went down**. We added a
pool-vocabulary leak guard; leak-free, v2 provides no top-1 gain. This is exactly
the "don't make a biased set" failure mode, caught by watching the vocab and the
top-1 (not just the loss).

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
gap is small (ID top-1 ≈ 0.74 vs OOD 0.635). Sub-word tokenization is the one lever
with a genuinely higher ceiling (it could compose ~26–55% of unseen steps from
known words); we scoped it as future work after confirming the cleaner levers.

---

## 7. The three submission tasks (final model)

Final model: medium (25M), all three families + family-dropout 0.15, bf16. ID
validation loss **0.3315 ≈ the 0.328 floor** (train 0.327 — almost no gap).

- **Task 1 — Next-step:** strong in-distribution; on the OOD proxy, top-1 0.635
  (near the vocabulary ceiling), valid-completion ≈1.0.
- **Task 2 — Completion:** trained model rolls forward into **valid** routes
  (validator-confirmed), including family-specific detail it must infer from
  context (IGBT `ALIGN MASK LEVEL 5/6`, IC `DEPOSIT TUNGSTEN SEED → FILL VIA
  TUNGSTEN`). Baseline produces 150–220-step routes with dozens of violations.
- **Task 3 — Anomaly detection:** the LM's surprise spike (max per-step NLL)
  separates valid from rule-violating routes **perfectly on our validator-labeled
  eval: ROC-AUC 1.000, F1 1.000**, with a clean margin (valid sequences never
  exceed 8.3 nats of surprise; violations always exceed the 10.2 threshold).
  **100% recall across all 8 tested rule types.** This is learned logic — no rule
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
python -m process_lm.anomaly --ckpt process_lm/runs/final/best.pt
python -m process_lm.submit  --ckpt process_lm/runs/final/best.pt --selfmake
python -m process_lm.demo    --ckpt process_lm/runs/final/best.pt
python -m process_lm.plots
```

Figures: `process_lm/runs/figures/{loss_curves,scaling,hybrid_dose,levers}.png`.

---

## 10. What we'd claim, and what we wouldn't

**We claim:** a small from-scratch model learns transferable process logic — it
hits the exact information floor in-distribution, produces valid routes for an
unseen family, and flags rule violations from its own surprise. We know these are
real because we measured against the data's true entropy, not against our hopes.

**We won't claim:** that data-augmentation tricks improved generalization. They
didn't; one of ours leaked the answer until we caught it. The honest lever was
small (family-dropout). Reporting that, with the trap we avoided, is the result.
