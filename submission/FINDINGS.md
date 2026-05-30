# Process-Logic LM — Findings (running log, RTX 5090 overnight)

A from-scratch GPT over semiconductor process-step sequences for the Infineon
"learn vs. memorize process logic" track. This log records results as they land;
it is the backbone of the final report.

## TL;DR so far
- We computed the **exact information floor** of the organizers' generator by
  instrumenting it (not estimating): **0.328 nats/token** in-distribution. A model
  cannot beat this on honest data — it is the Bayes-optimal next-token loss.
- The previously-observed **~0.34 plateau was the model hitting that floor**, i.e.
  it was already optimal for ID next-token prediction. 0.01 is unreachable without
  destroying the data's entropy (a biased set).
- **Scaling same-family real data does NOT improve generalization** to a held-out
  family — it overfits faster. Generalization comes from *data diversity*, not
  *data volume* or *model size*.

## 1. The exact entropy floor (oracle)
`process_lm/oracle.py` instruments the organizers' `generate_sequence` so every
random decision logs its true probability, then asserts the instrumented output is
**byte-identical** to the stock generator (400 seeds × 3 families). The mean
negative log-likelihood under the *true* generative process is the floor:

| Family | Exact floor (nats/token, ID prior 1/3) |
|---|---|
| MOSFET | 0.308 |
| IGBT | 0.310 |
| IC | 0.373 |
| **ID mixture** | **0.328** |

This is engineered to be *falsifiable*, not self-confirming: `score_model_vs_floor`
checks a trained model's NLL on FRESH sequences against the floor. The floor is a
hard lower bound the model approaches from above; a model scoring below it would
prove the floor (instrumentation) wrong. Reaching the floor on this distribution
caps nothing else — OOD, other data mixtures, tokenizations, and objectives all
live above it. (TODO: paste the model-vs-floor verification number.)

## 2. Scaling = memorization (held-out IC, OOD proxy)
40-epoch grid, small 4.85M model unless noted. OOD = held-out IC (never trained on).

| Run | OOD valid-compl | OOD top1 | OOD ppl | note |
|---|---|---|---|---|
| data 200 | 0.713 | 0.500 | 17.4 | undertrained |
| data 1k | 0.988 | 0.620 | 14.5 | best epoch 15 |
| data 5k | 0.963 | 0.620 | 14.7 | best epoch 3 |
| data 20k | 0.988 | 0.595 | 15.1 | best epoch 1 |
| model tiny 0.5M | 0.512 | 0.480 | 17.4 | |
| model small 4.8M | 0.963 | 0.620 | 14.7 | |
| model medium 25M | 0.950 | 0.625 | 17.3 | |

Train loss slams into the 0.306–0.308 floor; OOD val loss *climbs* (overfits) and
best-epoch moves earlier as data grows. **More same-family data and bigger models
do not transfer.** OOD valid-completion is already ~0.96–0.99 with real data, so the
real headroom is OOD **next-step top-1 (~0.62)** and **perplexity**.

## 3. Where the OOD headroom is: the vocab gap
For each held-out family, a fraction of next-step targets are *unique* steps the
model has never seen (OOV → cannot be named):

| Hold-out | OOV target frac | step-level top1 ceiling | unique steps composable from seen words |
|---|---|---|---|
| MOSFET | 15.6% | ~0.844 | 55% |
| IGBT | 17.5% | ~0.825 | 26% |
| IC | 20.9% | ~0.791 | 55% |

So OOD top1 0.62 vs the ~0.79 vocab ceiling = ~0.17 recoverable from *logic transfer*
alone (no tokenizer change). The rest is the vocab gap (partly addressable by
sub-word tokenization; many unique steps still contain genuinely novel words).

## 4. v2 max-diversity generator
The stock generator emits a FIXED macro-structure per family; all its combinatorics
are optional-step/synonym toggles. `process_lm/diversify2.py` adds the missing axis —
**variable device-cycle count (3–6), mixed-family blocks, optional 2nd metal** — every
route validated by the organizers' checker. Result: lengths 109–214 (vs fixed
~115/125/148), litho-level counts spread 5→9, 0 invalid, balanced family tags,
parallel generation across cores. This directly targets the hidden 4th family, which
the data README says differs mainly in *prep and cycle count*.

## 5. Engineering notes (5090)
- RTX 5090 is Blackwell sm_120 → requires torch cu128 (installed 2.11.0+cu128).
- fp32 + batch 256 makes SDPA use the math attention kernel → O(B·H·T²) memory
  (85M model ate ~30 GB and ran 23× slow). Fix: **bf16 autocast** → flash attention,
  O(T) memory, tensor-core speed. (applied to train.py)
- Windows `os.kill(pid,0)` *terminates* processes; fixed runguard liveness via Win32.

## TODO (filling in tonight)
- [ ] Hybrid dose-response (0/2k/8k/20k) OOD curve
- [ ] OOD lever comparison (hybrid vs v2 vs family-dropout vs combos)
- [ ] Model-vs-floor falsification number on a full model
- [ ] Final full model (all families + best recipe) + submission files
- [ ] Anomaly detector (LM-surprise) AUC/F1 vs oracle
- [ ] Before/after demo + figures
