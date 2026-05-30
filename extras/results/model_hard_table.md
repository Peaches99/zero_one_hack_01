# Hard table — how the `final` model behaves (and where headroom is / isn't)

All numbers on **fresh, unseen** sequences from the organizers' generator (deduped
against training), scored with the **official `eval_metrics.py`** functions. Goal:
find where real headroom exists before spending compute, and not fool ourselves.

## 1. Loss vs the proven floor — we are AT the floor

| model | train recipe | best val loss | vs floor 0.328 |
|---|---|---|---|
| `final` | 25M, fam-dropout 0.15, +2k/fam | 0.331 | +0.003 |
| `id_opt` | 25M, fam-dropout 0.0, +8k/fam | **0.3289** | +0.001 |

The exact Bayes floor (instrumented generator, byte-identical selftest) is **0.328
nats/tok**. Both models sit on it. A model at the cross-entropy floor has learned the
true conditional distribution → its Top-1/MRR **are** the Bayes-optimal accuracy
ceilings. So large next-step gains should be impossible; confirmed below.

## 2. Next-step (Task 1) — at ceiling

| condition | Top-1 | Top-3 | Top-5 | MRR |
|---|---|---|---|---|
| `final`, random cut [0.5,0.9], n=1000 | 0.731 | 0.990 | **1.000** | 0.861 |
| `id_opt`, random cut, n=1000 | 0.731 | 0.996 | 1.000 | 0.861 |
| `final`, official 60/80 cuts, n=2000 | 0.696 | 0.995 | 1.000 | 0.845 |

Per family (random cut): mosfet 0.716 / igbt 0.754 / ic 0.724. Top-5 is **maxed
(1.000)** — the right step is always in the top 5. Top-1 (~0.70–0.73) is capped by
the grammar's interchangeable synonyms + optional steps (the same coin-flips that set
the 0.328 floor). **Two independent recipes give the identical Top-1 → this is the
ceiling.**

## 3. Completion (Task 2) — decoding doesn't move it

`final`, official 60/80 cuts, n=2000 (greedy = what we submit):
Block-acc **0.705**, NED **0.226**, Token-acc **0.421**, Exact 0.004, **Valid 1.000**
(600/600 reach SHIP LOT). Exact/NED are stochasticity-limited (billions of valid
endings; we emit a different legal one). Block-acc carries the structural signal.

Decoding-strategy paired sweep (n=300, random cut) — delta vs greedy:

| strategy | dBlock-acc | dNED | verdict |
|---|---|---|---|
| guided (validity veto) | +0.0025 | +0.0016 | noise |
| beam (width 5, len-norm) | +0.0062 | −0.0027 | ~1 SE, noise |
| MBR (k=8, consensus) | −0.0172 | −0.0053 | hurts block-acc |
| ensemble-completion (final+full) | −0.071 | — | hurts (weak 2nd model) |

## 4. Inference-time levers on the fixed model — all flat/negative

| lever | metric | result |
|---|---|---|
| grammar-masked next-step | Top-1 | **+0.000** (model never ranks an illegal step #1) |
| ensemble next-step (final+id_opt, 2 strong) | Top-1 | **−0.009** selection / **−0.011** confirm |
| temperature | Top-1 | irrelevant (argmax invariant to T) |

## 5. Training-side lever tested so far

`id_opt` (drop family-dropout, 4× data): val loss ↓ to 0.329 but Top-1 **identical**
(0.731), completion **marginally worse** (Block 0.653 vs 0.661). No ID gain.

## 6. Eval noise (so we don't chase ghosts)

Single-model Top-1 swings **0.704 (seed 99991) → 0.670 (seed 7)** at n=400. So the
eval-set SD is ~0.02–0.03; **any "improvement" under ~0.03 is noise.** This is why the
sweep uses a fixed selection set AND a separate confirmation seed.

## 7. Official organizer eval (real set, their scorer)

- Anomaly: Accuracy/Precision/Recall/F1/ROC-AUC **1.000**, Rule Attribution **1.000**.
- Completion: 600/600 valid routes → SHIP LOT.
- Next-step: submitted (answer key organizer-held); local estimate Top-5 ~1.0.

## 8. Conclusion so far

Every inference lever and the no-dropout/more-data retrain leave the in-distribution
metrics unchanged within noise — consistent with the model being **Bayes-optimal at
the 0.328 floor**. The train sweep below settles the model/data side.

## 9. Full train sweep — 19 configs, official 60/80 cuts, n=3000, Top-1

| lever swept | range of Top-1 | trend |
|---|---|---|
| **size** 4L/256 → 12L/768 | tiny 0.689 · small 0.688 · med 0.678 · large 0.687 · xl 0.688 | **none** (tiny = xl) |
| **data** +1k → +40k/fam | 0.685 · 0.683 · 0.684 (8k=0.678) | none |
| **epochs** 20 / 40 / 60 | 0.678 · 0.667 · 0.684 | none (40 slightly worse) |
| **lr** 3e-4 / 6e-4 / 1e-3 | 0.678 · 0.680 · 0.672 | none |
| **dropout** 0 / 0.1 / 0.2 | 0.677 · 0.678 · 0.690 | within noise |
| **family-dropout** 0 / 0.15 | 0.678 · 0.688 | none (no ID penalty) |
| **5 seeds** (same arch) | 0.678 · 0.681 · 0.684 · 0.682 · 0.689 | spread 0.011 = pure noise |

All 19 configs: **Top-1 0.667–0.690, val loss ~0.328.** The whole spread (0.023) is
inside the same-arch seed band (0.678–0.689) — i.e. **no config beats another beyond
noise.** A 4-layer model equals a 12-layer model. Decisive: the ceiling is the data's,
not the model's.

## 10. Ensembles — fail the confirmation test (the anti-self-deception guard working)

| ensemble | selection (seed 99991) | confirm (seed 7) | verdict |
|---|---|---|---|
| 2-model (final + id_opt) | −0.009 Top-1 | −0.011 Top-1 | hurts |
| 5-seed (same arch) | **+0.0085** Top-1 | **−0.0050** Top-1 | **noise** (gain doesn't replicate) |

The 5-seed "gain" on the selection set is exactly the selection-bias trap; the
confirmation seed refutes it. No verified ensemble gain.

## 11. Completion on the biggest model — also at ceiling

`size_xl` (12L/768): Block-acc 0.668 vs `final` 0.661 (+0.007, within noise);
NED 0.224 vs 0.226. No completion headroom from scale.

## VERDICT

Across 5 decoding strategies, 3 next-step re-rankers, 2 ensembles (with confirmation),
and a 19-config train sweep (size/data/epochs/lr/dropout/family-dropout/seeds), **no
approach yields a verifiable improvement on any in-distribution metric.** Anomaly is
1.000 (maxed), next-step Top-5 is 1.000 (maxed), and Top-1 / MRR / completion sit at the
ceiling set by the proven 0.328-nat entropy floor (the grammar's irreducible synonym +
optional-step coin-flips). **The model is Bayes-optimal in-distribution** — a result we
proved by trying to break it, not by assuming it.

## 12. OOD (Task-4 proxy) — guided helps validity, not the scored metrics

LOFO models on their held-out family, n=100/fam, 60/80 cuts:

| decode | Block-acc | NED | Valid |
|---|---|---|---|
| greedy | 0.497 | 0.436 | 0.805 |
| guided+repair | 0.485 | 0.451 | **0.998** |
| delta | −0.012 | +0.015 | **+0.193** |

Guided decoding's known win (validity 0.80→1.00) is real, but it **trades a little
scored-metric match for legality** — it does not raise Block-acc/NED. OOD scored metrics
are far below ID (Block 0.50 vs 0.70) because the model cannot predict an unfamiliar
family's unique steps — a fundamental limit, not a fixable gap. **No verifiable
scored-metric gain exists on OOD either.**

## Final verdict

Across **5 decoding strategies, 3 next-step re-rankers, 2 ensembles (confirmation-
tested), a 19-config train sweep, and an OOD plain-vs-guided test**, no approach
produces a verifiable improvement on any scored metric (ID or OOD). The model is at the
information-theoretic ceiling everywhere it can be — Bayes-optimal in-distribution, and
fundamentally vocabulary-limited (not fixable) out-of-distribution. The honest, hard-won
result is **provable optimality**, established by exhaustively trying to beat it.
