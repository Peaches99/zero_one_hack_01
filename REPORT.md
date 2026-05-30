# GiveMeCompute — Infineon Industrial Track

## Team

- **Thomas Boigner** — ML
- **Nicolas Safarik** — ML

**Track:** Industrial AI (Infineon)

## TL;DR
We built a decoder only GPT from scratch with 25M parameters. It predicts next individual Process steps (Tokens), predicts whole sequences, and can detect anomalies/errors/issues in existing Blocks/Sequences.


## Problem
We very quickly identified that the synthetic data itself was going to be a problem stemming from the fact that part of it gets generated completely randomly with no further logic behind it. Thus we knew that was going to be our mathematical ceiling for exact token/process step matching.
This ended up limiting us to the following exact token matching results.

Top-1: 0.682
Top-3: 0.997
TOP-5: 1.000

## Approach

Due to the mathematical ceiling we stopped concentrating on exact token matching and rather continued on model generalization performance so that we could reliable and always generate valid processes and reliably detect anomalies instead of harping on individual "perfect" matching. We specifically tried to push Out of Distribution results as far as we could since thats what we found most interesting in this challenge. 

### Architectural Decision 1 - Tokenizer
Based on the requirements and the given vocabulary it was clear to us that we could build our custom tokenizer that would treat each step as 1 token.

### Architectural Decision 2 - Custom Decoder GPT
Since this task seemed computationally trivial given the relatively simple task and limited vocabulary + data we decided to use a custom nanoGPT Style Decoder-only Transformer model that would predict the previously mentioned Tokens. Since each process is also very short and smaller than 256 Steps that gave us our tiny context window size. We used Pytorch for this.

### Where it runs
Since this is computationally not heavy at all we ran it on our laptops as well as ran training on a private RTX 5090

The laptops hardware was still more than enough and consisted of a M1 Macbook pro 1 windows laptop with a 1650ti. 

The training vram requirements were at most 10gb and using the 5090 1 training run at 25.6M parameters took about 10 minutes. Running the model takes about 115MB and it runs fine on cpus.

## How to run it

Run everything from the repo root. The trained checkpoint ships in the repo (`submission/model/best.pt`), so training is optional.

```bash
# 1. Setup
git clone https://github.com/Peaches99/zero_one_hack_01.git
cd zero_one_hack_01
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # PyTorch; CUDA GPU optional (CPU works for inference)

# 2. Reproduce the headline numbers with the shipped model (no training)
python -m process_lm.oracle                       # proves the 0.328-nat entropy floor
python -m process_lm.predict --ckpt submission/model/best.pt --mode nextstep   # Task 1: Top-1/3/5, MRR
python -m process_lm.block_gap --queries 150      # Task 2: completion vs the Bayes ceiling

# 3. Score Task 3 with the official scorer (files already in repo)
python tracks/industrial-infineon/participant_files/eval_metrics.py --task anomaly \
  --ground-truth submission/official/gt_forbidden.csv \
  --predictions submission/official/task3_anomaly.csv \
  --valid-supplement submission/official/gt_valid.csv

# 4. Regenerate the submission CSVs from the official eval inputs
python -m process_lm.submit --ckpt submission/model/best.pt \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out-dir submission/out --completion-mode mbrblk --anomaly-mode hybrid

# 5. Demo — baseline vs trained on identical inputs
python -m process_lm.demo --ckpt submission/model/best.pt

# 6. (Optional) retrain the flagship from scratch — ~1 hour on one CUDA GPU
python -m process_lm.train --out-dir process_lm/runs/final \
  --extra-per-family 2000 --family-dropout 0.15 --epochs 30 \
  --n-layer 8 --n-embd 512 --n-head 8 --batch-size 256 --lr 3e-4 --seed 0

# 7. (Optional) regenerate all figures
python -m process_lm.plots
```

## Results

**Task 1: Next-Step Prediction** (n=600 held-out, 60/80 cuts):

| Top-1 | Top-3 | Top-5 | MRR |
|---|---|---|---|
| 0.682 | 0.997 | 1.000 | 0.838 |

Top-1 0.682 ≈ the Bayes ceiling

The Bayes ceiling is the mathematical limit of accuracy based on the synthetic dataset. In essence as much as we can predict the data until were just trying to predict the random number generator (Which funnily enough we were thinking of doing as a joke since pythons random has been reverse engineered). The headline here is though, that we achieved almost exactly as much as is mathematically possible, are almost always at 99.7% in the top 3, and definitely within Top 5. This is a result we are proud of even though in production id hope for a higher Top 1.

**Task 2: Sequence Completion** (official metric functions)):

| | Block-level Acc | Token Acc | Norm. Edit Dist ↓ | Exact | Process-valid |
|---|---|---|---|---|---|
| Overall | **0.71** | 0.43 | 0.22 | ~0.00 | **100%** |
| 60% cut | 0.47 | 0.31 | 0.24 | 0.00 | **100%** |
| 80% cut | **0.93** | 0.53 | 0.21 | 0.00 | **100%** |

Block-level 0.71 is the **Bayes-optimal ceiling** (oracle 0.7105 — proven, not asserted). 

We argue with the given conditions and rules it is not possible to get an exact match much higher as guessing 20-40% of a complete process would be the equivalent of perfectly guessing over 10 50/50 chances in a row which is practically impossible at 0.0009765625% for 10 steps.

Per-family completion (Block-acc, 60/80): MOSFET 0.46 / 0.97 · IGBT 0.48 / 0.88 · IC 0.45 / 0.94.

*Task 3: Anomaly Detection** (official scorer, all 987 inputs):

| Binary Acc | Precision | Recall | F1    | ROC-AUC | Rule Attribution |
| ---------- | --------- | ------ | ----- | ------- | ---------------- |
| 1.000      | 1.000     | 1.000  | 1.000 | 1.000   | 1.000            |

Confusion matrix **TP 387 · FP 0 · FN 0 · TN 600**

All 10 rule types detected at 100%. (Detection is pure-model surprise; the validator names the rule as disclosed hybrid. Pure-model attribution alone is 0.58.)

**Baseline comparison.**
Untrained model (same architecture, random init): next-step Top-1 ≈ chance, and free-running generation produces rule-breaking routes (161 violations on a sample MOSFET route in the demo). 

Trained flagship: Top-1 0.682, **100% process-valid completions, 0 violations**. 

**OOD** On a held-out family, block structure transfers (Block-acc 0.50) while exact tokens collapse (0.15 — vocabulary ceiling). Diversity-scaling experiment: at fixed data volume, going 1→2 training families lifts OOD Block-acc **+0.16 (+0.32 at the 80% cut)**, so our 3-family model should generalize to the hidden 4th family better than that proxy.

---

## What worked

- **Measuring the entropy floor.** Instrumenting the generator to get the exact 0.328-nat limit proved that we are at the optimal. It saved us a lot of time unsuccesfully trying to optimize exact token predictions and let us work on other generalization tasks.
- 
- **Anomaly detection from the model's own surprise.** Per-step likelihood spikes flag rule violations with zero false positives across 987 sequences (perfect F1/AUC). No separate classifier — the language model's uncertainty *is* the detector. We didnt need a seperate classifier as the models uncertainty is functionally our detector. By running it over a sequence and seeing if it suddenly gets uncertain we know the previus token must be wrong.

- **Validity-guided decoding.** Letting the rule validator veto illegal next-steps gives 100% process-valid completions while the model supplies the content — guardrail + knowledge, cleanly separated.

- **Diversity > scale for generalization.** The controlled finding that training-family *diversity* lifts OOD while model/data size do nothing is the most important finding, and it directly answers the question on if the model understands the data or just memorized it.

---

## What didn't work

- **Scaling anything.** A 19-config sweep (4-layer → 12-layer, 1k → 40k sequences, 20 → 60 epochs): Top-1 flat at 0.667–0.690, all within seed noise. A 4-layer model equals a 12-layer one. Bigger/more/longer does not help, it just overfits faster.
-
- **Word-level tokenization.** Tried decomposing steps into words to handle unseen-family vocabulary. Backfired hard: OOD Top-1 0.41 vs 0.635, valid-completion 0.22 vs 1.00 — compounding word errors outweighed the small OOV rescue. We tried this before we realized the learning that we are at the mathematical limit for exact matching.
- 
- **Hybrid / max-diversity data augmentation.** Helped one family (MOSFET valid 0.76→0.96) while tanking another (IGBT 0.66→0.48), and one variant leaked held-out vocabulary into training — a fake "win" that was actually data contamination.

- **Grokking.** Trained tiny datasets for 12,000 epochs trying delayed generalization. Didn't happen, the model reaches its best held-out loss by epoch ~13, then overfits. The grammar is too easy to learn for grokking to happen.


## What you'd do with another 36 hours

Honestly with this data. Nothing. The data here is the hard limit. If a rule allowed copyrighted data such as real Infineon FAB level data we would have more to work with. Maybe run 100 Claude/GPT/Gemini Agents to Bruteforce all other paths that we didnt think of ourselves. But that would cost more than what its worth we believe.


## Track-specific deliverables

Each track has additional required outputs beyond this report. Confirm yours are present:

### ⚙️ Industrial AI (Infineon)
- [] Eval submission files in `extras/results/`:
  - `nextstep.csv` (Task 1 format)
  - `completion.csv` (Task 2 format)
  - `anomaly.csv` (Task 3 format)
- [ ] Training artifacts: checkpoint(s), training logs, loss curves
- [ ] Scores from `eval_metrics.py` on all three tasks, with per-family breakdown
- [ ] Demo shows baseline vs. trained output on identical inputs


## Credits & dependencies

- **Open-source libraries used** (with versions): {list}
- **Pre-trained models used**: None
- **External APIs called**: None
- **AI coding assistants used during the hackathon**: Claude Code + Claude Deep Research
- **Datasets**: Industrial track synthetic dataset
