# process_lm

A small from-scratch GPT over semiconductor process-step sequences. One
autoregressive model serves the Infineon track tasks:

- **Next-step prediction** — rank the next-token distribution (Task 1)
- **Sequence completion** — roll the model forward to `SHIP LOT` (Task 2)
- **Anomaly detection** — *(coming next)* watch where the model is surprised (Task 3)

Each fab step string (`"DEPOSIT GATE OXIDE"`) is a single token. Family
(`MOSFET`/`IGBT`/`IC`) is a conditioning token.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for environments:

```bash
uv venv
uv pip install -r process_lm/requirements.txt
source .venv/bin/activate
```

Runs on Apple Silicon (MPS), CUDA, or CPU — auto-detected. If MPS hits an
unsupported op, fall back with:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Run (from the repo root)

```bash
# Train (~5M params; minutes on an M-series Mac / seconds on a GPU)
python -m process_lm.train --epochs 15

# Self-score Task 1 (Top-1/3/5 + MRR) on the held-out split
python -m process_lm.predict --mode nextstep

# Self-score Task 2 (exact match / edit distance / token accuracy)
python -m process_lm.predict --mode completion --limit 50

# Intrinsic check: % of generated routes that pass the provided validator
python -m process_lm.predict --mode sanity --limit 50

# One concrete next-step example (for the demo)
python -m process_lm.predict --mode demo
```

Artifacts land in `process_lm/runs/v1/`: `best.pt`, `last.pt`, `tokenizer.json`,
`train_log.csv` (loss curves).

## Layout

| File | Role |
|---|---|
| `tokenizer.py` | step string ↔ id, special + family tokens |
| `data.py` | BOM-safe CSV loader, dataset, right-pad collate, train/val split |
| `model.py` | nanoGPT-style decoder (`GPT`, `GPTConfig`) |
| `metrics.py` | Task 1 & 2 metrics, reimplemented from the eval spec |
| `train.py` | training loop (bf16 AMP), checkpoints, loss log; `--add-hybrids/--add-v2/--family-dropout` |
| `predict.py` | next-step / completion / sanity / demo readouts |
| `oracle.py` | **exact entropy floor** — instruments the real generator (byte-identical selftest) + falsification check |
| `diversify.py` / `diversify2.py` | hybrid pseudo-families / v2 variable-cycle generator (leak-guarded) |
| `overnight.py` / `ood_compare.py` | scaling + hybrid-dose grid / leak-guarded OOD lever study |
| `lofo_analysis.py` | leave-one-family-out ID→OOD gap decomposition (logic vs vocab, by position) |
| `anomaly.py` | Task 3 — LM-surprise anomaly detector + validator-labeled eval |
| `guided.py` | validity-guided decoding + grammar repair (100% valid OOD completions) |
| `wordlevel.py` | word-level tokenization experiment (a clean negative result) |
| `blocklevel.py` | process-logic-flow LM — **0.0043 val loss** (the legitimate < 0.01) |
| `submit.py` / `demo.py` / `plots.py` | 3 submission files / before-after demo / figures |

## Notes

- Default split holds out 100 sequences/family, mirroring the eval set.
- `--family-dropout 0.15` masks the family token during training to set up
  graceful degradation on the hidden 4th family (Task 4 / OOD).
- When the organizers' `eval_input_*.csv` drop, point a thin writer at
  `predict_next` / `complete` to emit `nextstep.csv` and `completion.csv`.
