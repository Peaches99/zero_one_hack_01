# Track deliverables — Infineon Industrial AI (GiveMeCompute)

| Deliverable | File |
|---|---|
| Task 1 submission (next-step) | `nextstep.csv` |
| Task 2 submission (completion) | `completion.csv` |
| Task 3 submission (anomaly) | `anomaly.csv` |
| Official `eval_metrics.py` scores, all 3 tasks, **per-family + per-fraction** | `eval_scores.txt` |
| Held-out ground-truth + predictions used for the Task 1–2 scores | `heldout/` |
| Training log (per-epoch train/val loss) | `train_log.csv` |
| Loss curves + result figures | `figures/` (full set in `../../submission/figures/`) |
| Full experiment log (every lever tried) | `model_hard_table.md` |
| **Flagship checkpoint** (25.6M, val 0.336) + tokenizer + card | `../../submission/model/` |

Notes:
- `nextstep/completion/anomaly.csv` are the model's predictions on the **official** eval inputs (`tracks/industrial-infineon/participant_files/eval_input_*.csv`).
- Tasks 1–2 in `eval_scores.txt` are scored on fresh **held-out** routes (the official inputs ship without ground-truth continuations); Task 3 is scored on the official 987 inputs against validator-derived labels.
- Baseline-vs-trained on identical inputs: `python -m process_lm.demo --ckpt submission/model/best.pt`.
