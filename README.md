# Process-Logic LM — Industrial AI (Infineon) submission

A from-scratch GPT over semiconductor process routes, benchmarked against the
**exact information-theoretic floor** of the data (computed by instrumenting the
organizers' own generator). It hits that floor in-distribution, scores top-5 1.000 /
100%-valid completions / anomaly AUC 1.000, and — with a model+grammar
validity-guided decoder — produces **100% rule-valid completions for unseen
families**.

➡️ **Jury writeup: [`REPORT.md`](REPORT.md)** · full technical deep-dive:
[`submission/REPORT.md`](submission/REPORT.md) · slides: [`submission/SLIDES.md`](submission/SLIDES.md)

### Setup
RTX 50-series (Blackwell, sm_120) needs the CUDA-12.8 PyTorch build — handled by
`requirements.txt`. The code auto-detects CUDA / Apple MPS / CPU. No API keys, no
external data (the organizers' `tracks/industrial-infineon/` data is sufficient).
```bash
uv venv && uv pip install -r requirements.txt      # or: pip install -r requirements.txt
```

### Run
```bash
python -m process_lm.oracle                          # exact entropy floor (0.328) + faithfulness selftest
python -m process_lm.train --n-layer 8 --n-embd 512 --n-head 8 \
    --family-dropout 0.15 --extra-per-family 2000 --out-dir process_lm/runs/final
python -m process_lm.predict --ckpt process_lm/runs/final/best.pt --mode nextstep        # Task 1
python -m process_lm.submit  --ckpt process_lm/runs/final/best.pt --selfmake     # 3 task files
python -m process_lm.anomaly --ckpt process_lm/runs/final/best.pt                         # Task 3
python -m process_lm.demo    --ckpt process_lm/runs/final/best.pt                         # baseline vs trained
python -m process_lm.plots                                                                # loss/scaling figures
```

Code: [`process_lm/`](process_lm/) ([module README](process_lm/README.md)) · self-eval
outputs: `extras/results/` · figures: `process_lm/runs/figures/`.
