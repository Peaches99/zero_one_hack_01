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
git clone https://github.com/Peaches99/zero_one_hack_01.git
cd zero_one_hack_01
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # PyTorch; CUDA GPU optional (CPU works for inference)
```

### Run
```bash
# 1. Reproduce the headline numbers with the shipped model (no training)
python -m process_lm.oracle                       # proves the 0.328-nat entropy floor
python -m process_lm.predict --ckpt submission/model/best.pt --mode nextstep   # Task 1: Top-1/3/5, MRR
python -m process_lm.block_gap --queries 150      # Task 2: completion vs the Bayes ceiling

# 2. Score Task 3 with the official scorer (files already in repo)
python tracks/industrial-infineon/participant_files/eval_metrics.py --task anomaly \
  --ground-truth submission/official/gt_forbidden.csv \
  --predictions submission/official/task3_anomaly.csv \
  --valid-supplement submission/official/gt_valid.csv

# 3. Regenerate the submission CSVs from the official eval inputs
python -m process_lm.submit --ckpt submission/model/best.pt \
  --valid tracks/industrial-infineon/participant_files/eval_input_valid.csv \
  --anomaly tracks/industrial-infineon/participant_files/eval_input_anomaly.csv \
  --out-dir submission/out --completion-mode mbrblk --anomaly-mode hybrid

# 4. Demo — baseline vs trained on identical inputs
python -m process_lm.demo --ckpt submission/model/best.pt
```


### Retrain (Optional)
```bash
python -m process_lm.train --out-dir process_lm/runs/final \
  --extra-per-family 2000 --family-dropout 0.15 --epochs 30 \
  --n-layer 8 --n-embd 512 --n-head 8 --batch-size 256 --lr 3e-4 --seed 0
```

### Regenerate all figures (Optional)
```bash
python -m process_lm.plots
```

Code: [`process_lm/`](process_lm/) ([module README](process_lm/README.md)) · self-eval
outputs: `extras/results/` · figures: `process_lm/runs/figures/`.
