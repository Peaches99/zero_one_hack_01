# Overnight run on the RTX 5090

Goal: come back to **results**, not just a checkpoint — the Level-3 scaling study
and the hybrid dose-response, all scored on the leave-one-family-out (OOD) proxy.

## 0. One-time setup on the 5090 box

```bash
git pull
uv venv && uv pip install -r process_lm/requirements.txt
source .venv/bin/activate
python -c "import torch; print('cuda', torch.cuda.is_available())"   # must print True
```

The code auto-detects CUDA — no flags needed. The MPS-specific env var is
harmless on CUDA. The single-instance lock still applies (one training run at a
time per machine; `--force` to override).

## 1. Smoke test first (2-3 min — proves the harness on this box)

```bash
python -m process_lm.overnight --hold-out ic --smoke
```

Expect two quick runs and `[done:...]` lines with OOD numbers. If that works, the
real grid will too.

## 2. Launch the overnight grid

```bash
nohup python -m process_lm.overnight --hold-out ic --epochs 25 --batch-size 256 \
  > process_lm/runs/overnight/console.log 2>&1 &
```

12 runs across three studies (each trains + self-evaluates, ~minutes each on a
5090; the whole grid is well under an hour, so "overnight" is comfortable — you
can raise `--epochs` to 40+ or add `--hold-out mosfet` / `igbt` as extra grids):

1. **Data scaling** — 200 / 1k / 5k / 20k real sequences
2. **Model scaling** — tiny / small / medium / large at fixed data
3. **Hybrid dose** — 0 / 2k / 8k / 20k validated pseudo-families (the OOD study)

Results stream to `process_lm/runs/overnight/results.jsonl` (one row per finished
run — a crash never loses completed work; re-running skips done runs).

### Want a bigger run? Scale data, not just time
The grammar is an infinite validated faucet, so push volume on the 5090:

```bash
# heavier: 50k extra real/family + 40k hybrids, larger model, more epochs
python -m process_lm.train --hold-out-family ic \
  --extra-per-family 50000 --add-hybrids 40000 --hybrid-tag random \
  --n-layer 12 --n-embd 768 --n-head 12 --batch-size 256 --epochs 40 \
  --out-dir process_lm/runs/overnight/big_ic
```

## 3. In the morning

```bash
python -m process_lm.overnight_report
```

Prints three tables + a verdict line. The headline we're after:

- **Data scaling**: does OOD valid-completion rise / perplexity fall with more data?
- **Model scaling**: does ID saturate while the **OOD gap widens** (the
  memorization signature — a great figure for the report)?
- **Hybrid dose**: does OOD valid-completion climb toward 1.0 as we add validated
  hybrid pseudo-families? (Baseline OOD on the Mac was 0.79-0.94.)

Push the `results.jsonl` + `console.log` back so we analyze together:

```bash
git add -f process_lm/runs/overnight/results.jsonl process_lm/runs/overnight/console.log
git commit -m "overnight: 5090 scaling + hybrid-dose results" && git push
```

(Checkpoints stay local — `.gitignore` keeps `*.pt` out of git; only the small
results/log files are force-added.)
```
