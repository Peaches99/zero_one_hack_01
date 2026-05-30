# Flagship checkpoint

`best.pt` — 25.6M-param decoder-only GPT (8 layers, 512-dim, 8 heads, block 256,
vocab 206), trained from scratch on all three families (~3k real routes + ~6k
generated from the grammar), 30 epochs, dropout 0.1, family-dropout 0.15.
Final train loss 0.327 / val loss 0.336 — on the proven 0.328-nat entropy floor.

## Files
- `best.pt` — weights + `config` + training `args` (load with `process_lm.predict.load_model`)
- `tokenizer.json` — step + family vocab (`process_lm.tokenizer.Tokenizer.load`)
- `train_log.csv` — per-epoch train/val loss

## Reproduce
```
python -m process_lm.train --out-dir process_lm/runs/final \
  --extra-per-family 2000 --family-dropout 0.15 --epochs 30 \
  --n-layer 8 --n-embd 512 --n-head 8 --batch-size 256 --lr 3e-4 --seed 0
```

## Score (official metrics)
```
python -m process_lm.submit --ckpt submission/model/best.pt \
  --valid <eval_input_valid.csv> --anomaly <eval_input_anomaly.csv> --out-dir submission/official
python tracks/industrial-infineon/participant_files/eval_metrics.py --task anomaly \
  --ground-truth submission/official/gt_forbidden.csv \
  --predictions submission/official/task3_anomaly.csv \
  --valid-supplement submission/official/gt_valid.csv
```
