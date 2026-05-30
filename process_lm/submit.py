"""Generate the three submission files in the organizers' exact format.

Task 1  next-step    : EXAMPLE_ID, RANK_1..RANK_5
Task 2  completion   : EXAMPLE_ID, PREDICTED_SEQUENCE   (pipe-joined, steps AFTER the cut)
Task 3  anomaly      : EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE

Reads the organizers' eval_input_valid.csv / eval_input_anomaly.csv when present.
With --selfmake it builds stand-in eval inputs from held-out real sequences so the
whole pipeline runs end-to-end before the official files arrive.

    python -m process_lm.submit --ckpt RUN/best.pt --selfmake --out-dir submission/out
    python -m process_lm.submit --ckpt RUN/best.pt --valid eval_input_valid.csv \
        --anomaly eval_input_anomaly.csv --out-dir submission/out
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

from .anomaly import anomaly_score
from .data import build_records, load_all_families, split_records
from .predict import complete, get_device, load_model, predict_next
from .tokenizer import Tokenizer

_DATA_DIR = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
import generate_sequences as gs  # type: ignore  # noqa: E402


def _read_valid(path):
    """eval_input_valid.csv -> [(example_id, family, frac, partial_steps)]."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip().strip('"'): v for k, v in r.items()}
            rows.append((r["EXAMPLE_ID"], r["FAMILY"].strip().lower(),
                         float(r.get("COMPLETION_FRACTION", 0) or 0),
                         [s for s in r["PARTIAL_SEQUENCE"].split("|") if s]))
    return rows


def _read_anomaly(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip().strip('"'): v for k, v in r.items()}
            rows.append((r["EXAMPLE_ID"], r["FAMILY"].strip().lower(),
                         [s for s in r["SEQUENCE"].split("|") if s]))
    return rows


def selfmake(seed=0, n_per_family=100):
    """Build stand-in eval inputs from held-out real sequences (valid + anomaly)."""
    recs = build_records(load_all_families(_DATA_DIR))
    _, val = split_records(recs, n_per_family, seed)
    rng = random.Random(seed)
    valid_rows, anomaly_rows, anomaly_truth = [], [], {}
    eid = 0
    for fam, steps in val:
        for frac in (0.6, 0.8):
            cut = max(1, int(len(steps) * frac))
            valid_rows.append((f"valid_{eid:04d}", fam, frac, steps[:cut], steps[cut:]))
            eid += 1
    from .anomaly import corrupt
    aid = 0
    for fam, steps in val:
        if rng.random() < 0.5:
            anomaly_rows.append((f"anom_{aid:04d}", fam, steps))
            anomaly_truth[f"anom_{aid:04d}"] = (1, None)
        else:
            c = corrupt(steps, rng)
            if c:
                anomaly_rows.append((f"anom_{aid:04d}", fam, c[0]))
                anomaly_truth[f"anom_{aid:04d}"] = (0, c[1])
            else:
                anomaly_rows.append((f"anom_{aid:04d}", fam, steps))
                anomaly_truth[f"anom_{aid:04d}"] = (1, None)
        aid += 1
    return valid_rows, anomaly_rows, anomaly_truth


def write_nextstep(model, tok, valid_rows, device, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for row in valid_rows:
            eid, fam, _frac, partial = row[0], row[1], row[2], row[3]
            preds = predict_next(model, tok, fam, partial, device, k=5)
            preds += [""] * (5 - len(preds))
            w.writerow([eid, *preds[:5]])


def write_completion(model, tok, valid_rows, device, out, guided=False):
    gen = None
    if guided:
        from .guided import complete_guided
        gen = complete_guided
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for row in valid_rows:
            eid, fam, _frac, partial = row[0], row[1], row[2], row[3]
            comp = gen(model, tok, fam, partial, device) if gen else complete(model, tok, fam, partial, device)
            w.writerow([eid, "|".join(comp)])


def write_anomaly(model, tok, anomaly_rows, device, out, thr, mode="model"):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for eid, fam, steps in anomaly_rows:
            if mode == "oracle":
                viol = gs.validate_sequence(steps)
                is_valid = 0 if viol else 1
                score = 1.0 if is_valid else 0.0
                rule = "" if is_valid else viol[0].rule
            else:
                sc, _pos = anomaly_score(model, tok, fam, steps, device)
                is_valid = 1 if sc < thr else 0
                score = 1.0 / (1.0 + math.exp(sc - thr))  # P(valid): high when surprise low
                rule = ""  # rule attribution left blank for the honest LM detector
            w.writerow([eid, is_valid, f"{score:.4f}", rule])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--valid", default=None, help="organizers' eval_input_valid.csv")
    ap.add_argument("--anomaly", default=None, help="organizers' eval_input_anomaly.csv")
    ap.add_argument("--selfmake", action="store_true", help="build stand-in eval inputs from held-out data")
    ap.add_argument("--out-dir", default="submission/out")
    ap.add_argument("--anomaly-mode", default="model", choices=["model", "oracle"])
    ap.add_argument("--anomaly-thr", type=float, default=6.0)
    ap.add_argument("--guided", action="store_true",
                    help="validity-guided decoding for Task 2 (guarantees rule-valid routes)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model = load_model(Path(args.ckpt), device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anomaly_truth = None
    if args.selfmake:
        valid_rows, anomaly_rows, anomaly_truth = selfmake()
    else:
        valid_rows = _read_valid(args.valid) if args.valid else []
        anomaly_rows = _read_anomaly(args.anomaly) if args.anomaly else []

    if valid_rows:
        write_nextstep(model, tok, valid_rows, device, out_dir / "task1_nextstep.csv")
        write_completion(model, tok, valid_rows, device, out_dir / "task2_completion.csv", guided=args.guided)
        print(f"wrote task1_nextstep.csv + task2_completion.csv ({len(valid_rows)} rows, "
              f"guided={args.guided})")
    if anomaly_rows:
        write_anomaly(model, tok, anomaly_rows, device, out_dir / "task3_anomaly.csv",
                      args.anomaly_thr, args.anomaly_mode)
        print(f"wrote task3_anomaly.csv ({len(anomaly_rows)} rows, mode={args.anomaly_mode})")

    # If self-made, score against our own ground truth for a quick sanity read.
    if anomaly_truth:
        import csv as _csv
        preds = {}
        with open(out_dir / "task3_anomaly.csv", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                preds[r["EXAMPLE_ID"]] = int(r["IS_VALID"])
        tp = sum(1 for k, (v, _r) in anomaly_truth.items() if v == 0 and preds.get(k) == 0)
        fp = sum(1 for k, (v, _r) in anomaly_truth.items() if v == 1 and preds.get(k) == 0)
        fn = sum(1 for k, (v, _r) in anomaly_truth.items() if v == 0 and preds.get(k) == 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  self-scored anomaly F1={f1:.3f} (prec {prec:.3f}, rec {rec:.3f})")


if __name__ == "__main__":
    main()
