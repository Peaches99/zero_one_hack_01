"""Summarize the overnight grid into report-ready tables.

    python -m process_lm.overnight_report
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("process_lm/runs/overnight/results.jsonl")


def load() -> list[dict]:
    if not RESULTS.exists():
        raise SystemExit(f"no results at {RESULTS} — run process_lm.overnight first")
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def table(rows, title, key, cols):
    print(f"\n=== {title} ===")
    print("  " + "  ".join(f"{c:>16}" for c in [key] + cols))
    for r in rows:
        vals = [r.get("name", "")] + [r.get(c) for c in cols]
        cells = []
        for v in vals:
            if isinstance(v, float):
                cells.append(f"{v:>16.3f}")
            elif isinstance(v, int):
                cells.append(f"{v:>16,}")
            else:
                cells.append(f"{str(v):>16}")
        print("  " + "  ".join(cells))


def main() -> None:
    rows = load()
    by = lambda pfx: [r for r in rows if r["name"].startswith(pfx)]

    cols = ["params", "id_top1", "ood_top1", "ood_valid_completion", "ood_block_ned", "ood_ppl"]

    data_rows = sorted(by("data"), key=lambda r: int(r["name"].replace("data", "").split("_")[0]))
    if data_rows:
        table(data_rows, "1. DATA SCALING (more real data -> ?)", "name", cols)
        print("  read: does OOD_valid rise / OOD_ppl fall as data grows?")

    order = {"tiny": 0, "small": 1, "medium": 2, "large": 3}
    model_rows = sorted(by("model_"), key=lambda r: order.get(r["name"].split("_")[1], 9))
    if model_rows:
        table(model_rows, "2. MODEL SCALING (bigger model, fixed data)", "name", cols)
        print("  read: does ID saturate while OOD gap WIDENS (memorization)?")

    hyb_rows = sorted(by("hybrid"), key=lambda r: int(r["name"].replace("hybrid", "").split("_")[0]))
    if hyb_rows:
        table(hyb_rows, "3. HYBRID DOSE (validated pseudo-families -> OOD)", "name", cols)
        print("  read: the headline — does OOD_valid climb toward 1.0 with more hybrids?")
        if len(hyb_rows) >= 2:
            base, best = hyb_rows[0], max(hyb_rows, key=lambda r: r.get("ood_valid_completion") or 0)
            dv = (best.get("ood_valid_completion") or 0) - (base.get("ood_valid_completion") or 0)
            dp = (best.get("ood_ppl") or 0) - (base.get("ood_ppl") or 0)
            print(f"\n  VERDICT: best hybrid dose '{best['name']}' vs no-hybrid: "
                  f"OOD_valid {dv:+.3f}, OOD_ppl {dp:+.2f}  "
                  f"-> {'HYBRIDS HELP OOD' if dv > 0 else 'no clear OOD gain'}")


if __name__ == "__main__":
    main()
