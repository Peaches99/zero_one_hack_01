"""Figures for the report: loss curves (vs the proven floor), scaling, hybrid dose.

    python -m process_lm.plots
Reads process_lm/runs/** (train_log.csv per run, results.jsonl grids) and writes
PNGs to process_lm/runs/figures/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("process_lm/runs")
FIG = RUNS / "figures"
ID_FLOOR = 0.328  # exact oracle floor (ID mixture); see process_lm.oracle


def _log(run: str):
    p = RUNS / "overnight" / run / "train_log.csv"
    if not p.exists():
        return None
    ep, tr, vl = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); tr.append(float(r["train_loss"])); vl.append(float(r["val_loss"]))
    return ep, tr, vl


def _results(name: str):
    p = RUNS / name / "results.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def fig_loss_curves():
    """Train vs OOD-val for the data-scaling runs, against the ID floor."""
    runs = [("data200_ic", "200"), ("data1000_ic", "1k"), ("data5000_ic", "5k"), ("data20000_ic", "20k")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, lbl in runs:
        d = _log(run)
        if not d:
            continue
        ep, tr, vl = d
        ax[0].plot(ep, tr, label=f"{lbl} seqs")
        ax[1].plot(ep, vl, label=f"{lbl} seqs")
    ax[0].axhline(ID_FLOOR, ls="--", c="k", lw=1, label=f"exact ID floor {ID_FLOOR}")
    ax[0].set_title("TRAIN loss hits the information floor"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("nats/token"); ax[0].legend(fontsize=8)
    ax[1].set_title("OOD (held-out IC) val loss — overfits, does not transfer"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "loss_curves.png", dpi=130); plt.close(fig)


def _scaling_panel(ax, rows, xkey, xlabel, logx=False):
    rows = [r for r in rows if r.get(xkey) is not None]
    rows.sort(key=lambda r: r[xkey])
    xs = [r[xkey] for r in rows]
    ax.plot(xs, [r["ood_valid_completion"] for r in rows], "o-", label="OOD valid-completion")
    ax.plot(xs, [r["ood_top1"] for r in rows], "s-", label="OOD next-step top1")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)


def fig_scaling():
    rows = _results("overnight")
    data = sorted((r for r in rows if r["name"].startswith("data")),
                  key=lambda r: int(r["name"].replace("data", "").split("_")[0]))
    model = [r for r in rows if r["name"].startswith("model_")]
    order = {"tiny": 0, "small": 1, "medium": 2, "large": 3}
    model.sort(key=lambda r: order.get(r["name"].split("_")[1], 9))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    if data:
        for r in data:
            r["_n"] = int(r["name"].replace("data", "").split("_")[0])
        _scaling_panel(ax[0], data, "_n", "real training sequences", logx=True)
        ax[0].set_title("Data scaling: OOD flat (memorization, not learning)")
    if model:
        _scaling_panel(ax[1], model, "params", "model parameters", logx=True)
        ax[1].set_title("Model scaling: bigger != better OOD")
    fig.tight_layout(); fig.savefig(FIG / "scaling.png", dpi=130); plt.close(fig)


def fig_hybrid_dose():
    rows = [r for r in _results("overnight") if r["name"].startswith("hybrid")]
    if not rows:
        return
    for r in rows:
        r["_h"] = int(r["name"].replace("hybrid", "").split("_")[0])
    rows.sort(key=lambda r: r["_h"])
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    hs = [r["_h"] for r in rows]
    ax1.plot(hs, [r["ood_valid_completion"] for r in rows], "o-", c="tab:green", label="OOD valid-completion")
    ax1.plot(hs, [r["ood_top1"] for r in rows], "s-", c="tab:blue", label="OOD next-step top1")
    ax1.set_xlabel("validated hybrid pseudo-families added"); ax1.set_ylabel("rate"); ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(hs, [r["ood_ppl"] for r in rows], "^--", c="tab:red", label="OOD perplexity")
    ax2.set_ylabel("perplexity")
    ax1.legend(loc="lower left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax1.set_title("Hybrid dose-response (OOD generalization)")
    fig.tight_layout(); fig.savefig(FIG / "hybrid_dose.png", dpi=130); plt.close(fig)


def fig_levers():
    rows = _results("ood")
    if not rows:
        return
    # keep the last occurrence per config name
    by = {}
    for r in rows:
        by[r["name"]] = r
    rows = list(by.values())
    rows.sort(key=lambda r: r.get("ood_valid_completion") or 0)
    names = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(rows))
    ax.barh([i - 0.2 for i in y], [r["ood_valid_completion"] for r in rows], 0.4, label="valid-completion")
    ax.barh([i + 0.2 for i in y], [r["ood_top1"] for r in rows], 0.4, label="next-step top1")
    ax.set_yticks(list(y)); ax.set_yticklabels(names); ax.set_xlim(0, 1.02)
    ax.set_title("OOD generalization levers (hold-out IC)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "levers.png", dpi=130); plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_loss_curves, fig_scaling, fig_hybrid_dose, fig_levers):
        try:
            fn()
            made.append(fn.__name__)
        except Exception as e:  # keep going; partial figures are fine
            print(f"[skip] {fn.__name__}: {e}")
    print(f"figures written to {FIG}: {made}")


if __name__ == "__main__":
    main()
