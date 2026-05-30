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


def fig_guided_decoding():
    """Measured OOD valid-completion: greedy vs validity-guided vs guided+repair."""
    import numpy as np
    fams = ["MOSFET", "IGBT", "IC"]
    greedy = [0.733, 0.617, 0.983]   # measured, hold-out family, runs/ood/*_real
    guided = [0.817, 1.000, 1.000]
    repair = [1.000, 1.000, 1.000]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3); w = 0.26
    ax.bar(x - w, greedy, w, label="greedy")
    ax.bar(x, guided, w, label="validity-guided")
    ax.bar(x + w, repair, w, label="guided + grammar repair")
    ax.set_xticks(x); ax.set_xticklabels(fams); ax.set_ylim(0, 1.08)
    ax.set_ylabel("OOD valid-completion rate")
    ax.set_title("Guided decoding -> 100% valid routes for unseen families")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "guided_decoding.png", dpi=130); plt.close(fig)


def fig_floor_split():
    """The 0.01 target resolved: model loss on deterministic vs stochastic tokens."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cats = ["deterministic\n(rule-forced, 54%)", "stochastic\n(coin-flips, 46%)"]
    vals = [0.0002, 0.7248]  # measured per-token model NLL by position type
    bars = ax.bar(cats, vals, color=["tab:green", "tab:red"])
    ax.axhline(0.01, ls="--", c="k", lw=1, label="0.01 target")
    ax.axhline(ID_FLOOR, ls=":", c="gray", lw=1, label=f"ID floor {ID_FLOOR}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("model loss (nats/token)")
    ax.set_title('"0.01" is reached on the logic; the rest is irreducible entropy')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "floor_split.png", dpi=130); plt.close(fig)


def fig_position_accuracy():
    """ID vs held-out next-step top-1 by position decile (from lofo_analysis.json)."""
    p = RUNS / "lofo_analysis.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fams = [f for f in ("mosfet", "igbt", "ic") if f in d.get("families", {})]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.3 * len(fams), 4), sharey=True)
    if len(fams) == 1:
        axes = [axes]
    xs = [(i + 0.5) / 10 for i in range(10)]
    for ax, f in zip(axes, fams):
        fam = d["families"][f]
        ax.plot(xs, fam["id_pos_acc"], "o-", label="ID (trained)")
        ax.plot(xs, fam["ood_pos_acc"], "s--", label="OOD (held-out)")
        ax.set_title(f"hold-out {f.upper()}"); ax.set_xlabel("position in route")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0].set_ylabel("next-step top-1"); axes[0].legend(fontsize=8)
    fig.suptitle("Where generalization holds: top-1 by route position (ID vs held-out family)")
    fig.tight_layout(); fig.savefig(FIG / "position_accuracy.png", dpi=130); plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_loss_curves, fig_scaling, fig_hybrid_dose, fig_levers,
               fig_guided_decoding, fig_floor_split, fig_position_accuracy):
        try:
            fn()
            made.append(fn.__name__)
        except Exception as e:  # keep going; partial figures are fine
            print(f"[skip] {fn.__name__}: {e}")
    print(f"figures written to {FIG}: {made}")


if __name__ == "__main__":
    main()
