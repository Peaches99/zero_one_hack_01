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


def fig_block_ceiling():
    """Task-2 completion: model block-acc sits ON the Bayes-optimal ceiling.
    Measured by process_lm.block_gap (token-conditioned oracle ceiling)."""
    import numpy as np
    cuts = ["60% cut", "80% cut", "overall"]
    model = [0.4651, 0.9284, 0.6968]    # greedy; submitted MBR overall = 0.711
    ceil = [0.4904, 0.9305, 0.7105]     # token-conditioned Bayes ceiling
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, model, w, label="our model", color="tab:blue")
    ax.bar(x + w / 2, ceil, w, label="Bayes ceiling (oracle)", color="tab:gray")
    for i, (m, c) in enumerate(zip(model, ceil)):
        ax.text(i, max(m, c) + 0.02, f"gap {c - m:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cuts); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Block-level Accuracy")
    ax.set_title("Task 2: at the ceiling — 80% cut is maxed, 60% is information-limited")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "block_ceiling.png", dpi=130); plt.close(fig)


def fig_id_vs_ood_transfer():
    """The transferable-understanding result: on an unseen family, block STRUCTURE
    survives while exact-token prediction collapses (vocabulary ceiling)."""
    import numpy as np
    groups = ["Block-level Acc\n(structure)", "Token Acc\n(exact step)"]
    id_vals = [0.697, 0.418]    # in-distribution (block_gap)
    ood_vals = [0.501, 0.157]   # held-out family (ood_eval, LOFO)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, id_vals, w, label="in-distribution", color="tab:green")
    ax.bar(x + w / 2, ood_vals, w, label="OOD (unseen family)", color="tab:orange")
    for i, (a, b) in enumerate(zip(id_vals, ood_vals)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 0.85)
    ax.set_ylabel("accuracy")
    ax.set_title("Process STRUCTURE transfers to unseen families; exact tokens don't")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "id_vs_ood_transfer.png", dpi=130); plt.close(fig)


def fig_diversity_scaling():
    """Headline OOD study: at FIXED data volume, does training on more families lift
    OOD block-acc? Reads process_lm/runs/diversity_results.csv (diversity_ood.py)."""
    import numpy as np
    p = RUNS / "diversity_results.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p, newline="")))
    if not rows:
        return
    fams = [f for f in ("mosfet", "igbt", "ic")
            if any(r["held_out"] == f for r in rows)]
    series = {"1 family": [], "2 families": [], "2 fam + family-dropout": []}
    labels = []
    for f in fams + ["MEAN"]:
        sub = rows if f == "MEAN" else [r for r in rows if r["held_out"] == f]
        one = [float(r["ood_blk"]) for r in sub if r["condition"].startswith("1fam_")]
        two = [float(r["ood_blk"]) for r in sub if r["condition"] == "2fam"]
        twofd = [float(r["ood_blk"]) for r in sub if r["condition"] == "2fam_fd15"]
        if not (one and two):
            continue
        labels.append(f.upper())
        series["1 family"].append(sum(one) / len(one))
        series["2 families"].append(sum(two) / len(two))
        series["2 fam + family-dropout"].append(sum(twofd) / len(twofd) if twofd else np.nan)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels)); w = 0.26
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1) * w, vals, w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, max(0.7, max(
        v for vs in series.values() for v in vs if v == v) + 0.08))
    ax.set_ylabel("OOD Block-level Accuracy (held-out family)")
    ax.set_title("Diversity scaling: more training families -> better OOD (fixed data volume)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "diversity_scaling.png", dpi=130); plt.close(fig)


def _grok_log(name: str):
    p = RUNS / "grok" / name / "train_log.csv"
    if not p.exists():
        return None
    ep, tr, vl = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); tr.append(float(r["train_loss"])); vl.append(float(r["val_loss"]))
    return (ep, tr, vl) if ep else None


def fig_grokking():
    """Delayed generalization on a tiny training set: train memorizes early (loss <<
    floor) while held-out loss lags, then (if grokking) snaps down toward the floor."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, lbl, c in [("d100_wd1.0", "100 seqs", "tab:red"),
                         ("d300_wd1.0", "300 seqs", "tab:blue"),
                         ("d800_wd1.0", "800 seqs (control)", "tab:green")]:
        d = _grok_log(name)
        if not d:
            continue
        ax[0].plot(d[0], d[2], c=c, label=f"{lbl} — held-out")
    dt = _grok_log("d100_wd1.0")
    if dt:
        ax[0].plot(dt[0], dt[1], ls=":", c="gray", lw=1, label="100 seqs — train (memorized)")
    ax[0].axhline(ID_FLOOR, ls="--", c="k", lw=1, label=f"entropy floor {ID_FLOOR}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("epoch (log scale)"); ax[0].set_ylabel("loss (nats/token)")
    ax[0].set_title("Grokking? held-out loss vs training time (wd=1.0)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for name, lbl in [("d300_wd1.0", "wd=1.0"), ("d300_wd0.1", "wd=0.1")]:
        d = _grok_log(name)
        if not d:
            continue
        ax[1].plot(d[0], d[2], label=f"{lbl} — held-out")
    ax[1].axhline(ID_FLOOR, ls="--", c="k", lw=1, label=f"floor {ID_FLOOR}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("epoch (log scale)")
    ax[1].set_title("Weight-decay effect on grokking (300 seqs)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "grokking.png", dpi=130); plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_loss_curves, fig_scaling, fig_hybrid_dose, fig_levers,
               fig_guided_decoding, fig_floor_split, fig_position_accuracy,
               fig_block_ceiling, fig_id_vs_ood_transfer, fig_diversity_scaling,
               fig_grokking):
        try:
            fn()
            made.append(fn.__name__)
        except Exception as e:  # keep going; partial figures are fine
            print(f"[skip] {fn.__name__}: {e}")
    print(f"figures written to {FIG}: {made}")


if __name__ == "__main__":
    main()
