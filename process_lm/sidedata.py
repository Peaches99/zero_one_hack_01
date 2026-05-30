"""Predict a process step (and a full route) WITH its required side data.

The organizers ship, per step, a DESCRIPTION and REALISTIC FAB-LEVEL PARAMETERS
(e.g. EPITAXIAL DEPOSITION -> "RPCVD; SiHCl3 20 sccm, H2 10 slm; 1050 C; 40 Torr").
This is the track's "process parameters" stretch goal.

The parameters are **family-specific** (FILL VIA METAL differs MOSFET vs IGBT vs IC)
and mostly deterministic given (step, family). So the model does the hard part —
predicting the *step* from the process logic — and we attach the correct side data
with a robust (step, family) lookup that handles the grammar's synonyms, level
suffixes, cross-family fallback, and finally a category-level default so EVERY
predicted step gets sensible side data (important for the unseen 4th family).

    python -m process_lm.sidedata --ckpt process_lm/runs/final/best.pt --family mosfet
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from .guided import complete_guided, repair_route
from .metrics import step_category
from .predict import get_device, load_model, predict_next
from .tokenizer import Tokenizer

_DATA = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
if str(_DATA) not in sys.path:
    sys.path.insert(0, str(_DATA))

FAMS = ["mosfet", "igbt", "ic"]
_PARAM_FILE = {f: f"{f.upper()}_longdescription_parameters.csv" for f in FAMS}
_DESC_FILE = {f: f"{f.upper()}_Longdescr.csv" for f in FAMS}

# Generic per-category fallback parameters for steps absent from the side-data
# tables (synonyms not listed, or genuinely novel steps in an unseen family).
_CATEGORY_DEFAULT = {
    "CLEAN": "Wet/clean per fab standard; DI rinse + dry",
    "LITHO": "Spin/expose/develop; track + scanner, standard recipe",
    "ETCH": "Dry plasma etch; endpoint-controlled, standard chemistry",
    "IMPLANT": "Ion implant; species/energy/dose per device target",
    "THERMAL": "Furnace/RTA anneal; temperature + time per recipe",
    "DEPOSIT": "CVD/PVD deposition; thickness + rate per target",
    "CMP": "Chemical-mechanical planarization; standard slurry + downforce",
    "METROLOGY": "In-line measurement/inspection; spec-limit gating",
    "TEST": "Electrical/parametric test; spec-limit pass/fail",
    "VIA_FILL": "Barrier + seed + fill; CMP planarization",
    "STRIP": "Resist strip; ash + wet clean",
    "PREP": "Substrate preparation/conditioning per fab standard",
    "LOGISTICS": "Lot handling/logistics; MES tracked",
    "OTHER": "Process step per fab standard recipe",
}


def _norm_header(h: str) -> str:
    return h.lstrip("﻿").strip().strip('"').strip()


def _read(path: Path, want_param: bool):
    """Return {step: [values...]} where value is params (or description)."""
    out: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        m = {_norm_header(h): h for h in (r.fieldnames or [])}
        skey = m["STEP"]
        if want_param:
            pk = next((v for k, v in m.items() if "PARAM" in k.upper()), None)
        else:
            pk = m.get("DESCRIPTION")
        for row in r:
            st = row[skey].strip().strip('"')
            val = (row[pk].strip().strip('"') if pk else "")
            out.setdefault(st, []).append(val)
    return out


def _extract_synonym_groups() -> list[set]:
    """Parse the organizers' generator for rng.choice([...]) string lists — the
    authoritative synonym groups (STRIP PHOTORESIST / STRIP RESIST, etc.)."""
    src = (_DATA / "generate_sequences.py").read_text(encoding="utf-8")
    groups = []
    for m in re.finditer(r"rng\.choice\(\s*\[(.*?)\]\s*\)", src, re.DOTALL):
        items = re.findall(r'"([^"]+)"', m.group(1))
        if len(items) >= 2:
            groups.append(set(items))
    return groups


def _level_key(step: str) -> str:
    return re.sub(r"\d+", "N", step)


class SideData:
    def __init__(self):
        self.params: dict[tuple, str] = {}   # (family, step) -> params (most common)
        self.desc: dict[tuple, str] = {}
        self.by_step_any: dict[str, dict[str, str]] = {}  # step -> {family: params}
        self.level_index: dict[tuple, str] = {}  # (family, level_key) -> step
        for f in FAMS:
            for step, vals in _read(_DATA / _PARAM_FILE[f], True).items():
                p = Counter(vals).most_common(1)[0][0]
                self.params[(f, step)] = p
                self.by_step_any.setdefault(step, {})[f] = p
                self.level_index[(f, _level_key(step))] = step
            for step, vals in _read(_DATA / _DESC_FILE[f], False).items():
                self.desc[(f, step)] = Counter(vals).most_common(1)[0][0]
        # synonym -> set of equivalents
        self.syn: dict[str, set] = {}
        for g in _extract_synonym_groups():
            for s in g:
                self.syn.setdefault(s, set()).update(g - {s})

    def lookup(self, family: str, step: str) -> tuple[str, str, str]:
        """Return (description, parameters, source) for a (family, step)."""
        family = family.lower()
        # 1. exact (family, step)
        if (family, step) in self.params:
            return self.desc.get((family, step), ""), self.params[(family, step)], "exact"
        # 2. a synonym, same family
        for syn in self.syn.get(step, ()):
            if (family, syn) in self.params:
                return self.desc.get((family, syn), ""), self.params[(family, syn)], f"synonym({syn})"
        # 3. level-normalized, same family (ALIGN MASK LEVEL 6 -> ...LEVEL N)
        alt = self.level_index.get((family, _level_key(step)))
        if alt:
            return self.desc.get((family, alt), ""), self.params[(family, alt)], f"level({alt})"
        # 4. same step (or synonym) in ANY family — prefer a consistent default
        for cand in [step, *self.syn.get(step, ())]:
            fam_map = self.by_step_any.get(cand)
            if fam_map:
                ff = next(iter(fam_map))
                return self.desc.get((ff, cand), ""), fam_map[ff], f"cross-family({ff})"
        # 5. category default (covers novel steps in the unseen 4th family)
        cat = step_category(step)
        return f"{step.title()} (process step)", _CATEGORY_DEFAULT.get(cat, _CATEGORY_DEFAULT["OTHER"]), f"category({cat})"


def predict_next_with_sidedata(model, tok, family, partial, device, side, k=5):
    steps = predict_next(model, tok, family, partial, device, k=k)
    return [(s, *side.lookup(family, s)) for s in steps]


def complete_with_sidedata(model, tok, family, partial, device, side):
    comp = repair_route(list(partial) + complete_guided(model, tok, family, partial, device))[len(partial):]
    return [(s, *side.lookup(family, s)) for s in comp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--family", default="mosfet", choices=FAMS)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    model = load_model(Path(args.ckpt), device)
    tok = Tokenizer.load(Path(args.ckpt).parent / "tokenizer.json")
    side = SideData()

    from .data import build_records, load_all_families, split_records
    _, val = split_records(build_records(load_all_families(_DATA)), 100, 0)
    fam, steps = next(r for r in val if r[0] == args.family)
    cut = max(1, int(len(steps) * args.frac))
    partial = steps[:cut]

    print("=" * 80)
    print(f"FAMILY {fam.upper()} — context tail: ...{' | '.join(partial[-3:])}")
    print("\n### 1. NEXT PROCESS STEP + REQUIRED SIDE DATA (top-3) ###")
    for i, (s, d, p, src) in enumerate(predict_next_with_sidedata(model, tok, fam, partial, device, side, k=3), 1):
        mark = "  <-- TRUE" if s == steps[cut] else ""
        print(f"  [{i}] {s}{mark}")
        print(f"      description: {d}")
        print(f"      parameters : {p}    [{src}]")

    print("\n### 2. FULL ROUTE COMPLETION + REQUIRED SIDE DATA (each remaining step) ###")
    rows = complete_with_sidedata(model, tok, fam, partial, device, side)
    print(f"  completing from step {cut}/{len(steps)} -> +{len(rows)} steps")
    for s, d, p, src in rows[:8]:
        print(f"  {s:32} | {p}")
    if len(rows) > 8:
        print(f"  ... (+{len(rows)-8} more, ending at {rows[-1][0]})")
    cov = Counter(src.split("(")[0] for _s, _d, _p, src in rows)
    print(f"\n  side-data source coverage: {dict(cov)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
