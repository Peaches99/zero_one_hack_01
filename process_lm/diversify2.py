"""v2 max-diversity generator — structural variety the stock generator lacks.

The organizers' `generate_sequence` emits a FIXED macro-structure per family
(MOSFET always 4 device cycles, IGBT 6, IC 4). All of its cited combinatorics
come from optional-step toggles and synonyms — never from structural change. A
model can memorize those three templates; the hidden 4th family almost certainly
differs in exactly the axis the stock data holds fixed: *prep and cycle count*
(per the data README). That is a generalization trap.

v2 attacks it directly. Every route is assembled from the organizers' own
VALIDATED block generators, but with:
  * a VARIABLE number of device cycles (3..6), levels renumbered sequentially,
  * generic device cycles drawn from the union of all families' primitives,
  * an optional second metal layer,
  * blocks (prep / oxidation / backside / test) mixed across families.
Every emitted route is checked by the organizers' `validate_sequence`; anything
that breaks a rule is rejected, so the output distribution is guaranteed valid
and far broader than the stock data. Generation is parallelized across cores.

CLI:
    python -m process_lm.diversify2 --n 5000 --report      # stats + validity
    python -m process_lm.diversify2 --n 200000 --out data.csv --workers 14
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

import generate_sequences as gs  # type: ignore  # noqa: E402

FAMILIES = ["mosfet", "igbt", "ic"]
_FAMILY_FILES = {"mosfet": "MOSFET_variants.csv", "igbt": "IGBT_variants.csv", "ic": "IC_variants.csv"}
_VOCAB_CACHE: dict = {}


def _pool_vocab(families: list[str]) -> set:
    """Union of step strings actually present in the given families' real data.

    Used to keep v2 LEAK-FREE during LOFO: any assembled route containing a step
    outside the pool's vocabulary is rejected, so a held-out family's unique steps
    (e.g. IMPLANT N-TYPE, MEASURE CD LEVEL 2 for IC) never sneak into training and
    inflate the OOD proxy. Without this, structural diversity is confounded with
    vocabulary leakage — the exact 'biased set' failure mode to avoid.
    """
    key = tuple(sorted(families))
    if key not in _VOCAB_CACHE:
        vocab: set = set()
        for fam in families:
            seqs = gs.read_csv_sequences(_DATA_DIR / _FAMILY_FILES[fam])
            for s in seqs.values():
                vocab.update(s)
        _VOCAB_CACHE[key] = vocab
    return _VOCAB_CACHE[key]

_CYCLE_INSPECT = [
    "INSPECT PATTERN LEVEL {L}", "PATTERN INSPECTION LEVEL {L}", "POLY PATTERN INSPECTION",
]
_POST_ETCH_CLEAN = ["CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH", "CLEAN AFTER POLY ETCH"]
_CYCLE_CD = ["MEASURE OPENING CD", "MEASURE CD LEVEL {L}", "MEASURE GATE CD", "MEASURE WINDOW CD"]
_IMPLANTS = [
    "IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT N-TYPE", "IMPLANT P BODY",
    "IMPLANT CHANNEL STOP", "IMPLANT LDD", "IMPLANT SOURCE REGION",
]


def _generic_cycle(rng: random.Random, level: int) -> list[str]:
    """One rule-valid device cycle at mask `level`.

    Valid by construction: any deposit is preceded by a clean (RULE_DEP_NO_CLEAN);
    every patterned etch is preceded by a full litho develop (RULE_ETCH_NO_MASK);
    any implant is preceded by that same develop/oxide-etch (RULE_IMPLANT_NO_MASK).
    """
    s: list[str] = []
    if rng.random() < 0.5:  # deposit-then-pattern cycle
        s += [rng.choice(["RCA CLEAN 1", "WET CLEAN RCA1"]), "HF DIP"]  # clean for the deposit
        dep = rng.choice(["THERMAL OXIDATION", "DEPOSIT POLYSILICON", "DEPOSIT FIELD OXIDE"])
        s.append(dep)
        if dep == "DEPOSIT POLYSILICON":
            s.append(rng.choice(["POLYSILICON ANNEAL", "ANNEAL POLYSILICON"]))
            s += gs._meas(rng, "MEASURE POLY THICKNESS")
            etch = rng.choice(["POLYSILICON ETCH", "POLYSILICON ETCH DRY"])
        elif dep == "DEPOSIT FIELD OXIDE":
            s.append(rng.choice(["DENSIFY OXIDE", "DENSIFY DIELECTRIC"]))
            s += gs._meas(rng, "MEASURE FILM THICKNESS")
            etch = "FIELD OXIDE ETCH"
        else:  # THERMAL OXIDATION
            s += gs._meas(rng, "MEASURE OXIDE THICKNESS")
            etch = rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"])
    else:  # pattern an existing surface
        etch = rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW"])

    inspect = rng.choice(_CYCLE_INSPECT).format(L=level)
    s += gs._litho(rng, level, inspect)
    s += [etch, rng.choice(["STRIP PHOTORESIST", "STRIP RESIST"]), rng.choice(_POST_ETCH_CLEAN)]
    s += gs._meas(rng, rng.choice(_CYCLE_CD).format(L=level))

    if rng.random() < 0.6:  # optional implant + anneal
        s.append(rng.choice(_IMPLANTS))
        s += gs._pre_anneal(rng)
        s.append(rng.choice(["RAPID THERMAL ANNEAL", "DRIVE IN DIFFUSION"]))
        if rng.random() < 0.5:
            s.append("RAPID THERMAL ANNEAL")
        s += gs._meas(rng, rng.choice(["MEASURE JUNCTION DEPTH", "MEASURE SHEET RESISTANCE"]))
    return s


def _assemble_v2(rng: random.Random, families: list[str]) -> tuple[list[str], int, bool]:
    """Assemble one structurally-varied route. Returns (steps, n_cycles, extra_metal)."""
    prep = rng.choice(families)
    oxid = rng.choice(families)
    back = rng.choice(families)
    test = rng.choice(families)
    suffix = rng.choice(families)
    n_cycles = rng.randint(3, 6)
    extra_metal = rng.random() < 0.35

    s: list[str] = []
    s += gs._gen_prefix(rng)
    s += gs._gen_initial_measurements(rng, prep)
    s += gs._gen_pre_process_clean(rng, prep)
    s += gs._FAMILY_PREP[prep](rng)
    s += gs._gen_first_oxidation(rng, oxid)
    for level in range(1, n_cycles + 1):
        s += _generic_cycle(rng, level)
    s += gs._gen_ild_block(rng)
    s += gs._gen_via_block(rng, n_cycles + 1, prep)
    s += gs._gen_metal_block(rng, n_cycles + 2, prep)
    if extra_metal:
        s += gs._gen_metal_block(rng, n_cycles + 3, prep)
    s += gs._gen_passivation_block(rng)
    s += gs._gen_backside_block(rng, back)
    s += gs._gen_final_inspection(rng, prep)
    s += gs._gen_test_suite(rng, test)
    s += gs._gen_suffix(rng, suffix)
    return s, n_cycles, extra_metal


def generate_v2(n: int, seed: int = 1, pool: list[str] | None = None,
                tag: str = "random", max_attempts_factor: int = 8) -> list[tuple[str, list[str]]]:
    """Generate `n` unique VALIDATED structurally-diverse routes.

    `pool` restricts which families' blocks may be used (pass the trained
    families during LOFO so the held-out family stays unseen). `tag` controls the
    family token: 'random' varies it per route (decouples logic from identity),
    else a fixed string.
    """
    families = pool or FAMILIES
    # LOFO leak guard: when a pool is given, reject any route that uses a step
    # outside the pool's real vocabulary (would leak the held-out family).
    allowed = _pool_vocab(families) if pool else None
    rng = random.Random(seed)
    out: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(out) < n and attempts < n * max_attempts_factor:
        attempts += 1
        try:
            steps, _, _ = _assemble_v2(rng, families)
        except Exception:
            continue
        if allowed is not None and any(s not in allowed for s in steps):
            continue  # would leak a non-pool (held-out-family) step
        key = tuple(steps)
        if key in seen:
            continue
        if gs.validate_sequence(steps):  # reject rule-breakers (should be rare)
            continue
        seen.add(key)
        fam_tag = rng.choice(families) if tag == "random" else tag
        out.append((fam_tag, steps))
    return out


def _worker(args) -> list[tuple[str, list[str]]]:
    n, seed, pool, tag = args
    return generate_v2(n, seed=seed, pool=pool, tag=tag)


def generate_v2_parallel(n: int, seed: int = 1, pool: list[str] | None = None,
                         tag: str = "random", workers: int = 8) -> list[tuple[str, list[str]]]:
    """Parallel generation across processes, then global dedup."""
    import multiprocessing as mp
    per = (n + workers - 1) // workers
    jobs = [(per, seed + 100003 * w, pool, tag) for w in range(workers)]
    with mp.Pool(workers) as pool_:
        chunks = pool_.map(_worker, jobs)
    out: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for ch in chunks:
        for fam, steps in ch:
            key = tuple(steps)
            if key in seen:
                continue
            seen.add(key)
            out.append((fam, steps))
            if len(out) >= n:
                return out
    return out


def _report(recs: list[tuple[str, list[str]]]) -> None:
    from collections import Counter
    lens = sorted(len(s) for _, s in recs)
    n = len(recs)
    bad = sum(1 for _, s in recs if gs.validate_sequence(s))
    cyc = Counter()
    for _, s in recs:
        cyc[sum(1 for x in s if x.startswith("ALIGN MASK LEVEL "))] += 1
    print(f"generated {n} unique validated v2 routes")
    print(f"invalid among emitted: {bad} (must be 0)")
    print(f"length: min={lens[0]} median={lens[n // 2]} max={lens[-1]} mean={sum(lens) / n:.1f}")
    print(f"litho-level (~cycle+via+metal) count distribution: "
          f"{dict(sorted(cyc.items()))}")
    fam = Counter(f for f, _ in recs)
    print(f"family-tag distribution: {dict(fam)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None, help="write a SEQUENCE_ID,STEP CSV")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tag", default="random")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    if args.workers > 1:
        recs = generate_v2_parallel(args.n, seed=args.seed, tag=args.tag, workers=args.workers)
    else:
        recs = generate_v2(args.n, seed=args.seed, tag=args.tag)

    if args.report:
        _report(recs)
    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["SEQUENCE_ID", "STEP"])
            for i, (_fam, steps) in enumerate(recs, 1):
                sid = f"v2_{i:06d}"
                for step in steps:
                    w.writerow([sid, step])
        print(f"wrote {len(recs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
