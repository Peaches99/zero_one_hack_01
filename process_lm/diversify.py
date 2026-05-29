"""Diversity-cranked data generator — the smarter faucet for OOD robustness.

Two ideas, both built ON TOP of the organizers' own validated block generators
(so every route we emit is checked by their validate_sequence):

1. GRAMMAR RANDOMIZATION — assemble routes from the real per-family blocks but
   vary the assembly more aggressively than the default generator does.

2. HYBRID PSEUDO-FAMILIES — Frankenstein valid routes that mix blocks across
   families (e.g. MOSFET prep + IGBT cycles + IC backside). These are novel
   *kinds* of route the model has never seen — a stand-in for the hidden 4th
   family. Each candidate is validated and rejected if it breaks any rule.

The hybrids are labeled with a distinct family tag ("<hybrid>"/given tag) so the
model learns "process logic holds regardless of family identity" rather than
binding logic to a known family token.

Usage:
    from process_lm.diversify import generate_diverse_records
    recs = generate_diverse_records(n_hybrid=4000, seed=1)  # [(family, [steps])]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
sys.path.insert(0, str(_DATA_DIR))

# Reuse the organizers' validated block generators directly.
from generate_sequences import (  # type: ignore  # noqa: E402
    validate_sequence,
    _gen_prefix, _gen_initial_measurements, _gen_pre_process_clean,
    _gen_first_oxidation, _gen_ild_block, _gen_via_block, _gen_metal_block,
    _gen_passivation_block, _gen_backside_block, _gen_final_inspection,
    _gen_test_suite, _gen_suffix,
    _gen_family_prep_mosfet, _gen_family_prep_igbt, _gen_family_prep_ic,
    _gen_cycles_mosfet, _gen_cycles_igbt, _gen_cycles_ic,
    _VIA_LEVEL, _METAL_LEVEL,
)

_PREP = {"mosfet": _gen_family_prep_mosfet, "igbt": _gen_family_prep_igbt, "ic": _gen_family_prep_ic}
_CYCLES = {"mosfet": _gen_cycles_mosfet, "igbt": _gen_cycles_igbt, "ic": _gen_cycles_ic}
FAMILIES = ["mosfet", "igbt", "ic"]


def _assemble(rng, prep_fam, cycle_fam, oxid_fam, back_fam, test_fam, suffix_fam):
    """Assemble one route from blocks that may come from different families."""
    steps = []
    steps += _gen_prefix(rng)
    steps += _gen_initial_measurements(rng, prep_fam)
    steps += _gen_pre_process_clean(rng, prep_fam)
    steps += _PREP[prep_fam](rng)
    steps += _gen_first_oxidation(rng, oxid_fam)
    steps += _CYCLES[cycle_fam](rng)
    steps += _gen_ild_block(rng)
    steps += _gen_via_block(rng, _VIA_LEVEL[cycle_fam], cycle_fam)
    steps += _gen_metal_block(rng, _METAL_LEVEL[cycle_fam], cycle_fam)
    steps += _gen_passivation_block(rng)
    steps += _gen_backside_block(rng, back_fam)
    steps += _gen_final_inspection(rng, prep_fam)
    steps += _gen_test_suite(rng, test_fam)
    steps += _gen_suffix(rng, suffix_fam)
    return steps


def generate_hybrids(n: int, seed: int = 1, tag: str = "hybrid",
                     pool: list[str] | None = None,
                     max_attempts_factor: int = 60) -> list[tuple[str, list[str]]]:
    """Generate `n` unique, VALIDATED hybrid routes mixing family blocks.

    `pool` restricts which families' blocks may be used. During LOFO we pass the
    TRAINED families only, so the held-out family stays genuinely unseen — the
    hybrids must help via transferable logic, not by leaking the target family.
    """
    families = pool or FAMILIES
    rng = random.Random(seed)
    out: list[tuple[str, list[str]]] = []
    seen: set = set()
    attempts = 0
    while len(out) < n and attempts < n * max_attempts_factor:
        attempts += 1
        # Independently pick a family for each block — this is the novelty.
        prep, cycle, oxid, back, test, suffix = (rng.choice(families) for _ in range(6))
        try:
            steps = _assemble(rng, prep, cycle, oxid, back, test, suffix)
        except Exception:
            continue
        key = tuple(steps)
        if key in seen:
            continue
        if validate_sequence(steps):   # reject anything that breaks a rule
            continue
        seen.add(key)
        out.append((tag, steps))
    return out


def generate_diverse_records(
    n_hybrid: int = 4000, seed: int = 1, hybrid_tag: str = "hybrid",
    pool: list[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Just the validated hybrid pseudo-family records (mix with real data upstream)."""
    return generate_hybrids(n_hybrid, seed=seed, tag=hybrid_tag, pool=pool)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Preview the diversity generator.")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    recs = generate_hybrids(args.n, seed=args.seed)
    lens = [len(s) for _, s in recs]
    # How novel are these vs the canonical per-family routes? Report block mixing.
    print(f"generated {len(recs)} unique validated hybrid routes")
    if lens:
        lens.sort()
        print(f"length: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
    # Validate-rate sanity: all emitted routes are valid by construction.
    bad = sum(1 for _, s in recs if validate_sequence(s))
    print(f"invalid among emitted: {bad} (must be 0)")
