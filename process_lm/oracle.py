"""Exact entropy oracle for the Infineon process grammar.

The most important anti-self-deception tool in this repo. It computes the *exact*
Bayes-optimal next-token cross-entropy of the organizers' generator — the true
information floor — by instrumenting the REAL generator (not a re-implementation)
to record the probability of every random decision it makes.

Why this floor can be trusted (it is engineered to be falsifiable, not
self-confirming):

  1. FAITHFULNESS. ``selftest`` asserts the instrumented generator emits
     byte-identical sequences to the stock ``generate_sequence`` for the same
     seed across hundreds of seeds per family. If recording ever perturbed the
     RNG draw order, the assert fails loudly and the floor is rejected.

  2. COMPLETENESS. Every randomness site in generate_sequences.py routes through
     either a patched helper (``_opt`` / ``_meas`` / ``_pre_anneal`` plus the
     three inline-Bernoulli blocks) or ``RecordingRandom.choice`` — verified by
     grep. The one degenerate site (``_gen_test_suite`` eagerly evaluates an IC
     ``rng.choice`` whose result is discarded for MOSFET/IGBT) is handled by
     enumerating its four local outcomes exactly, so the floor is neither over-
     nor under-counted there.

  3. FALSIFIABILITY. ``score_model_vs_floor`` checks a trained model's per-token
     NLL on FRESH held-out sequences against this floor. A correct floor is a
     hard lower bound that the model approaches from ABOVE; if a model ever
     scores below it, the floor (the instrumentation) is wrong — not the model.
     And reaching the floor on THIS distribution caps nothing else: OOD,
     alternative data mixtures, tokenizations, and objectives all live above it.

Token accounting matches the model exactly. The model scores targets
``[FAM, s1..sK, EOS]`` given ``[BOS, FAM, s1..sK]`` (K+2 positions). ``s1``
(always RECEIVE WAFER LOT) and ``EOS`` (end is forced once the route completes)
are deterministic and cost ~0 nats; ``FAM`` costs ``-log P(family)`` under the
training mixture; the steps cost the sum of the generator's decision log-probs.

Run:
    python -m process_lm.oracle                 # selftest + floor report
    python -m process_lm.oracle --n 8000        # tighter estimate
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import math
import random
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "tracks/industrial-infineon/training_data"
FAMILIES = ["mosfet", "igbt", "ic"]
LOG_HALF = math.log(0.5)


def _load_generator():
    """Import the organizers' generator in isolation (its own module object)."""
    spec = importlib.util.spec_from_file_location(
        "gen_seq_oracle", _DATA_DIR / "generate_sequences.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gs = _load_generator()

# Originals captured before any patching, so the faithfulness test compares
# instrumented output against the genuine stock generator.
_ORIG = {
    name: getattr(gs, name)
    for name in (
        "_opt", "_meas", "_pre_anneal",
        "_gen_initial_measurements", "_gen_pre_process_clean", "_gen_test_suite",
    )
}


class RecordingRandom(random.Random):
    """``random.Random`` that logs the log-prob of every ``.choice`` outcome.

    Bernoulli decisions are logged by the patched helpers (which know their
    thresholds). ``.choice`` is logged here because its probability (1/len) is
    self-contained. Selection is delegated to ``super()`` so the draw stream —
    and therefore every generated sequence — is byte-identical to a stock RNG.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rec: list[tuple[float, object]] = []  # (logp, emitted_step_or_None)

    def choice(self, seq):
        out = super().choice(seq)
        self.rec.append((math.log(1.0 / len(seq)), out))
        return out

    def choice_norec(self, seq):
        """Advance the RNG exactly like ``.choice`` but record nothing.

        Used for the one site where the stock generator eagerly evaluates an
        ``rng.choice`` whose result is then discarded (so it carries no entropy).
        """
        return super().choice(seq)


def _rec(rng, logp, step):
    r = getattr(rng, "rec", None)
    if r is not None:
        r.append((logp, step))


# --------------------------------------------------------------------------- #
# Patched helpers — identical behavior + decision logging.                    #
# Each consumes the RNG in the exact same order/count as the original so the  #
# generated sequence is unchanged (guarded by selftest).                      #
# --------------------------------------------------------------------------- #

def _p_opt(rng, step, prob=0.75):
    inc = rng.random() < prob
    _rec(rng, math.log(prob) if inc else math.log(1.0 - prob), step if inc else None)
    return [step] if inc else []


def _p_meas(rng, step, prob=0.75):
    inc = rng.random() < prob
    _rec(rng, math.log(prob) if inc else math.log(1.0 - prob), step if inc else None)
    return [step] if inc else []


def _p_pre_anneal(rng):
    inc = rng.random() > 0.4  # present with prob 0.6
    _rec(rng, math.log(0.6) if inc else math.log(0.4), "PRE ANNEAL CHECK" if inc else None)
    return ["PRE ANNEAL CHECK"] if inc else []


def _p_gen_initial_measurements(rng, family):
    steps: list[str] = []
    thickness = {
        "mosfet": ["MEASURE THICKNESS"],
        "igbt": ["MEASURE INITIAL THICKNESS"],
        "ic": ["MEASURE INITIAL GEOMETRY", "MEASURE INITIAL THICKNESS"],
    }[family]
    inc = rng.random() > 0.15  # present with prob 0.85
    _rec(rng, math.log(0.85) if inc else math.log(0.15), None)
    if inc:
        steps.append(rng.choice(thickness))  # recorded by RecordingRandom.choice
    surface = {
        "mosfet": ["MEASURE SURFACE PARTICLES"],
        "igbt": ["MEASURE SURFACE PARTICLES"],
        "ic": ["MEASURE SURFACE DEFECTS", "MEASURE SURFACE PARTICLES"],
    }[family]
    inc2 = rng.random() > 0.15
    _rec(rng, math.log(0.85) if inc2 else math.log(0.15), None)
    if inc2:
        steps.append(rng.choice(surface))
    return steps


def _p_gen_pre_process_clean(rng, family):
    steps: list[str] = []
    steps.append("WAFER CLEAN PRE PROCESS" if family == "ic" else "PRE CLEAN WAFER")
    # backside: IGBT always (short-circuits, no RNG draw); others 50/50
    if family == "igbt":
        steps.append("BACKSIDE CLEAN")
    else:
        inc = rng.random() > 0.5
        _rec(rng, LOG_HALF, "BACKSIDE CLEAN" if inc else None)
        if inc:
            steps.append("BACKSIDE CLEAN")
    # frontside: IGBT always; others present with prob 0.4
    if family == "igbt":
        steps.append("FRONTSIDE CLEAN")
    else:
        inc = rng.random() > 0.6
        _rec(rng, math.log(0.4) if inc else math.log(0.6), "FRONTSIDE CLEAN" if inc else None)
        if inc:
            steps.append("FRONTSIDE CLEAN")
    steps.append(rng.choice(["RCA CLEAN 1", "WET CLEAN RCA1"]))
    steps.append(rng.choice(["RCA CLEAN 2", "WET CLEAN RCA2"]))
    steps.append("HF DIP")
    steps += _p_opt(rng, "DRY WAFER", 0.6)
    return steps


def _p_gen_test_suite(rng, family):
    # param token (emitted as s[0]); recorded by RecordingRandom.choice -> log(1/2)
    param = rng.choice(["PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST"])
    # The stock generator builds a dict literal whose IC value EAGERLY calls
    # rng.choice for EVERY family; the draw is discarded for MOSFET/IGBT. Replay
    # the draw (to keep the stream identical) WITHOUT recording it.
    ic_val = rng.choice_norec(["THRESHOLD VOLTAGE TEST", "PARAMETRIC TEST"])
    family_test = {
        "mosfet": "THRESHOLD VOLTAGE TEST",
        "igbt": "BREAKDOWN VOLTAGE TEST",
        "ic": ic_val,
    }[family]
    s = [param, "LEAKAGE TEST"]
    if family_test != param:
        s.append(family_test)
    s.append("SWITCHING TEST")
    # For IC, the (param, ic_val) pair reduces to a single 50/50 branch on the
    # token right after LEAKAGE TEST (THRESHOLD vs SWITCHING vs PARAMETRIC), all
    # conditioned on the already-emitted param -> exactly log(1/2) of entropy.
    if family == "ic":
        _rec(rng, LOG_HALF, "TEST_BRANCH")
    if family == "igbt":
        order = rng.random() > 0.5
        _rec(rng, LOG_HALF, "TEST_ORDER")
        if order:
            s += ["YIELD ANALYSIS", "WAFER SORT TEST"]
        else:
            s += ["WAFER SORT TEST", "YIELD ANALYSIS"]
    else:
        s += ["WAFER SORT TEST", "YIELD ANALYSIS"]
    return s


_PATCHES = {
    "_opt": _p_opt,
    "_meas": _p_meas,
    "_pre_anneal": _p_pre_anneal,
    "_gen_initial_measurements": _p_gen_initial_measurements,
    "_gen_pre_process_clean": _p_gen_pre_process_clean,
    "_gen_test_suite": _p_gen_test_suite,
}


@contextlib.contextmanager
def _patched():
    for name, fn in _PATCHES.items():
        setattr(gs, name, fn)
    try:
        yield
    finally:
        for name, fn in _ORIG.items():
            setattr(gs, name, fn)


def instrumented_generate(family: str, seed: int) -> tuple[list[str], list[tuple[float, object]]]:
    """Generate one sequence and the log-prob trace of every emitted decision."""
    rng = RecordingRandom(seed)
    with _patched():
        steps = gs.generate_sequence(family, rng)
    return steps, rng.rec


# --------------------------------------------------------------------------- #
# Floor                                                                        #
# --------------------------------------------------------------------------- #

def seq_floor_nats(records: list[tuple[float, object]], family_prior: float) -> float:
    """Total Bayes-optimal NLL (nats) for one sequence's scored tokens.

    = -sum(decision log-probs)  [steps]   -log P(family)  [the FAM token].
    s1 and EOS are deterministic (0 nats) and need no term.
    """
    return -sum(lp for lp, _ in records) - math.log(family_prior)


def scored_token_count(steps: list[str]) -> int:
    """Targets the model scores per sequence: FAM, s1..sK, EOS -> len(steps)+2."""
    return len(steps) + 2


def floor_stats(family: str, n: int, family_prior: float, seed0: int = 1_000_000) -> dict:
    """Monte-Carlo estimate of the exact floor for one family."""
    tot_nats = 0.0
    tot_tokens = 0
    lengths = []
    for i in range(n):
        steps, rec = instrumented_generate(family, seed0 + i)
        tot_nats += seq_floor_nats(rec, family_prior)
        tot_tokens += scored_token_count(steps)
        lengths.append(len(steps))
    lengths.sort()
    return {
        "family": family,
        "n": n,
        "family_prior": family_prior,
        "per_token_nll": tot_nats / tot_tokens,
        "per_seq_nll": tot_nats / n,
        "mean_len": sum(lengths) / n,
        "median_len": lengths[n // 2],
        "tot_tokens": tot_tokens,
    }


def selftest(n_per_family: int = 300, seed0: int = 0) -> bool:
    """Assert instrumented output is byte-identical to the stock generator.

    This is the guard that makes the floor trustworthy: if any patched function
    consumed the RNG differently, the sequences would diverge and this raises.
    """
    for family in FAMILIES:
        for seed in range(seed0, seed0 + n_per_family):
            instr, _ = instrumented_generate(family, seed)
            ref = gs.generate_sequence(family, random.Random(seed))  # stock (unpatched)
            if instr != ref:
                # find first divergence for a useful message
                k = next((j for j in range(min(len(instr), len(ref))) if instr[j] != ref[j]), -1)
                raise AssertionError(
                    f"FAITHFULNESS FAIL family={family} seed={seed} at step {k}: "
                    f"instr={instr[k] if k >= 0 else '<len>'} ref={ref[k] if k >= 0 else '<len>'} "
                    f"(len instr={len(instr)} ref={len(ref)})"
                )
    return True


def combined_floor(per_family: dict[str, dict], priors: dict[str, float]) -> float:
    """Token-weighted per-token NLL of the family mixture the model trains on."""
    num = sum(per_family[f]["per_seq_nll_at"][priors[f]] * per_family[f]["n"] for f in priors)
    den = sum(per_family[f]["tot_tokens"] for f in priors)
    return num / den


# --------------------------------------------------------------------------- #
# Falsification: a trained model must sit ABOVE this floor on fresh data       #
# --------------------------------------------------------------------------- #

def score_model_vs_floor(model, tok, family: str, n: int, family_prior: float,
                          device: str, seed0: int = 5_000_000) -> dict:
    """Compare a trained model's per-token NLL to the exact floor on FRESH seqs.

    Same sequences for both. Returns model/floor/gap (nats/token). gap >> 0 means
    real headroom remains; gap ~ 0 means the model is at the information limit for
    THIS distribution; gap < 0 means the floor (instrumentation) is buggy.
    """
    import torch  # lazy: keep the pure floor importable without torch

    model.eval()
    floor_nats = 0.0
    model_nats = 0.0
    tot_tokens = 0
    with torch.no_grad():
        for i in range(n):
            steps, rec = instrumented_generate(family, seed0 + i)
            ids = tok.encode_sequence(steps, family)
            ids = ids[: model.cfg.block_size + 1]
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            _, loss = model(x, y)  # mean CE over the (len-1) scored targets
            n_tok = len(ids) - 1
            model_nats += loss.item() * n_tok
            # floor over the SAME scored tokens; if truncated, scale by fraction
            full_tokens = scored_token_count(steps)
            seq_nats = seq_floor_nats(rec, family_prior)
            if n_tok < full_tokens:  # block_size truncation: prorate
                seq_nats *= n_tok / full_tokens
            floor_nats += seq_nats
            tot_tokens += n_tok
    m = model_nats / tot_tokens
    fl = floor_nats / tot_tokens
    return {"family": family, "n": n, "model_nll": m, "floor_nll": fl,
            "gap": m - fl, "ratio": m / fl if fl else float("inf")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="samples per family for the floor estimate")
    ap.add_argument("--selftest-n", type=int, default=300)
    args = ap.parse_args()

    print("=== FAITHFULNESS SELFTEST (instrumented == stock generator) ===")
    selftest(args.selftest_n)
    print(f"  PASS: instrumented output byte-identical to generate_sequence "
          f"for {args.selftest_n} seeds x {len(FAMILIES)} families\n")

    print(f"=== EXACT ENTROPY FLOOR (Bayes-optimal next-token NLL, n={args.n}/family) ===")
    print("  Two priors reported: ID = uniform over 3 trained families (1/3);")
    print("  LOFO = uniform over the 2 trained families (1/2).\n")
    per = {}
    for f in FAMILIES:
        content_nats = 0.0   # -sum(decision log-probs); excludes the family token
        tot_tokens = 0
        lens = []
        for i in range(args.n):
            steps, rec = instrumented_generate(f, 1_000_000 + i)
            content_nats += -sum(lp for lp, _ in rec)
            tot_tokens += scored_token_count(steps)
            lens.append(len(steps))
        lens.sort()
        fam_term_id = args.n * (-math.log(1 / 3))    # FAM-token cost, ID mixture
        fam_term_lofo = args.n * (-math.log(1 / 2))  # FAM-token cost, LOFO mixture
        per[f] = {
            "n": args.n, "tot_tokens": tot_tokens,
            "content_only": content_nats / tot_tokens,
            "per_token_id": (content_nats + fam_term_id) / tot_tokens,
            "per_token_lofo": (content_nats + fam_term_lofo) / tot_tokens,
            "mean_len": sum(lens) / args.n, "median_len": lens[args.n // 2],
            "per_seq_nll_at": {
                1 / 3: (content_nats + fam_term_id) / args.n,
                1 / 2: (content_nats + fam_term_lofo) / args.n,
            },
        }
        print(f"  {f:6}: per-token floor  ID(1/3)={per[f]['per_token_id']:.4f}  "
              f"LOFO(1/2)={per[f]['per_token_lofo']:.4f}  "
              f"content-only={per[f]['content_only']:.4f}  "
              f"(mean len {per[f]['mean_len']:.0f}, median {per[f]['median_len']})")

    # ID mixture floor (all 3 families, prior 1/3 each)
    id_priors = {f: 1 / 3 for f in FAMILIES}
    id_floor = combined_floor(per, id_priors)
    print(f"\n  >> ID mixture floor (3 families, 1/3 each): {id_floor:.4f} nats/token")
    print("     This is the hard lower bound for in-distribution next-token val loss.")
    print("     A model below it on held-out data => the floor is buggy (a falsifier).")


if __name__ == "__main__":
    main()
