"""process_lm — a small from-scratch GPT for semiconductor process-step sequences.

Each fab step string is one token. One autoregressive model serves all tasks:
next-step prediction (rank the distribution), sequence completion (roll it
forward), and — later — anomaly detection (watch where it is surprised).
"""
import os

# Cap Apple MPS memory BEFORE torch is imported anywhere in the package.
# On Apple Silicon, MPS shares the machine's unified RAM, and PyTorch's default
# high-watermark ratio (1.7) lets it over-commit relative to recommended max —
# which can push a low-RAM Mac into swap-death and force a reboot. PyTorch
# requires this ratio to be 0.0 (unbounded) or >= 1.0; we set 1.5, a valid bound
# below the 1.7 default so a runaway allocation raises a catchable "MPS out of
# memory" instead of a hard crash. Our model is tiny (<1 GB peak), so this never
# limits a legitimate single run. Override by exporting the var yourself.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.5")

__version__ = "0.1.0"
