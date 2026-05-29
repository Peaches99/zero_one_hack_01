"""process_lm — a small from-scratch GPT for semiconductor process-step sequences.

Each fab step string is one token. One autoregressive model serves all tasks:
next-step prediction (rank the distribution), sequence completion (roll it
forward), and — later — anomaly detection (watch where it is surprised).
"""

__version__ = "0.1.0"
