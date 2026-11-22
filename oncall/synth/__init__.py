"""Synthetic incident generator: development data that looks like the sandbox's output.

Each incident is simulated as platform *state* over time (job runs, freshness, dashboard
latency, connections, ...). That state goes through the real health oracle and incident
detector, so alert texts and firing order are produced by the same code the sandbox uses.
Logs use the services' exact formats. Every incident also carries an evidence bundle (one
slice per expert domain) and gets random variants, red herrings and missing evidence.

Development data only. Final results must come from real sandbox runs.
"""
from .generate import GENERATOR_VERSION, generate, write_dataset

__all__ = ["GENERATOR_VERSION", "generate", "write_dataset"]
