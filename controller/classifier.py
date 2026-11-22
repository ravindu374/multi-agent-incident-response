"""Block 2 - Domain classification: which domains own this incident, and who should work on it.

Every classifier maps IncidentFeatures -> DomainScores:
  owner_probs     P(owner = d), sums to 1. The owner is the domain whose fix resolves the
                  root cause (single label).
  involved_probs  P(domain d holds useful evidence), independent per domain (multi-label).

choose_team() turns scores into a domain-level team: just the owner when the classifier is
confident, otherwise the owner plus the other plausible owners. Block 4 (router) will map
domains to concrete agents and replace the fixed thresholds here with conformal calibration
fitted on real incidents.

Implemented so far: RuleClassifier, the "static keyword rules" baseline (Baseline A).
"""
from __future__ import annotations

import math
from typing import Protocol

from pydantic import BaseModel, Field

from . import DOMAINS
from .contracts import IncidentFeatures


class DomainScores(BaseModel):
    incident_id: str
    method: str
    owner_probs: dict[str, float]
    involved_probs: dict[str, float]
    evidence: dict[str, list[str]] = Field(default_factory=dict)  # domain -> signals pointing there

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.owner_probs.items(), key=lambda kv: (-kv[1], DOMAINS.index(kv[0])))


class DomainTeam(BaseModel):
    incident_id: str
    owner: str
    owner_prob: float
    confident: bool
    team: list[str]  # owner first, then the other plausible owners (consults)
    reason: str


class DomainClassifier(Protocol):
    name: str

    def predict(self, features: IncidentFeatures) -> DomainScores: ...


def softmax(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    top = max(scores.values())
    exps = {d: math.exp((s - top) / temperature) for d, s in scores.items()}
    total = sum(exps.values())
    return {d: e / total for d, e in exps.items()}


# -- Baseline A: static keyword rules ---------------------------------------------------------
# Written from what each signal *means* to an on-call engineer, not tuned on any data.
# Ambiguous signals split their weight (a statement timeout can be a heavy query or a starved CPU).

RULE_WEIGHTS: dict[str, dict[str, float]] = {
    "network_timeout": {"network": 1.5, "infra": 0.5},
    "connection_reset": {"network": 2.0},
    "connect_failed": {"network": 1.0, "infra": 0.5},
    "name_resolution": {"network": 2.0},
    "lock_timeout": {"data": 2.0},
    "statement_timeout": {"data": 1.0, "infra": 1.0},
    "too_many_clients": {"data": 1.5, "security": 0.5},
    "db_saturation": {"data": 1.0, "security": 0.5},
    "auth_failed": {"security": 1.0, "data": 0.5},
    "schema_mismatch": {"data": 2.5},
    "data_test_failed": {"data": 2.5},
    "oom_killed": {"infra": 2.5},
    "disk_full": {"infra": 2.5},
    "config_invalid": {"infra": 2.0},
    "service_down": {"infra": 1.5},
    "http_5xx": {"infra": 0.5, "data": 0.5},
    "high_latency": {"infra": 1.0, "data": 0.5, "network": 0.5},
    "stale_data": {"data": 0.5},
    "login_failures": {"security": 2.5},
    "unexpected_db_client": {"security": 2.5},
}


class RuleClassifier:
    """Sum the rule weights of every signal present, then softmax into owner probabilities.

    Signals count by presence, not frequency: the same error repeated in an alert, a log
    line and a job record is one fact, not three.
    """

    name = "rules"

    def __init__(self, weights: dict[str, dict[str, float]] | None = None, temperature: float = 0.5):
        self.weights = weights or RULE_WEIGHTS
        self.temperature = temperature

    def predict(self, features: IncidentFeatures) -> DomainScores:
        evidence = {d: 0.0 for d in DOMAINS}
        pointed: dict[str, list[str]] = {d: [] for d in DOMAINS}
        for signal in features.signals:
            for domain, weight in self.weights.get(signal, {}).items():
                evidence[domain] += weight
                pointed[domain].append(signal)
        return DomainScores(
            incident_id=features.incident_id,
            method=self.name,
            owner_probs=softmax(evidence, self.temperature),
            involved_probs={d: 1.0 - math.exp(-e) for d, e in evidence.items()},
            evidence={d: s for d, s in pointed.items() if s},
        )


# -- from scores to a team ----------------------------------------------------------------------

def choose_team(scores: DomainScores, confident_at: float = 0.8, coverage: float = 0.9,
                max_team: int = 3) -> DomainTeam:
    """Confident -> the owner alone. Unsure -> the smallest set of most likely owners whose
    probabilities add up to `coverage` (capped at `max_team`), so the true owner is very
    likely on the team even when the top guess is wrong.

    The defaults are a-priori choices; Block 4 calibrates them on real incidents.
    """
    ranked = scores.ranked()
    owner, p_owner = ranked[0]
    if p_owner >= confident_at:
        return DomainTeam(incident_id=scores.incident_id, owner=owner, owner_prob=p_owner, confident=True,
                          team=[owner], reason=f"confident: P(owner={owner}) = {p_owner:.2f} >= {confident_at}")
    team, mass = [], 0.0
    for domain, p in ranked:
        team.append(domain)
        mass += p
        if mass >= coverage or len(team) == max_team:
            break
    return DomainTeam(
        incident_id=scores.incident_id, owner=owner, owner_prob=p_owner, confident=False, team=team,
        reason=f"unsure: P(owner={owner}) = {p_owner:.2f} < {confident_at}; "
               f"team covers {mass:.0%} of the owner probability",
    )
