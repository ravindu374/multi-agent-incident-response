"""Block 4 - Router: from an incident to a primary agent, consults and fallbacks.

  1. Block 1 + 2: features -> owner probabilities.
  2. Team of domains: the conformal prediction set (calibrated coverage), or choose_team's fixed
     thresholds when no calibrator is given.
  3. Per team domain: the registry's hard filter gives the eligible agents (Block 3); each is scored
         utility = value(priority) * P(owner = domain) * P(agent succeeds) - cost - latency_weight * latency
     and the best one is chosen; the rest become ordered fallbacks.
  4. The owner's agent is the primary; the others are consults. Domains with no eligible agent are
     left unassigned and a human is notified.

Priority changes how much spending is worth (the value of fixing), not which agent "looks" best.
P(agent succeeds) is a prior per tier, lowered on hard (ambiguous) incidents. With an experience memory
(Block 6b, a `trust_model`), it becomes the agent's verified record, on the whole memory and on incidents
like this one (controller/trust.py); the table travels in the RoutingDecision for the later rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .cards import RoutingProfile
from .classifier import DomainClassifier, choose_team
from .conformal import ConformalCalibrator
from .contracts import IncidentReport
from .intake import extract
from .registry import AgentRegistry, Candidate
from .trust import TrustTable

ALERT_PRIORITY = {
    "DashboardDown": "P1", "LoginFailureSpike": "P1", "UnexpectedWarehouseClient": "P1",
    "OrdersMartStale": "P2", "DashboardSlow": "P2", "WarehouseConnectionsSaturated": "P2",
    "IngestJobFailing": "P3", "TransformJobFailing": "P3", "MonitoringGap": "P3",
}
PRIORITY_VALUE = {"P1": 30.0, "P2": 15.0, "P3": 8.0}  # value of resolving, in agent cost units


def priority_of(report: IncidentReport) -> str:
    levels = [ALERT_PRIORITY.get(a.name, "P3") for a in report.alerts]
    return min(levels) if levels else "P3"  # "P1" < "P2" < "P3"


class SuccessModel:
    """Prior chance that an agent resolves a task in its domain, lower on ambiguous incidents.
    Planning assumptions, not measurements: Block 6 learns the real values from outcomes."""

    PRIOR = {"fast": 0.70, "strong": 0.88, "generalist": 0.78}
    DIFFICULTY_PENALTY = {"fast": 0.45, "strong": 0.15, "generalist": 0.20}

    def expected_success(self, profile: RoutingProfile, domain: str, difficulty: float) -> float:
        kind = "generalist" if profile.role == "generalist" else profile.tier
        p = self.PRIOR[kind] - self.DIFFICULTY_PENALTY[kind] * difficulty
        return min(0.99, max(0.01, p))


@dataclass
class RoutingPolicy:
    # "auto": conformal set if a calibrator is given, else fixed thresholds.
    # "top1" / "all": baselines (only the most likely domain / every domain).
    team_rule: str = "auto"
    latency_weight: float = 0.05  # cost units per second of expected latency
    max_team: int = 3
    confident_at: float = 0.8  # only used without a calibrator
    coverage: float = 0.9  # only used without a calibrator
    priority_value: dict[str, float] = field(default_factory=lambda: dict(PRIORITY_VALUE))


class AgentChoice(BaseModel):
    domain: str
    agent_id: str
    tier: str
    p_owner: float
    expected_success: float
    cost_units: float
    expected_latency_s: float
    utility: float
    fallbacks: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    incident_id: str
    classifier: str
    calibrated: bool
    alpha: float | None = None
    priority: str
    owner_probs: dict[str, float]
    team: list[str]  # domains, most likely owner first
    confident: bool
    primary: AgentChoice | None = None
    consults: list[AgentChoice] = Field(default_factory=list)
    unassigned: list[str] = Field(default_factory=list)
    expected_cost: float = 0.0
    notify_human: bool = False
    reasons: list[str] = Field(default_factory=list)
    trust: TrustTable | None = None  # Block 6b: agents' verified record (None = planning numbers)

    @property
    def agents(self) -> list[str]:
        return ([self.primary.agent_id] if self.primary else []) + [c.agent_id for c in self.consults]


class Router:
    def __init__(self, classifier: DomainClassifier, registry: AgentRegistry,
                 calibrator: ConformalCalibrator | None = None, policy: RoutingPolicy | None = None,
                 success: SuccessModel | None = None, trust_model=None):
        self.classifier = classifier
        self.registry = registry
        self.calibrator = calibrator
        self.policy = policy or RoutingPolicy()
        self.success = success or SuccessModel()
        self.trust_model = trust_model  # Block 6b: .table(features) -> TrustTable (controller/reliability.py)

    def _team(self, scores) -> tuple[list[str], bool, str]:
        rule = self.policy.team_rule
        if rule in ("top1", "all"):
            ranked = [d for d, _ in scores.ranked()]
            team = ranked[:1] if rule == "top1" else ranked
            return team, len(team) == 1, f"baseline team rule '{rule}'"
        if rule != "auto":
            raise ValueError(f"unknown team rule {rule!r}")
        if self.calibrator is not None:
            team = self.calibrator.prediction_set(scores.owner_probs)[: self.policy.max_team]
            return team, len(team) == 1, self.calibrator.describe()
        t = choose_team(scores, self.policy.confident_at, self.policy.coverage, self.policy.max_team)
        return t.team, t.confident, t.reason

    def _score(self, c: Candidate, domain: str, p_owner: float, value: float, difficulty: float,
               trust: TrustTable | None = None) -> AgentChoice:
        p_success = self.success.expected_success(c.profile, domain, difficulty)
        if trust is not None:  # the planning guess, moved towards the agent's verified record
            p_success = min(0.99, max(0.01, trust.success(c.agent_id, p_success)))
        utility = (value * p_owner * p_success - c.profile.cost_units
                   - self.policy.latency_weight * c.profile.expected_latency_s)
        return AgentChoice(domain=domain, agent_id=c.agent_id,
                           tier="generalist" if c.profile.role == "generalist" else c.profile.tier,
                           p_owner=p_owner, expected_success=round(p_success, 3), cost_units=c.profile.cost_units,
                           expected_latency_s=c.profile.expected_latency_s, utility=round(utility, 3))

    def pick_agent(self, domain: str, decision: RoutingDecision, p_owner: float | None = None,
                   exclude: set[str] | frozenset[str] = frozenset(), by: str = "utility",
                   now: float | None = None) -> AgentChoice | None:
        """Best eligible agent for one more domain of an already-routed incident.

        by="utility": as in route() (e.g. an ownership transfer: another agent's evidence points there, so
        the caller may pass a higher p_owner than the classifier's). by="success": the agent most likely
        to succeed, cheaper first on ties (escalation after cheaper agents found nothing)."""
        candidates, _ = self.registry.eligible(domain, now=now)
        candidates = [c for c in candidates if c.agent_id not in exclude]
        if not candidates:
            return None
        value = self.policy.priority_value[decision.priority]
        difficulty = 1.0 - max(decision.owner_probs.values())
        p = decision.owner_probs.get(domain, 0.0) if p_owner is None else p_owner
        scored = [self._score(c, domain, p, value, difficulty, decision.trust) for c in candidates]
        key = (lambda a: (-a.expected_success, a.cost_units)) if by == "success" else (lambda a: -a.utility)
        ranked = sorted(scored, key=key)
        best = ranked[0]
        best.fallbacks = [a.agent_id for a in ranked[1:]]
        return best

    def route(self, report: IncidentReport, now: float | None = None, trust: TrustTable | None = None) -> RoutingDecision:
        """trust: the agents' verified record (Block 6c: from the history agent); else from trust_model, if any."""
        features = extract(report)
        scores = self.classifier.predict(features)
        team, confident, team_reason = self._team(scores)
        priority = priority_of(report)
        value = self.policy.priority_value[priority]
        difficulty = 1.0 - max(scores.owner_probs.values())
        reasons = [f"team {team}: {team_reason}", f"priority {priority} (value {value:g})"]
        if trust is None and self.trust_model is not None:
            trust = self.trust_model.table(features)
        if trust is not None:
            reasons.append(f"agents' verified record: {trust.mode}, {trust.neighbours} past incidents at least "
                           f"{trust.min_similarity:.2f} alike")

        choices: list[AgentChoice] = []
        unassigned: list[str] = []
        used: set[str] = set()
        for domain in team:
            candidates, excluded = self.registry.eligible(domain, now=now)
            for x in excluded:
                reasons.append(f"{domain}: skipped {x.agent_id} ({x.reason})")
            if not candidates:
                unassigned.append(domain)
                reasons.append(f"{domain}: no eligible agent")
                continue
            ranked = sorted((self._score(c, domain, scores.owner_probs[domain], value, difficulty, trust)
                             for c in candidates), key=lambda a: -a.utility)
            best = next((a for a in ranked if a.agent_id not in used), ranked[0])  # prefer distinct agents
            best.fallbacks = [a.agent_id for a in ranked if a.agent_id != best.agent_id]
            used.add(best.agent_id)
            choices.append(best)
            if trust is not None:
                reasons.append(f"{domain}: " + ", ".join(
                    f"{a.agent_id} P(success)={a.expected_success:.2f} ({'/'.join(map(str, trust.cases(a.agent_id)))}"
                    " cases all/near)" for a in ranked))

        primary = choices[0] if choices else None
        if primary and primary.domain != team[0]:
            reasons.append(f"most likely owner {team[0]} has no eligible agent; {primary.domain} leads instead")
        notify = bool(unassigned) or (priority == "P1" and not confident)
        if priority == "P1" and not confident:
            reasons.append("P1 incident with an uncertain owner: notify the on-call human")
        consults = choices[1:]
        distinct_cost = {c.agent_id: c.cost_units for c in choices}
        return RoutingDecision(
            incident_id=report.incident_id, classifier=self.classifier.name,
            calibrated=self.calibrator is not None, alpha=self.calibrator.alpha if self.calibrator else None,
            priority=priority, owner_probs=scores.owner_probs, team=team, confident=confident,
            primary=primary, consults=consults, unassigned=unassigned,
            expected_cost=round(sum(distinct_cost.values()), 2), notify_human=notify, reasons=reasons, trust=trust,
        )
