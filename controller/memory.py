"""Block 6a - Experience memory: what happened on past incidents, verified, and searchable by similarity.

When an incident is closed, the controller stores one Experience:
  the incident   its masked text, signals, alerts and components (Block 1), and an embedding vector
  the decision   routing (classifier probabilities, team, priority), every agent's proposal, every review,
                 the plan (kind and stages)
  the outcome    VERIFIED feedback only, never an LLM's opinion: the executor's report (which actions ran,
                 resolved or not by the platform's own health checks) and, when a person was involved, the
                 operator's postmortem (which domain owned it, which agents' fixes would have fixed it)
  labels         per proposal: right / wrong / unknown; per review: right / wrong (see label_proposal)

Labels follow what was verified. Feedback arrives mostly for what was TRIED (bandit feedback): when the
controller fixed an incident on its own, nobody checks the fixes it did not run.
  right    the operator said so, or it is the fix the incident was resolved by (checks.same_fix)
  wrong    the operator said so, or it ran and the incident was still broken afterwards, or it was a fix in
           another domain than the verified owner
  unknown  not tried, not the fix that worked, but in the owner's domain: it may have been another valid
           fix (S1 and N3 have two), so calling it wrong would teach the learner something false

Similarity search: a fixed hashing embedder (no training, no download, so old vectors never change as the
store grows) over the masked text's words and word pairs plus the signal / alert / component names; cosine
similarity. Masking makes two incidents that differ only in pids, times or addresses look the same.
Storage: a NumPy matrix in memory + JSON lines on disk, enough for thousands of incidents. Behind the same
methods (add, similar, save, load) it could live in LanceDB or Chroma, or in Fabric as an Eventhouse table
with vector search or in Azure AI Search (indicative: check the current Fabric documentation).

Block 6a only REMEMBERS: nothing reads the memory to decide yet. Block 6b turns the labels into each agent's
learned reliability (replacing the fixed tier precisions and verdict factors in controller/optimizer.py),
and Block 6c shows similar past cases to the agents.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from . import DOMAINS
from .checks import same_fix
from .contracts import IncidentFeatures, IncidentReport
from .intake import extract
from .learned import structured, tokens
from .orchestrator import Deliberation
from .plan import ExecutionReport, PlannedAction, RemediationPlan

FORMAT = "a2a-oncall/experience-memory"
VERSION = 1
Label = Literal["right", "wrong", "unknown"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))  # Python 3.10 does not read a trailing Z


# -- verified feedback ------------------------------------------------------------------------------

class Feedback(BaseModel):
    """The verified outcome of one incident: the executor's report, plus the operator's postmortem when a
    person was involved (an approval, an escalation, or a plan that did not fix it)."""

    incident_id: str
    resolved: bool  # from the platform's own health checks
    source: Literal["executor", "operator"] = "executor"
    owner: str | None = None  # the domain whose fix resolved it (postmortem); None: inferred from fixed_by
    fixed_by: list[PlannedAction] = Field(default_factory=list)  # the stage after which it was healthy again
    failed: list[PlannedAction] = Field(default_factory=list)  # actions that ran and did not fix it
    credits: dict[str, bool] | None = None  # postmortem: agent id -> "its fix would have fixed it"
    verified_at: str | None = None

    @model_validator(mode="after")
    def _check_owner(self) -> Feedback:
        if self.owner is not None and self.owner not in DOMAINS:
            raise ValueError(f"owner must be one of {DOMAINS}")
        return self

    @classmethod
    def from_report(cls, plan: RemediationPlan, report: ExecutionReport, owner: str | None = None,
                    credits: dict[str, bool] | None = None) -> Feedback:
        """Read the executor's report against the plan. The executor reports one outcome per plan action, in
        the plan's order (docs/HANDOVER.md); if the counts differ, outcomes are matched by intent in order."""
        if report.resolved is None:
            raise ValueError(f"{report.incident_id}: not verified yet (resolved is None); learn only from "
                             "verified reports")
        actions = list(plan.actions)
        if len(report.outcomes) == len(actions):
            pairs = list(zip(actions, report.outcomes))
        else:
            pairs, unused = [], list(actions)
            for outcome in report.outcomes:
                match = next((a for a in unused if a.intent == outcome.intent), None)
                if match is not None:
                    unused.remove(match)
                    pairs.append((match, outcome))
        done = [a for a, o in pairs if o.status == "done"]
        if report.resolved and done:
            last = max(a.stage for a in done)
            fixed_by, failed = [a for a in done if a.stage == last], [a for a in done if a.stage < last]
        else:
            fixed_by, failed = [], done
        return cls(incident_id=report.incident_id, resolved=report.resolved, fixed_by=fixed_by, failed=failed,
                   owner=owner, credits=credits, verified_at=report.verified_at,
                   source="operator" if owner or credits is not None else "executor")


# -- one experience ------------------------------------------------------------------------------------

class ProposalRecord(BaseModel):
    agent_id: str
    domain: str
    tier: str
    reasoner: str = ""
    verdict: str
    confidence: float
    points_to: str | None = None
    root_cause: str = ""
    actions: list[PlannedAction] = Field(default_factory=list)
    label: Label | None = None  # about its fix; None = it handed in no fix
    label_why: str = ""
    domain_owned: bool | None = None  # was its domain the verified owner? None = owner unknown


class ReviewRecord(BaseModel):
    reviewer: str
    domain: str
    claim_agent: str
    claim_domain: str
    stance: str
    right: bool | None = None  # supported a right fix / contradicted a wrong one; None = cannot tell or unknown


class Experience(BaseModel):
    incident_id: str
    opened_at: str | None = None
    text: str
    alerts: list[str] = Field(default_factory=list)
    signals: dict[str, int] = Field(default_factory=dict)
    components: list[str] = Field(default_factory=list)
    priority: str
    owner_probs: dict[str, float]  # what the classifier believed before any agent answered
    team: list[str]
    proposals: list[ProposalRecord] = Field(default_factory=list)
    reviews: list[ReviewRecord] = Field(default_factory=list)
    plan_kind: str
    plan_actions: list[PlannedAction] = Field(default_factory=list)
    plan_confidence: float
    outcome: Feedback
    owner: str | None = None  # verified owner: from the postmortem, or the domain whose fix resolved it
    recorded_at: str = Field(default_factory=_now)

    def features(self) -> IncidentFeatures:
        return IncidentFeatures(incident_id=self.incident_id, alerts=self.alerts, signals=self.signals,
                                components=self.components, error_lines=[], text=self.text)

    def fix_that_worked(self) -> list[PlannedAction]:
        return self.outcome.fixed_by


def verified_owner(d: Deliberation, fb: Feedback) -> str | None:
    """The postmortem's owner, else the domain of the agents whose fix resolved it, else unknown."""
    if fb.owner:
        return fb.owner
    if fb.fixed_by:
        domains = Counter(p.domain for p in d.proposals if p.actions and same_fix(p.actions, fb.fixed_by))
        if domains:
            return domains.most_common(1)[0][0]
    return None


def label_proposal(p, fb: Feedback, owner: str | None) -> tuple[Label | None, str]:
    if not p.actions:
        return None, "no fix handed in"
    if fb.credits is not None and p.agent_id in fb.credits:
        return ("right" if fb.credits[p.agent_id] else "wrong"), "operator's postmortem"
    if fb.fixed_by and same_fix(p.actions, fb.fixed_by):
        return "right", "the fix that resolved it"
    if fb.failed and same_fix(p.actions, fb.failed):
        return "wrong", "it ran and the incident stayed broken"
    if owner and p.domain != owner:
        return "wrong", f"the owner was {owner}"
    return "unknown", "not tried"


def experience_from(report: IncidentReport, d: Deliberation, plan: RemediationPlan, feedback: Feedback,
                    tiers: dict[str, str]) -> Experience:
    """Everything worth remembering about one closed incident, labelled from the verified feedback."""
    if feedback.incident_id != report.incident_id:
        raise ValueError(f"feedback for {feedback.incident_id} given for incident {report.incident_id}")
    f = extract(report)
    owner = verified_owner(d, feedback)
    proposals, labels = [], {}
    for p in d.proposals:
        label, why = label_proposal(p, feedback, owner)
        labels[(p.agent_id, p.domain)] = label
        proposals.append(ProposalRecord(
            agent_id=p.agent_id, domain=p.domain, tier=tiers.get(p.agent_id, "fast"), reasoner=p.reasoner,
            verdict=p.verdict, confidence=p.confidence, points_to=p.points_to, root_cause=p.root_cause,
            actions=p.actions, label=label, label_why=why, domain_owned=None if owner is None else p.domain == owner))
    reviews = []
    for r in d.reviews:
        label = labels.get((r.claim_agent, r.claim_domain))
        right = None
        if r.stance != "cannot_tell" and label in ("right", "wrong"):
            right = (r.stance == "supports") == (label == "right")
        reviews.append(ReviewRecord(reviewer=r.reviewer, domain=r.domain, claim_agent=r.claim_agent,
                                    claim_domain=r.claim_domain, stance=r.stance, right=right))
    return Experience(
        incident_id=report.incident_id, opened_at=report.opened_at, text=f.text, alerts=f.alerts,
        signals=f.signals, components=f.components, priority=d.routing.priority,
        owner_probs=d.routing.owner_probs, team=d.routing.team, proposals=proposals, reviews=reviews,
        plan_kind=plan.kind, plan_actions=plan.actions, plan_confidence=plan.confidence, outcome=feedback,
        owner=owner)


# -- similarity --------------------------------------------------------------------------------------

class HashingEmbedder:
    """Words and word pairs of the masked text, plus the signal / alert / component names, hashed into `dim`
    buckets with a random sign (the hashing trick); counts are sublinear (1 + log n); the vector is
    L2-normalised, so a dot product is the cosine similarity. Nothing is trained, so an experience's vector
    never changes when more experiences arrive."""

    name = "hashing-v1"

    def __init__(self, dim: int = 2048, structured_weight: float = 2.0):
        self.dim, self.structured_weight = dim, structured_weight

    def spec(self) -> dict:
        return {"name": self.name, "dim": self.dim, "structured_weight": self.structured_weight}

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if h >> 63 else -1.0)  # Python's hash() changes per process: not usable

    def embed(self, features: IncidentFeatures) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        weights = {tok: 1.0 + math.log(n) for tok, n in Counter(tokens(features.text)).items()}
        for tok in structured(features):
            weights[tok] = weights.get(tok, 0.0) + self.structured_weight
        for tok, w in weights.items():
            i, sign = self._bucket(tok)
            v[i] += sign * w
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v


EMBEDDERS = {HashingEmbedder.name: HashingEmbedder}


class ExperienceStore:
    """Experiences + their vectors. `similar` is a brute-force cosine search (exact; fast at this size)."""

    def __init__(self, embedder: HashingEmbedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.experiences: list[Experience] = []
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.experiences)

    def add(self, experience: Experience) -> None:
        """Store one closed incident. The same incident again replaces the old record (a corrected postmortem)."""
        vector = self.embedder.embed(experience.features())
        for i, old in enumerate(self.experiences):
            if old.incident_id == experience.incident_id:
                self.experiences[i], self._vectors[i] = experience, vector
                break
        else:
            self.experiences.append(experience)
            self._vectors.append(vector)
        self._matrix = None

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (np.vstack(self._vectors) if self._vectors
                            else np.zeros((0, self.embedder.dim), dtype=np.float32))
        return self._matrix

    def similar(self, query: IncidentReport | IncidentFeatures, k: int = 5, before: str | None = None,
                exclude: set[str] | frozenset[str] = frozenset(), min_score: float = 0.0
                ) -> list[tuple[float, Experience]]:
        """The k most similar past experiences, most similar first. before: only incidents opened earlier
        (never learn from the future); exclude: incident ids to leave out (e.g. the query itself)."""
        if not self.experiences:
            return []
        features = extract(query) if isinstance(query, IncidentReport) else query
        scores = self.matrix @ self.embedder.embed(features)
        cutoff = _time(before)
        out = []
        for i in np.argsort(-scores, kind="stable"):
            e = self.experiences[i]
            if float(scores[i]) < min_score:
                break
            if e.incident_id in exclude or e.incident_id == features.incident_id:
                continue
            if cutoff is not None and (_time(e.opened_at) is None or _time(e.opened_at) >= cutoff):
                continue
            out.append((float(scores[i]), e))
            if len(out) == k:
                break
        return out

    # -- what the memory says, in counts (Block 6b turns these into probabilities) ----------------------
    def summary(self) -> dict:
        fixes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        domains: dict[tuple[str, str], Counter] = defaultdict(Counter)
        reviews: dict[str, Counter] = defaultdict(Counter)
        for e in self.experiences:
            for p in e.proposals:
                if p.label is not None:
                    fixes[(p.agent_id, p.verdict)][p.label] += 1
                if p.domain_owned is not None:
                    domains[(p.agent_id, p.verdict)]["owned" if p.domain_owned else "not owned"] += 1
            for r in e.reviews:
                reviews[r.reviewer]["cannot tell" if r.right is None else ("right" if r.right else "wrong")] += 1
        return {
            "experiences": len(self),
            "plans": dict(Counter(e.plan_kind for e in self.experiences)),
            "sources": dict(Counter(e.outcome.source for e in self.experiences)),
            "owners": dict(Counter(e.owner or "unknown" for e in self.experiences)),
            "fixes": {f"{a} {v}": dict(c) for (a, v), c in sorted(fixes.items())},
            "domain_owned": {f"{a} {v}": dict(c) for (a, v), c in sorted(domains.items())},
            "reviews": {a: dict(c) for a, c in sorted(reviews.items())},
        }

    # -- disk ------------------------------------------------------------------------------------------
    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "experiences.jsonl").write_text(
            "".join(e.model_dump_json() + "\n" for e in self.experiences), encoding="utf-8")
        np.save(folder / "vectors.npy", self.matrix)
        meta = {"format": FORMAT, "version": VERSION, "embedder": self.embedder.spec(), "experiences": len(self),
                "saved_at": _now()}
        (folder / "store.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: Path) -> ExperienceStore:
        folder = Path(folder)
        meta = json.loads((folder / "store.json").read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT or meta.get("version") != VERSION:
            raise ValueError(f"{folder} is not a {FORMAT} v{VERSION} store")
        spec = dict(meta["embedder"])
        name = spec.pop("name")
        if name not in EMBEDDERS:
            raise ValueError(f"unknown embedder {name!r}; known: {sorted(EMBEDDERS)}")
        store = cls(EMBEDDERS[name](**spec))
        lines = (folder / "experiences.jsonl").read_text(encoding="utf-8").splitlines()
        store.experiences = [Experience.model_validate_json(line) for line in lines if line.strip()]
        matrix = np.load(folder / "vectors.npy")
        if matrix.shape != (len(store.experiences), store.embedder.dim):
            raise ValueError(f"{folder}: {matrix.shape[0]} vectors for {len(store.experiences)} experiences")
        store._vectors = list(matrix)
        store._matrix = matrix
        return store


def format_neighbours(neighbours: list[tuple[float, Experience]]) -> str:
    """What the memory remembers about incidents like this one: verified owner, the fix that worked, and what
    each agent said (with its label)."""
    out = []
    for score, e in neighbours:
        fix = "; ".join(f"{a.intent} on {a.target} {a.params}" for a in e.fix_that_worked()) or "fixed by a person"
        out.append(f"  {score:.2f}  {e.incident_id}  {e.opened_at or '?'}  owner={e.owner or '?'}  plan={e.plan_kind}"
                   f"  ({e.outcome.source})")
        out.append(f"        alerts: {', '.join(e.alerts)}   fix that worked: {fix}")
        for p in e.proposals:
            if p.actions:
                said = "; ".join(f"{a.intent} {a.params}" for a in p.actions)
                out.append(f"        {p.agent_id:<16} {p.verdict:<8} {said}  -> {p.label} ({p.label_why})")
            else:
                out.append(f"        {p.agent_id:<16} {p.verdict:<8} no fix")
    return "\n".join(out)
