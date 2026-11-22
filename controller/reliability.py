"""Block 6b - Learn each agent's reliability from the experience memory (verified outcomes only).

For a new incident, ReliabilityModel.table() counts what the memory verified about every agent:
  over the whole memory      every past incident (except this one)
  on incidents like this     the past incidents at least `min_similarity` alike (cosine of the Block 6a
                             embedding), at most `k`. Block 6a showed why the threshold matters: a past
                             incident of the same fault is ~1.00 alike, the nearest one of another fault
                             only ~0.67, and that one usually has another owner. Below the threshold the
                             memory says nothing local, and only the whole-memory numbers are used.
What is counted (controller/trust.py turns counts into numbers):
  success        the agent's proposals where its domain owned the incident: right fix = success; wrong fix
                 or no fix = failure; an unchecked ("unknown") fix is left out
  fix by verdict the same, split by the verdict it gave (owner / unsure / not mine)
  verdicts       how often each verdict came when its domain owned the incident, and when it did not
  reviews        per stance, whether the reviewed fix was right (same-domain reviews) or its domain owned
                 the incident (other-domain reviews)
Only labels from verified feedback count (Block 6a): never the agent's own confidence, never an LLM judge.
"""
from __future__ import annotations

from .contracts import IncidentFeatures
from .memory import Experience, ExperienceStore
from .trust import AgentRecord, Counts, TrustTable


def count(experiences: list[Experience]) -> dict[str, AgentRecord]:
    records: dict[str, AgentRecord] = {}
    for e in experiences:
        for p in e.proposals:
            r = records.setdefault(p.agent_id, AgentRecord())
            if p.domain_owned is None:
                continue
            side = r.verdict_owned if p.domain_owned else r.verdict_not_owned
            side[p.verdict] = side.get(p.verdict, 0) + 1
            if p.domain_owned and p.label != "unknown":
                r.success.n += 1
                r.success.hits += p.label == "right"
                if p.label is not None:
                    c = r.fix.setdefault(p.verdict, Counts())
                    c.n += 1
                    c.hits += p.label == "right"
        for v in e.reviews:
            same = v.domain == v.claim_domain
            if same:
                if v.claim_label not in ("right", "wrong"):
                    continue  # the reviewed fix was never checked
                truth = v.claim_label == "right"
            elif v.claim_owned is None:
                continue
            else:
                truth = v.claim_owned
            r = records.setdefault(v.reviewer, AgentRecord())
            key = f"{'same' if same else 'other'}:{v.stance}"
            side = r.review_true if truth else r.review_false
            side[key] = side.get(key, 0) + 1
    return records


class ReliabilityModel:
    """mode "local": whole memory + similar incidents; "global": whole memory only."""

    def __init__(self, memory: ExperienceStore, mode: str = "local", min_similarity: float = 0.85, k: int = 30,
                 prior_strength: float = 4.0, near_strength: float = 2.0):
        if mode not in ("local", "global"):
            raise ValueError("mode must be 'local' or 'global'")
        self.memory, self.mode, self.min_similarity, self.k = memory, mode, min_similarity, k
        self.prior_strength, self.near_strength = prior_strength, near_strength
        self._cache: tuple[int, dict[str, AgentRecord]] | None = None

    def _global(self, exclude: str) -> dict[str, AgentRecord]:
        if any(e.incident_id == exclude for e in self.memory.experiences):  # never learn from itself
            return count([e for e in self.memory.experiences if e.incident_id != exclude])
        if self._cache is None or self._cache[0] != self.memory.version:  # the memory changed: count again
            self._cache = (self.memory.version, count(self.memory.experiences))
        return self._cache[1]

    def table(self, features: IncidentFeatures) -> TrustTable:
        near, neighbours = {}, 0
        if self.mode == "local":
            found = self.memory.similar(features, k=self.k, min_score=self.min_similarity)
            near, neighbours = count([e for _, e in found]), len(found)
        return TrustTable(mode=self.mode, min_similarity=self.min_similarity, neighbours=neighbours,
                          prior_strength=self.prior_strength, near_strength=self.near_strength,
                          agents=self._global(features.incident_id), near=near)
