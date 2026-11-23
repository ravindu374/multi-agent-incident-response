"""The history agent (Block 6c): the experience memory behind its own Agent Card ("history") and A2A server.

    lookup(HistoryQuery)       -> HistoryReport: the past incidents at least min_similarity alike (verified owner,
                                  the fix that worked, what every agent said), and every agent's verified record
                                  (controller/reliability.py), computed from the whole memory + the similar ones
    remember(RememberRequest)  -> RememberAck: store one closed incident with its verified outcome

The controller asks it once per incident, before routing, and sends it every verified outcome afterwards.
It reasons about nothing and proposes nothing: it only reports what was verified before.
"""
from __future__ import annotations

from pathlib import Path

from controller.cards import AgentCard
from controller.history import HistoryQuery, HistoryReport, RememberAck, RememberRequest
from controller.intake import extract
from controller.memory import Experience, ExperienceStore, past_case
from controller.reliability import ReliabilityModel


class HistoryAgent:
    def __init__(self, card: AgentCard, store: ExperienceStore | None = None, trust: str = "local",
                 cases: bool = True, path: Path | None = None):
        """trust: "local" / "global" (agents' verified record, Block 6b) or "off"; cases: hand out similar past
        incidents (False: the record only); path: save the store there after every remembered incident."""
        self.card, self.profile = card, card.routing_profile()
        if self.profile.role != "history":
            raise ValueError(f"{card.name} is not a history agent")
        if trust not in ("off", "local", "global"):
            raise ValueError("trust must be 'off', 'local' or 'global'")
        self.store = store if store is not None else ExperienceStore()
        self.trust, self.cases, self.path = trust, cases, path
        self._reliability = ReliabilityModel(self.store, mode=trust) if trust != "off" else None

    @property
    def agent_id(self) -> str:
        return self.card.name

    def lookup(self, query: HistoryQuery) -> HistoryReport:
        features = extract(query.incident)
        found = (self.store.similar(features, k=query.k, min_score=query.min_similarity, before=query.before)
                 if self.cases else [])
        return HistoryReport(incident_id=query.incident.incident_id, memory_size=len(self.store),
                             min_similarity=query.min_similarity, cases=[past_case(s, e) for s, e in found],
                             trust=self._reliability.table(features) if self._reliability else None)

    def remember(self, request: RememberRequest) -> RememberAck:
        experience = Experience.model_validate(request.experience)
        self.store.add(experience)
        if self.path is not None:
            self.store.save(self.path)
        return RememberAck(incident_id=experience.incident_id, stored=True, memory_size=len(self.store))
