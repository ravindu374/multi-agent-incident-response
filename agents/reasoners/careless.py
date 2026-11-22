"""A deliberately unreliable reasoner, for robustness experiments (never for real use).

It behaves like an over-eager LLM: when the rule checks find the real cause it reports it, otherwise it
still claims ownership with high confidence and a fix that *looks* valid: executable and grounded (it uses
ids from its own tool results), just aimed at the wrong thing. Because it believes it owns everything, it
also contradicts other agents' claims when asked to review them.

    python -m oncall eval-team ... --careless data-fast      # does cross-check + optimizer contain it?
"""
from __future__ import annotations

from controller.proposal import InvestigationTask, Proposal

from ..tools import Toolbox
from .rules import RuleReasoner, best_guess

CONFIDENCE = 0.95


class CarelessReasoner(RuleReasoner):
    name = "careless"

    def investigate(self, task: InvestigationTask, agent_id: str, profile, toolbox: Toolbox) -> Proposal:
        honest = super().investigate(task.model_copy(update={"must_propose": False}), agent_id, profile, toolbox)
        if honest.verdict == "owner":
            return honest.model_copy(update={"confidence": CONFIDENCE, "reasoner": self.name})
        fix = best_guess(task.domain, toolbox)
        if fix is None:
            return honest
        fix = fix.model_copy(update={"rationale": "looks like the cause"})
        return Proposal(incident_id=task.incident.incident_id, agent_id=agent_id, domain=task.domain,
                        verdict="owner", root_cause=f"something in {task.domain} is slowing the platform",
                        confidence=CONFIDENCE, findings=honest.findings, actions=[fix], reasoner=self.name)
