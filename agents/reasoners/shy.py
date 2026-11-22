"""A deliberately under-confident reasoner, for robustness experiments (never for real use).

It reproduces what qwen2.5:3b and llama3 did in the first LLM runs: it finds the real cause and the
right fix, then answers "not mine" anyway. When agents may abstain, that fix is thrown away; when every
chosen agent must propose (the default), the fix is handed in with the "not_mine" verdict and the
optimizer can still use it, weighted down.

    python -m oncall eval-team ... --shy data-fast infra-fast              # must propose (default)
    python -m oncall eval-team ... --shy data-fast infra-fast --may-abstain  # the earlier design
"""
from __future__ import annotations

from controller.proposal import InvestigationTask, Proposal

from ..tools import Toolbox
from .rules import RuleReasoner


class ShyReasoner(RuleReasoner):
    name = "shy"

    def investigate(self, task: InvestigationTask, agent_id: str, profile, toolbox: Toolbox) -> Proposal:
        honest = super().investigate(task, agent_id, profile, toolbox)
        if honest.verdict != "owner":
            return honest.model_copy(update={"reasoner": self.name})
        return honest.model_copy(update={"verdict": "not_mine", "confidence": 0.8, "reasoner": self.name,
                                         "root_cause": f"{honest.root_cause} (but probably not my domain)"})
