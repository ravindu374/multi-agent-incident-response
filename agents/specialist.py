"""One expert agent: its Agent Card, a reasoner, and access to its own evidence tools.

handle(task) runs the investigation and returns a Proposal. Behind an A2A server (agents/server.py)
the task arrives as a message and the proposal leaves as an artifact.

Before a proposal leaves, every action is checked, whatever reasoner wrote it (rules or LLM):
allowed for this agent, executable (names what it acts on), grounded (its ids appear in what the
agent saw) and located (a process to stop is in the target's process list, a field to map is one the
source sends). Failing actions are
dropped with a note in the findings, and an ownership claim left without any action becomes "unsure":
"it's mine" without a usable fix is not actionable. An agent cannot point to its own domain.

A task with must_propose (the controller's default: the classifier chose this agent) asks for the
agent's best fix whatever its verdict; the verdict then says how strongly its evidence backs it.
Without must_propose, a "not mine" / "unsure" answer carries no fixes.
"""
from __future__ import annotations

from controller.cards import SERVICE_ROLES, AgentCard
from controller.checks import CHECK, action_problems, evidence_text, location_problems
from controller.proposal import Finding, InvestigationTask, Proposal, Review

from .reasoners import Reasoner
from .tools import EvidenceSource, Toolbox


class SpecialistAgent:
    def __init__(self, card: AgentCard, reasoner: Reasoner, evidence: EvidenceSource):
        self.card, self.profile = card, card.routing_profile()
        if self.profile.role in SERVICE_ROLES:
            raise ValueError(f"{card.name} is a {self.profile.role} agent; use PolicyAgent / HistoryAgent")
        self.reasoner, self.evidence = reasoner, evidence

    @property
    def agent_id(self) -> str:
        return self.card.name

    def handle(self, task: InvestigationTask) -> Proposal:
        incident_id = task.incident.incident_id
        if self.profile.role == "specialist" and task.domain not in self.profile.domains:
            return Proposal(incident_id=incident_id, agent_id=self.agent_id, domain=task.domain, verdict="unsure",
                            confidence=0.0, root_cause=f"{self.agent_id} does not cover the {task.domain} domain",
                            reasoner=self.reasoner.name)
        toolbox = Toolbox(self.evidence, self.profile.tools, incident_id)
        proposal = self.reasoner.investigate(task, self.agent_id, self.profile, toolbox)
        proposal.tools_used = list(toolbox.calls)
        if proposal.points_to in (proposal.domain, task.domain):
            proposal.points_to = None  # "not mine, look at my own domain" is no pointer
        if proposal.verdict != "owner" and proposal.actions and not task.must_propose:
            proposal.findings.append(Finding(tool=CHECK, fact=f"dropped {len(proposal.actions)} fix(es): a "
                                                              f"'{proposal.verdict}' answer cannot propose fixes"))
            proposal.actions = []
        proposed = bool(proposal.actions)
        self._check_actions(proposal, evidence_text(toolbox.results, task.incident), toolbox.results)
        if task.must_propose and not proposed:
            proposal.findings.append(Finding(tool=CHECK, fact="was asked for a fix but proposed none"))
        return proposal

    def review(self, task: InvestigationTask) -> Review:
        """Judge another agent's claim (task.claim) from this agent's own evidence in task.domain."""
        if task.claim is None:
            raise ValueError("a review task must carry the claim to review")
        if self.profile.role == "specialist" and task.domain not in self.profile.domains:
            return Review(incident_id=task.incident.incident_id, reviewer=self.agent_id, domain=task.domain,
                          claim_agent=task.claim.agent_id, claim_domain=task.claim.domain, stance="cannot_tell",
                          confidence=0.0, reason=f"{self.agent_id} does not cover the {task.domain} domain",
                          reasoner=self.reasoner.name)
        toolbox = Toolbox(self.evidence, self.profile.tools, task.incident.incident_id)
        review = self.reasoner.review(task, self.agent_id, self.profile, toolbox)
        review.tools_used = list(toolbox.calls)
        return review

    def _check_actions(self, proposal: Proposal, seen: str, results: dict) -> None:
        kept = []
        for a in proposal.actions:
            problems = (["not allowed for this agent"] if a.intent not in self.profile.allowed_intents
                        else action_problems(a, seen) + location_problems(a, results))
            if problems:
                proposal.findings.append(Finding(tool=CHECK, fact=f"dropped {a.intent} on {a.target!r} "
                                                                  f"{a.params}: {'; '.join(problems)}"))
            else:
                kept.append(a)
        proposal.actions = kept
        if proposal.verdict == "owner" and not kept:
            proposal.verdict = "unsure"
            proposal.findings.append(Finding(tool=CHECK, fact="claimed ownership without an executable, "
                                                              "evidence-backed fix"))
