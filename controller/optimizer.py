"""Block 5c - Optimizer: from all the agents' suggestions to ONE plan with minimum expected loss.

1. Hypotheses. Every proposed fix is a hypothesis "this is the root cause, this is the fix": ownership
   claims, and (by default, must_propose) the best guesses every chosen agent must hand in even when its
   evidence is weak. Fixes that act on the same thing (same intent and the same pid / ip / target ...)
   are merged: two agents agreeing is one hypothesis with more evidence, not two.
2. Belief, in two steps:
     P(domain)        which domain owns it: the classifier's probabilities, moved by what agents found
                      (log-likelihood ratios: a claim pushes its domain up, more for stronger agents; a
                      domain's own agent saying "not mine" pushes it down; other domains' reviews push
                      up or down). Normalised over the domains: exactly one owns the incident.
     P(fix | domain)  which of that domain's fixes is right: how often this KIND of agent is right
                      (tier precision, the same planning numbers the router uses) times how often a fix
                      is right when its agent judged its own evidence that way (owner 1, unsure 0.6,
                      not_mine 0.3: a best guess is worth less than a claim), agreeing agents combined,
                      nudged by reviews from the same domain. Fixes of one domain share it.
     P(hypothesis) = P(domain) x P(fix | domain).
   So a weak guess in the domain the classifier is sure of can still beat nothing, while a strong claim
   elsewhere beats it: the classifier and the agents correct each other.
   Agents' self-reported confidence is NOT used: LLMs are confidently wrong (llama3 said 0.9 on wrong
   answers), and an over-eager agent would win by shouting. Block 6 learns from verified outcomes whether
   (and for which agent) confidence carries information, and replaces the tier precisions.
3. Plans. A plan runs hypotheses' fixes in stages: stage 1, check health, stage 2 only if still broken.
       expected loss = sum over stages of P(still broken before the stage) x (risk + stage cost)
                     + P(still broken at the end) x cost of then calling a human
   risk: low 0.5, medium 2, high 5, doubled if the action cannot be undone. Calling a human costs
   HUMAN_BASE + HUMAN_DELAY x value(priority): the more urgent, the more waiting costs.
   Stages are ordered by p / cost (fix chance per unit of risk; optimal for one root cause), and every
   prefix of that order is a candidate plan, next to "escalate to a human now". The cheapest wins.
   A plan runs automatically only if the policy agent would let it (controller/policy.py
   approval_reasons: no high-risk action, nothing irreversible while unsure, ...). When the policy holds
   a plan back, a human can approve it instead (plan kind "approve"): the human checks each stage
   before it runs, so a wrong fix is rejected and never runs, and a right one runs with the human
   watching. Assumed, as for escalation, that the human can tell a right fix from a wrong one:
       expected loss = approval cost (APPROVAL_BASE per fix to check + APPROVAL_DELAY x value(priority))
                     + sum over stages of P(this is the cause) x (APPROVED_RISK x risk + stage cost)
                     + P(still broken at the end) x cost of the human then taking over
   A human checking a ready fix is cheaper than a human diagnosing from zero (1 + 0.15 v vs 4 + 0.3 v),
   so a confident high-risk fix (a field mapping) goes to approval instead of to a human from scratch.

The optimizer never invents actions: it only orders and combines what agents proposed (and what passed
the agents' executable + grounded checks). If the best plan is unsure, the controller first asks for a
second look (controller/pipeline.py) and then optimizes again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import DOMAINS
from .checks import CHECK, same_fix
from .orchestrator import Deliberation
from .plan import INTENTS, PlannedAction, PlanOption, RemediationPlan
from .policy import approval_reasons
from .proposal import Proposal
from .router import PRIORITY_VALUE, SuccessModel

RISK_COST = {"low": 0.5, "medium": 2.0, "high": 5.0}


@dataclass
class OptimizerPolicy:
    # P(an agent's fix is right | it claims in the right domain), per tier (planning numbers; Block 6 learns)
    precision: dict[str, float] = field(default_factory=lambda: dict(SuccessModel.PRIOR))
    # ...scaled by how the agent judged its own evidence (a best guess handed in because it had to is right
    # less often than a claim). Planning numbers too; Block 6 measures them from verified outcomes.
    verdict_factor: dict[str, float] = field(default_factory=lambda: {"owner": 1.0, "unsure": 0.6,
                                                                      "not_mine": 0.3})
    # evidence about WHICH DOMAIN owns it, as log-likelihood ratios
    claim_llr: dict[str, float] = field(default_factory=lambda: {"fast": 1.0, "strong": 1.8, "generalist": 1.4})
    not_mine_llr: dict[str, float] = field(default_factory=lambda: {"fast": -0.3, "strong": -1.0,
                                                                    "generalist": -0.7})
    domain_review_llr: dict[str, float] = field(default_factory=lambda: {"supports": 0.7, "contradicts": -0.9,
                                                                         "cannot_tell": 0.0})
    # evidence about WHICH FIX within the domain (reviews by agents of the same domain), log-odds
    fix_review_llr: dict[str, float] = field(default_factory=lambda: {"supports": 1.0, "contradicts": -1.4,
                                                                      "cannot_tell": 0.0})
    fix_success: float = 0.9  # P(the fix works | its root cause is the real one)
    irreversible_factor: float = 2.0
    stage_cost: float = 0.5  # one more stage = more time until resolution
    human_base: float = 4.0
    human_delay: float = 0.3
    # "a human approves the fix": offered next to automatic plans and escalation (False = earlier design:
    # every plan priced as if it ran alone, and the policy agent's "needs a human" not priced at all)
    approval: bool = True
    approval_base: float = 1.0  # per fix the human must check (vs human_base to diagnose from zero)
    approval_delay: float = 0.15  # waiting for the yes: a ready fix is answered faster than an incident
    approved_risk: float = 0.2  # share of a fix's risk left when a human checked it and watches it run
    max_stages: int = 3
    priority_value: dict[str, float] = field(default_factory=lambda: dict(PRIORITY_VALUE))


@dataclass
class Hypothesis:
    domain: str
    claims: list[Proposal]
    actions: list[PlannedAction]
    belief: float = 0.0  # P(this is the root cause) = P(domain) x P(fix | domain)
    p_domain: float = 0.0
    p_fix: float = 0.0
    why: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.domain}: {self.claims[0].root_cause}"

    @property
    def agents(self) -> list[str]:
        return [c.agent_id for c in self.claims]


def _logit(p: float) -> float:
    p = min(0.98, max(0.02, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Optimizer:
    def __init__(self, tiers: dict[str, str], policy: OptimizerPolicy | None = None):
        """tiers: agent_id -> "fast" | "strong" | "generalist" (from the registry's cards)."""
        self.tiers, self.policy = tiers, policy or OptimizerPolicy()

    # -- 1 + 2: hypotheses and beliefs -------------------------------------------------------------
    def tier(self, agent_id: str) -> str:
        return self.tiers.get(agent_id, "fast")

    def domain_posterior(self, d: Deliberation) -> tuple[dict[str, float], dict[str, list[str]]]:
        pol = self.policy
        score = {x: math.log(max(1e-4, d.routing.owner_probs.get(x, 0.0))) for x in DOMAINS}
        why = {x: [f"classifier {d.routing.owner_probs.get(x, 0.0):.2f}"] for x in DOMAINS}
        for p in d.proposals:
            if p.domain not in score:
                continue
            t = self.tier(p.agent_id)
            if p.verdict == "owner" and p.actions:
                score[p.domain] += pol.claim_llr[t]
                why[p.domain].append(f"{p.agent_id} claims it")
            elif p.verdict == "not_mine":
                score[p.domain] += pol.not_mine_llr[t]
                why[p.domain].append(f"{p.agent_id} found nothing" + (" (best guess only)" if p.actions else ""))
            if p.actions:  # other domains' reviews of any proposed fix
                for r in d.reviews_of(p):
                    if r.domain != p.domain and r.stance != "cannot_tell":
                        score[p.domain] += pol.domain_review_llr[r.stance] * r.confidence
                        why[p.domain].append(f"{r.reviewer} ({r.domain}) {r.stance}")
        top = max(score.values())
        weights = {x: math.exp(v - top) for x, v in score.items()}
        total = sum(weights.values())
        return {x: w / total for x, w in weights.items()}, why

    def hypotheses(self, d: Deliberation) -> list[Hypothesis]:
        groups: list[Hypothesis] = []
        # claims first, so a merged hypothesis is labelled with a claim's root cause, not a guess's
        for claim in sorted(d.suggestions(), key=lambda p: p.verdict != "owner"):
            match = next((h for h in groups if h.domain == claim.domain and same_fix(h.actions, claim.actions)), None)
            if match:
                match.claims.append(claim)
            else:
                groups.append(Hypothesis(domain=claim.domain, claims=[claim], actions=list(claim.actions)))
        if not groups:
            return []
        pol = self.policy
        p_domain, why_domain = self.domain_posterior(d)
        raw: dict[int, float] = {}
        for i, h in enumerate(groups):
            wrong = 1.0  # noisy-or: the fix is wrong only if every agreeing agent is wrong
            for c in h.claims:
                q = _logit(pol.precision.get(self.tier(c.agent_id), 0.7) * pol.verdict_factor.get(c.verdict, 0.3))
                for r in d.reviews_of(c):
                    if r.domain == c.domain and r.reviewer not in h.agents and r.stance != "cannot_tell":
                        q += pol.fix_review_llr[r.stance] * r.confidence
                        h.why.append(f"{r.reviewer} {r.stance} the fix")
                wrong *= 1.0 - _sigmoid(q)
            raw[i] = 1.0 - wrong
            h.why.insert(0, ", ".join(f"{c.agent_id} ({self.tier(c.agent_id)}, "
                                      f"{'claim' if c.verdict == 'owner' else 'guess: ' + c.verdict})"
                                      for c in h.claims) + " propose it")
        for domain in {h.domain for h in groups}:  # fixes of one domain share that domain's probability
            members = [i for i, h in enumerate(groups) if h.domain == domain]
            mass = max(1.0, sum(raw[i] for i in members))
            for i in members:
                h = groups[i]
                h.p_domain, h.p_fix = p_domain[domain], raw[i] / mass
                h.belief = h.p_domain * h.p_fix
                h.why.insert(0, f"P({domain})={h.p_domain:.2f} [{'; '.join(why_domain[domain])}], "
                                f"P(fix|{domain})={h.p_fix:.2f}")
        return sorted(groups, key=lambda h: -h.belief)

    # -- 3: plans ----------------------------------------------------------------------------------
    def risk_cost(self, actions: list[PlannedAction]) -> float:
        return sum(RISK_COST[a.risk] * (1 if INTENTS[a.intent].reversible else self.policy.irreversible_factor)
                   for a in actions)

    def human_cost(self, priority: str) -> float:
        return self.policy.human_base + self.policy.human_delay * self.policy.priority_value[priority]

    def expected_loss(self, stages: list[Hypothesis], priority: str) -> tuple[float, float]:
        """(expected loss, P(fixed by the end)) of running these hypotheses' fixes in this order."""
        pol, broken, loss = self.policy, 1.0, 0.0
        for h in stages:
            loss += broken * (self.risk_cost(h.actions) + pol.stage_cost)
            broken -= h.belief * pol.fix_success  # one root cause: the hypotheses exclude each other
        broken = max(0.0, broken)
        return loss + broken * self.human_cost(priority), 1.0 - broken

    def approval_cost(self, priority: str, fixes: int) -> float:
        return self.policy.approval_base * fixes + self.policy.approval_delay * self.policy.priority_value[priority]

    def approval_loss(self, stages: list[Hypothesis], priority: str) -> tuple[float, float]:
        """(expected loss, P(fixed)) of asking a human to approve these stages: only the right fix runs."""
        pol = self.policy
        loss = self.approval_cost(priority, len(stages))
        loss += sum(h.belief * (pol.approved_risk * self.risk_cost(h.actions) + pol.stage_cost) for h in stages)
        fixed = min(1.0, sum(h.belief * pol.fix_success for h in stages))
        return loss + (1.0 - fixed) * self.human_cost(priority), fixed

    @staticmethod
    def staged(stages: list[Hypothesis]) -> list[PlannedAction]:
        return [a.model_copy(update={"stage": i}) for i, h in enumerate(stages, 1) for a in h.actions]

    def plan(self, d: Deliberation) -> RemediationPlan:
        hyps = self.hypotheses(d)
        pol, priority = self.policy, d.routing.priority
        ranked = sorted(hyps, key=lambda h: -(h.belief * pol.fix_success) / (self.risk_cost(h.actions) + pol.stage_cost))
        human = self.human_cost(priority)
        options = [PlanOption(label="escalate to a human now", p_fixed=0.0, expected_loss=round(human, 3))]
        plans: list[tuple[str, list[Hypothesis]]] = [("escalate", [])]  # what each option would hand over
        held_back: list[str] = []

        def offer(kind: str, stages: list[Hypothesis], loss: float, p_fixed: float) -> None:
            label = " then ".join(f"fix {h.domain} ({', '.join(h.agents)})" for h in stages)
            options.append(PlanOption(label=("a human approves: " if kind == "approve" else "") + label,
                                      p_fixed=round(p_fixed, 3), expected_loss=round(loss, 3),
                                      actions=[f"stage {a.stage}: {a.intent} on {a.target} {a.params}"
                                               for a in self.staged(stages)]))
            plans.append((kind, stages))

        k_max = min(len(ranked), pol.max_stages)
        for k in range(1, k_max + 1):
            stages = ranked[:k]
            blocked = approval_reasons(self.staged(stages), stages[0].belief, priority) if pol.approval else []
            if blocked:  # the policy agent would not let it run alone
                held_back += blocked
                continue
            offer("remediate", stages, *self.expected_loss(stages, priority))
        # The policy decides WHETHER a human must be involved; only then is approval an option. (Priced
        # alone, a human's check would also beat many safe automatic fixes, and nothing would run alone.)
        if pol.approval and held_back:
            likely = sorted(hyps, key=lambda h: -h.belief)  # the human checks the likeliest fix first
            for k in range(1, k_max + 1):
                offer("approve", likely[:k], *self.approval_loss(likely[:k], priority))
        best = min(range(len(options)), key=lambda i: (options[i].expected_loss, i))
        options[best].chosen = True
        kind, chosen = plans[best]

        evidence = [f"[{h.domain}] {w}" for h in hyps for w in h.why]
        if held_back:
            evidence.append(f"[policy] may not run without a human: {'; '.join(dict.fromkeys(held_back))}")
        evidence += [f"[{p.domain}] {f.fact}" for p in d.proposals for f in p.findings[:2] if f.tool != CHECK]
        lead = chosen[0] if chosen else (hyps[0] if hyps else None)
        owner = lead.domain if lead else (d.routing.team[0] if d.routing.team else
                                          max(d.routing.owner_probs, key=d.routing.owner_probs.get))
        team = [owner] + [x for x in dict.fromkeys(d.routing.team + [p.domain for p in d.proposals])
                          if x != owner and x in DOMAINS]
        common = dict(incident_id=d.incident_id, owner=owner, team=team, evidence=evidence[:12],
                      options=options, expected_loss=options[best].expected_loss)
        if kind == "escalate":
            reason = ("no agent found a root cause with a usable fix" if not hyps else
                      "acting is riskier than asking a human: " + "; ".join(
                          f"{h.label} (belief {h.belief:.2f})" for h in hyps[:2]))
            return RemediationPlan(kind="escalate", root_cause=reason,
                                   confidence=round(hyps[0].belief, 3) if hyps else 0.0, **common)
        return RemediationPlan(kind=kind, root_cause=chosen[0].label.split(": ", 1)[1],
                               confidence=round(chosen[0].belief, 3), actions=self.staged(chosen), **common)


def tiers_from(registry) -> dict[str, str]:
    return {a: ("generalist" if registry.profile(a).role == "generalist" else registry.profile(a).tier)
            for a in registry.agents()}
