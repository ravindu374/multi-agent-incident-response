"""Score the whole controller (routing + deliberation + optimizer + policy) against ground truth (lab code).

Agents run in-process (agents/local.py); over A2A the rounds are identical. Per incident:
  owner on team / owner asked      did the right domain get looked at?
  correct fix proposed             some agent handed in a correct fix (agent_eval.correct_fix), as a claim
                                   or as the best guess a chosen agent must give (must_propose)
  correct claim found              ...and claimed ownership for it
  wrong fixes (+ contradicted)     fixes in the wrong domain or not hitting the culprit, and how often a
                                   reviewer objected; right fixes contradicted = the price of reviewing
  plan                             remediate, approve or escalate; does some stage hold a correct fix; stage 1?
  realized loss                    run the plan stage by stage as the executor would (stop when fixed):
                                   risk + stage cost of every stage run, plus a human's cost if it never got
                                   fixed. An "approve" plan: the approval cost, then the human rejects wrong
                                   stages (they never run) and lets the first correct one run (a fifth of its
                                   risk, the human watching); no correct stage = the human takes over.
                                   Compared with: "top claim" (do what the most confident agent said),
                                   "always a human", and "oracle" (the cheapest way to the correct fix any
                                   agent found, under the same rules: alone if the policy allows it, else
                                   approved by a human; or a human from zero if that is cheaper).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from agents.local import LocalLink, local_agents
from agents.tools import RecordedEvidence
from controller.checks import evidence_text
from controller.classifier import RuleClassifier
from controller.conformal import ConformalCalibrator
from controller.orchestrator import OrchestratorPolicy, format_deliberation
from controller.pipeline import Controller, format_plan
from controller.optimizer import OptimizerPolicy
from controller.plan import PlannedAction
from controller.policy import approval_reasons
from controller.pool import default_pool
from controller.registry import AgentRegistry
from controller.router import Router

from .agent_eval import correct_fix
from .evaluate import load_dataset


def realized_loss(stages: list[list[PlannedAction]] | None, truth: dict, seen: str, optimizer, priority: str,
                  approve: bool = False) -> float:
    """Loss of actually running these stages (None = hand to a human right away). approve: a human checks
    each stage first, rejects the wrong ones and lets the first correct one run."""
    if not stages:
        return optimizer.human_cost(priority)
    pol = optimizer.policy
    loss = optimizer.approval_cost(priority, len(stages)) if approve else 0.0
    for stage in stages:
        right = any(correct_fix(a, truth, seen) for a in stage)
        if approve and not right:
            continue  # rejected: it never runs
        loss += (pol.approved_risk if approve else 1.0) * optimizer.risk_cost(stage) + pol.stage_cost
        if right:
            return loss
    return loss + optimizer.human_cost(priority)


def plan_loss(plan, truth: dict, seen: str, optimizer, priority: str) -> float:
    """What the plan really costs, given what the executor would do with it."""
    stages = plan.stages()
    if plan.kind == "approve":
        return realized_loss(stages, truth, seen, optimizer, priority, approve=True)
    runs = plan.kind == "remediate" and (plan.policy is None or plan.policy.approved or not optimizer.policy.approval)
    return realized_loss(stages if runs else None, truth, seen, optimizer, priority)


def oracle_loss(good: list[PlannedAction], truth: dict, seen: str, optimizer, priority: str) -> float:
    """The cheapest way to a correct fix an agent found, under the same rules as the controller."""
    options = [optimizer.human_cost(priority)]
    for a in good:  # knowing the truth, the plan is 100% sure: only a high-risk fix is held back
        held_back = optimizer.policy.approval and bool(approval_reasons([a], 1.0, priority))
        options.append(realized_loss([[a]], truth, seen, optimizer, priority, approve=held_back))
    return min(options)


def build_controller(data_dir: Path, model: Path | None = None, calibration: Path | None = None,
                     reasoner: str = "rules", ollama_model: str | None = None, careless: list[str] | None = None,
                     shy: list[str] | None = None, approval: bool = True, memory=None, trust: str = "local",
                     **orchestration):
    """The whole controller with in-process agents reading this dataset's evidence bundles.
    memory + trust: Block 6b (trust "off" = the memory is only written).
    orchestration: OrchestratorPolicy fields (cross_check, follow_pointers, escalate, ...)."""
    if model:
        from controller.learned import LogRegClassifier

        classifier = LogRegClassifier.load(model)
    else:
        classifier = RuleClassifier()
    calibrator = ConformalCalibrator.load(calibration) if calibration else None
    evidence = RecordedEvidence.from_incidents_file(Path(data_dir) / "incidents.jsonl")
    registry = AgentRegistry(clock=lambda: 0.0)  # in-process agents are always alive: no heartbeat expiry
    for card in default_pool():
        registry.register(card)
    agents = local_agents(evidence, reasoner, ollama_model)
    for agent_id in careless or []:  # robustness experiment: make these agents over-eager
        from agents.reasoners.careless import CarelessReasoner

        agents[agent_id].reasoner = CarelessReasoner()
    for agent_id in shy or []:  # robustness experiment: these agents find the fix, then say "not mine"
        from agents.reasoners.shy import ShyReasoner

        agents[agent_id].reasoner = ShyReasoner()
    controller = Controller(Router(classifier, registry, calibrator), LocalLink(agents),
                            OrchestratorPolicy(**orchestration), OptimizerPolicy(approval=approval), memory=memory,
                            trust=trust)
    return controller, evidence, classifier, calibrator


def evaluate_team(data_dir: Path, split: str | None = "val", model: Path | None = None,
                  calibration: Path | None = None, reasoner: str = "rules", ollama_model: str | None = None,
                  limit: int | None = None, cross_check: str = "always", follow_pointers: bool = True,
                  escalate: bool = True, careless: list[str] | None = None, second_look_below: float = 0.7,
                  must_propose: bool = True, shy: list[str] | None = None, approval: bool = True,
                  memory: Path | None = None, trust: str = "local", show: bool = False) -> dict:
    """memory: a saved experience memory (Block 6a) to trust agents by (Block 6b); it is only read here,
    never written, so the incidents scored cannot leak into it."""
    pairs = [(r, t) for r, t in load_dataset(data_dir) if split is None or t.get("split") == split][:limit]
    store = None
    if memory is not None:
        from controller.memory import ExperienceStore

        store = ExperienceStore.load(memory)
        overlap = {e.incident_id for e in store.experiences} & {r.incident_id for r, _ in pairs}
        if overlap:
            raise ValueError(f"{len(overlap)} scored incidents are in the memory (e.g. {min(overlap)}): "
                             "build it from another split")
    controller, evidence, classifier, calibrator = build_controller(
        data_dir, model, calibration, reasoner, ollama_model, careless, shy, approval, memory=store, trust=trust,
        cross_check=cross_check, follow_pointers=follow_pointers, escalate=escalate,
        second_look_below=second_look_below, must_propose=must_propose)
    optimizer = controller.optimizer

    keys = ["owner_on_team", "owner_asked", "correct_fix", "no_fix", "fixes", "correct_claim", "no_claim",
            "claims", "wrong_claims",
            "wrong_contradicted", "right_claims", "right_contradicted", "right_supported", "reviewed_claims",
            "transfers", "transfer_rescued", "escalations", "escalation_rescued", "calls", "failed_calls",
            "remediate", "approve", "approve_wasted", "plan_has_fix", "first_stage_right", "extra_actions",
            "escalated_but_fixable", "needs_human", "not_approved"]
    n = dict.fromkeys(keys, 0)
    loss = dict.fromkeys(["plan", "top_claim", "human", "oracle"], 0.0)
    cost, mistakes = 0.0, []
    for report, truth in pairs:
        res = asyncio.run(controller.handle(report))
        d, plan = res.deliberation, res.plan
        owner, priority = truth["owner"], d.routing.priority
        seen = evidence_text(evidence.bundles.get(report.incident_id), report)

        def correct(p) -> bool:
            return p.domain == owner and any(correct_fix(a, truth, seen) for a in p.actions)

        claims, fixes = d.claims(), d.suggestions()
        right = [c for c in fixes if correct(c)]  # right / wrong: every proposed fix (= claims when abstaining)
        wrong = [c for c in fixes if not correct(c)]
        on_team = owner in d.routing.team
        n["owner_on_team"] += on_team
        n["owner_asked"] += any(p.domain == owner for p in d.proposals)
        n["correct_fix"] += bool(right)
        n["no_fix"] += not fixes
        n["fixes"] += len(fixes)
        n["correct_claim"] += any(correct(c) for c in claims)
        n["no_claim"] += not claims
        n["claims"] += len(claims)
        n["wrong_claims"] += len(wrong)
        n["wrong_contradicted"] += sum(d.tally(c)["contradicts"] > 0 for c in wrong)
        n["right_claims"] += len(right)
        n["right_contradicted"] += sum(d.tally(c)["contradicts"] > 0 for c in right)
        n["right_supported"] += sum(d.tally(c)["supports"] > 0 for c in right)
        n["reviewed_claims"] += sum(bool(d.reviews_of(c)) for c in fixes)
        n["transfers"] += len(d.transfers)
        n["transfer_rescued"] += bool(d.transfers) and not on_team and bool(right)
        n["escalations"] += len(d.escalations)
        n["escalation_rescued"] += bool(d.escalations) and bool(right)
        n["calls"] += len(d.calls)
        n["failed_calls"] += sum(not c.ok for c in d.calls)
        cost += d.cost_units

        # the plan
        stages = plan.stages()
        fixed_at = next((i for i, st in enumerate(stages, 1) if any(correct_fix(a, truth, seen) for a in st)), None)
        n["remediate"] += plan.kind == "remediate"
        n["approve"] += plan.kind == "approve"
        n["approve_wasted"] += plan.kind == "approve" and fixed_at is None
        n["plan_has_fix"] += fixed_at is not None
        n["first_stage_right"] += fixed_at == 1
        if plan.kind == "remediate":  # under approval, wrong stages are rejected and never run
            n["extra_actions"] += sum(len(st) for st in stages[: (fixed_at or len(stages) + 1) - 1])
        n["escalated_but_fixable"] += plan.kind == "escalate" and bool(right)
        n["needs_human"] += bool(plan.policy and plan.policy.needs_human)
        n["not_approved"] += plan.kind != "escalate" and not (plan.policy and plan.policy.approved)

        loss["plan"] += plan_loss(plan, truth, seen, optimizer, priority)
        top = max(fixes, key=lambda c: c.confidence, default=None)  # the most confident agent's fix, run as is
        loss["top_claim"] += realized_loss([top.actions] if top else None, truth, seen, optimizer, priority)
        loss["human"] += optimizer.human_cost(priority)
        good = [a for c in right for a in c.actions if correct_fix(a, truth, seen)]
        loss["oracle"] += oracle_loss(good, truth, seen, optimizer, priority)

        # show what went wrong in the plan, or a false ownership claim (wrong best guesses are expected)
        if show and (fixed_at != 1 or any(not correct(c) for c in claims)):
            mistakes.append(f"{format_deliberation(d)}\n{format_plan(plan)}\n  TRUTH: {truth['fault_id']} "
                            f"owner={owner} fix={truth['valid_intents']} targets={truth.get('fix_targets') or '-'}")

    total = len(pairs) or 1

    def rate(key: str, of: int) -> float | None:
        return n[key] / of if of else None

    remediate = n["remediate"]
    return {
        "incidents": len(pairs), "split": split or "all", "reasoner": reasoner if reasoner == "rules"
        else f"ollama:{ollama_model or 'per-card model'}", "classifier": classifier.name,
        "calibrated": calibrator is not None, "cross_check": cross_check, "follow_pointers": follow_pointers,
        "escalate": escalate, "careless": careless or [], "shy": shy or [], "must_propose": must_propose,
        "approval": approval, "plan_approve": n["approve"] / total, "approve_wasted": n["approve_wasted"],
        "memory": None if store is None else {"path": str(memory), "experiences": len(store), "trust": trust},
        "owner_on_team": n["owner_on_team"] / total, "owner_asked": n["owner_asked"] / total,
        "correct_fix": n["correct_fix"] / total, "no_fix": n["no_fix"] / total,
        "fixes_per_incident": n["fixes"] / total,
        "correct_claim": n["correct_claim"] / total, "no_claim": n["no_claim"] / total,
        "claims_per_incident": n["claims"] / total, "wrong_claims": n["wrong_claims"],
        "wrong_contradicted": rate("wrong_contradicted", n["wrong_claims"]),
        "right_claims": n["right_claims"], "right_contradicted": rate("right_contradicted", n["right_claims"]),
        "right_supported": rate("right_supported", n["right_claims"]),
        "claims_reviewed": rate("reviewed_claims", n["fixes"]),
        "transfers": n["transfers"], "transfer_rescued": n["transfer_rescued"],
        "escalations": n["escalations"], "escalation_rescued": n["escalation_rescued"],
        "calls_per_incident": n["calls"] / total, "failed_calls": n["failed_calls"],
        "cost_per_incident": cost / total,
        "plan_remediate": remediate / total, "plan_has_fix": n["plan_has_fix"] / total,
        "first_stage_right": n["first_stage_right"] / total, "extra_actions": n["extra_actions"] / total,
        "escalated_but_fixable": n["escalated_but_fixable"], "needs_human": rate("needs_human", total),
        "not_approved": n["not_approved"],
        "loss": {k: v / total for k, v in loss.items()}, "mistakes": mistakes,
    }
