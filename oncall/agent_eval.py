"""Score every expert agent against ground truth (lab code: reads labels).

For each incident, every specialist examines its own domain and the generalist examines everything.
  solved       owner agents: verdict "owner" in the right domain with at least one CORRECT fix:
               a valid intent, executable, grounded in what the agent saw, and aimed at the culprit
               the fault really created (ground truth `fix_targets`: e.g. the session holding the lock,
               not the ingest session waiting for it)
  wrong target owner agents: the right kind of fix (valid intent) but none of them correct; the old
               "intent only" score would have counted these as solved
  false claim  non-owner agents: verdict "owner" for a fault that is not theirs
  unsure       verdict "unsure" (missing evidence, or the reasoner failed)
These rates are what Block 4's SuccessModel guesses today and Block 6 will learn from outcomes.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from agents.reasoners import ReasonerFailed
from agents.reasoners.ollama import OllamaReasoner
from agents.reasoners.rules import RuleReasoner
from agents.specialist import SpecialistAgent
from agents.tools import EvidenceUnavailable, RecordedEvidence
from controller import DOMAINS
from controller.checks import action_problems, evidence_text, identifiers
from controller.contracts import IncidentReport
from controller.plan import PlannedAction
from controller.pool import default_pool
from controller.proposal import InvestigationTask, Proposal

from .evaluate import load_dataset


def hits_fix_target(action: PlannedAction, fix_targets: dict | None) -> bool:
    """Does the action act on the culprit? True when the truth names no target for this intent."""
    alternatives = (fix_targets or {}).get(action.intent)
    if not alternatives:
        return True
    ids = identifiers(action)
    return any(all(ids.get(key) and set(ids[key]) <= {str(v) for v in allowed} for key, allowed in alt.items())
               for alt in alternatives)


def what_agent_saw(p: Proposal, report: IncidentReport, evidence: RecordedEvidence) -> str:
    seen = {}
    for tool in p.tools_used:
        try:
            seen[tool] = evidence.call(report.incident_id, tool)
        except (EvidenceUnavailable, KeyError):
            pass
    return evidence_text(seen, report)


def correct_fix(a: PlannedAction, truth: dict, seen: str) -> bool:
    return a.intent in truth["valid_intents"] and not action_problems(a, seen) and hits_fix_target(
        a, truth.get("fix_targets"))


def describe_mistake(p: Proposal, truth: dict, kind: str) -> str:
    fixes = "; ".join(f"{a.intent} on {a.target} {a.params}" for a in p.actions) or "none"
    notes = [f.fact for f in p.findings if f.tool == "action-check"]
    return (f"{p.agent_id:<16} {truth['incident_id']} {truth['fault_id']} ({kind})\n"
            f"    said:   {p.verdict} in {p.domain} ({p.confidence:.2f}): {p.root_cause or '-'}\n"
            f"    fixes:  {fixes}\n"
            + "".join(f"    check:  {x}\n" for x in notes)
            + f"    truth:  owner {truth['owner']}, fix {truth['valid_intents']} acting on "
              f"{truth.get('fix_targets') or 'anything'}")


def evaluate_agents(data_dir: Path, split: str | None = "val", reasoner: str = "rules", model: str | None = None,
                    agent_ids: list[str] | None = None, limit: int | None = None, show: bool = False) -> dict:
    pairs = [(r, t) for r, t in load_dataset(data_dir) if split is None or t.get("split") == split][:limit]
    evidence = RecordedEvidence.from_incidents_file(Path(data_dir) / "incidents.jsonl")
    brain = OllamaReasoner(model=model) if reasoner == "ollama" else RuleReasoner()
    agents = [SpecialistAgent(c, brain, evidence) for c in default_pool()
              if c.routing_profile().role != "policy" and (not agent_ids or c.name in agent_ids)]

    stats = defaultdict(lambda: {"own": 0, "solved": 0, "wrong_target": 0, "other": 0, "false_claims": 0,
                                 "unsure": 0, "n": 0, "tools": 0, "tokens": 0, "seconds": 0.0})
    per_fault = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # agent -> fault -> [solved, own]
    mistakes: list[str] = []
    for report, truth in pairs:
        for agent in agents:
            role = agent.profile.role
            domain = truth["owner"] if role == "generalist" else agent.profile.domains[0]
            try:
                p = agent.handle(InvestigationTask(incident=report, domain=domain))
            except ReasonerFailed as e:  # scored as "unsure" here; in a team the controller asks a fallback
                p = Proposal(incident_id=report.incident_id, agent_id=agent.agent_id, domain=domain,
                             verdict="unsure", confidence=0.0, root_cause=f"reasoner failed: {e}")
            s = stats[agent.agent_id]
            s["n"] += 1
            s["tools"] += len(p.tools_used)
            s["tokens"] += (p.usage.get("prompt_tokens") or 0) + (p.usage.get("completion_tokens") or 0)
            s["seconds"] += p.usage.get("seconds") or 0.0
            s["unsure"] += p.verdict == "unsure"
            if role == "generalist" or domain == truth["owner"]:
                claimed = p.verdict == "owner" and p.domain == truth["owner"]
                right_kind = claimed and any(a.intent in truth["valid_intents"] for a in p.actions)
                seen = what_agent_saw(p, report, evidence)
                ok = right_kind and any(correct_fix(a, truth, seen) for a in p.actions)
                s["own"] += 1
                s["solved"] += ok
                s["wrong_target"] += right_kind and not ok
                if show and not ok:
                    mistakes.append(describe_mistake(p, truth, "wrong target" if right_kind else "not solved"))
                per_fault[agent.agent_id][truth["fault_id"]][0] += ok
                per_fault[agent.agent_id][truth["fault_id"]][1] += 1
            else:
                s["other"] += 1
                s["false_claims"] += p.verdict == "owner"
                if show and p.verdict == "owner":
                    mistakes.append(describe_mistake(p, truth, "false claim"))

    rows = {}
    for agent_id, s in stats.items():
        rows[agent_id] = {
            "own_incidents": s["own"], "solved": s["solved"] / s["own"] if s["own"] else None,
            "wrong_target": s["wrong_target"] / s["own"] if s["own"] else None,
            "false_claims": s["false_claims"] / s["other"] if s["other"] else None,
            "unsure": s["unsure"] / s["n"], "mean_tools": s["tools"] / s["n"],
            "mean_tokens": s["tokens"] / s["n"] if s["tokens"] else None,
            "mean_seconds": s["seconds"] / s["n"] if s["seconds"] else None,
            "per_fault": {f: v[0] / v[1] for f, v in sorted(per_fault[agent_id].items())},
        }
    order = [f"{d}-{t}" for d in DOMAINS for t in ("fast", "strong")] + ["generalist"]
    return {"reasoner": brain.name if reasoner == "rules" else f"ollama:{model or 'per-card model'}",
            "split": split or "all", "incidents": len(pairs),
            "targets_checked": sum(bool(t.get("fix_targets")) for _, t in pairs), "mistakes": mistakes,
            "rows": {a: rows[a] for a in order if a in rows}}
