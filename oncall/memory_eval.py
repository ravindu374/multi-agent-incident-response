"""Block 6a in the lab: fill the experience memory from verified (simulated) outcomes, and check it.

build_memory        run the whole controller on one split, oldest incident first; after each incident, the
                    simulated verified feedback (oncall/feedback.py) -> controller.learn -> the store. Every
                    label is also checked against ground truth (only possible in the lab): a label the
                    feedback got wrong would teach Block 6b something false.
evaluate_retrieval  for each incident of another split, are the most similar past experiences the same kind
                    of incident? And leave-one-fault-out: with every past incident of the SAME fault hidden,
                    does the nearest one still have the right owner? Synthetic incidents of one fault are near
                    copies of each other, so the first numbers flatter the memory; the second is the honest
                    one for a kind of incident the memory has never seen. The fault catalogue gives every
                    fault a look-alike owned by another domain, so a low leave-one-fault-out score is
                    expected, and it says what the memory must NOT be trusted for.
"""
from __future__ import annotations

import asyncio
import random
from collections import Counter, defaultdict
from pathlib import Path

from controller.checks import evidence_text
from controller.intake import extract
from controller.memory import ExperienceStore

from .evaluate import load_dataset
from .feedback import is_right, simulated_feedback
from .team_eval import build_controller, plan_loss


def _in_time_order(pairs):
    return sorted(pairs, key=lambda rt: (rt[0].opened_at is None, rt[0].opened_at or "", rt[0].incident_id))


def build_memory(data_dir: Path, split: str = "train", model: Path | None = None, calibration: Path | None = None,
                 reasoner: str = "rules", ollama_model: str | None = None, mode: str = "bandit", noise: float = 0.0,
                 limit: int | None = None, seed: int = 0, approval: bool = True,
                 careless: list[str] | None = None, shy: list[str] | None = None, trust: str = "off",
                 windows: int = 4, store: ExperienceStore | None = None) -> tuple[ExperienceStore, dict]:
    """trust "off": only remember (Block 6a). "local" / "global": learn as you go (Block 6b): every incident
    is decided with the memory of the incidents before it, then remembered; `curve` shows loss and agent cost
    over `windows` equal parts of the run, so a falling curve is learning."""
    store = store if store is not None else ExperienceStore()
    pairs = _in_time_order([(r, t) for r, t in load_dataset(data_dir) if t.get("split") == split])[:limit]
    controller, evidence, classifier, _ = build_controller(data_dir, model, calibration, reasoner, ollama_model,
                                                           careless, shy, approval, memory=store, trust=trust)
    rng = random.Random(seed)
    labels = {lab: Counter() for lab in ("right", "wrong", "unknown")}  # label -> {truly right: n, ...}
    owners, sources, plans = Counter(), Counter(), Counter()
    per_incident: list[tuple[float, float]] = []  # (realized loss, agent cost units)
    for report, truth in pairs:
        resolution = asyncio.run(controller.handle(report))
        seen = evidence_text(evidence.bundles.get(report.incident_id), report)
        per_incident.append((plan_loss(resolution.plan, truth, seen, controller.optimizer,
                                       resolution.deliberation.routing.priority),
                             resolution.deliberation.cost_units))
        feedback = simulated_feedback(resolution.deliberation, resolution.plan, truth, seen, mode=mode, noise=noise,
                                      rng=rng, approval=approval)
        experience = controller.learn(report, resolution, feedback)
        for p in experience.proposals:
            if p.label is not None:
                labels[p.label]["truly right" if is_right(p, truth, seen) else "truly wrong"] += 1
        owners["unknown" if experience.owner is None else
               ("right" if experience.owner == truth["owner"] else "wrong")] += 1
        sources[feedback.source] += 1
        plans[resolution.plan.kind] += 1
    curve = []
    size = max(1, -(-len(per_incident) // max(1, windows)))  # ceiling division
    for start in range(0, len(per_incident), size):
        part = per_incident[start:start + size]
        curve.append({"from": start + 1, "to": start + len(part), "loss": sum(x for x, _ in part) / len(part),
                      "cost": sum(c for _, c in part) / len(part)})
    return store, {"incidents": len(pairs), "split": split, "mode": mode, "noise": noise, "trust": trust,
                   "careless": careless or [], "shy": shy or [],
                   "reasoner": reasoner, "classifier": classifier.name, "sources": dict(sources),
                   "plans": dict(plans), "labels": {k: dict(v) for k, v in labels.items()}, "owners": dict(owners),
                   "curve": curve, "summary": store.summary()}


def evaluate_retrieval(store: ExperienceStore, data_dir: Path, split: str = "val", k: int = 5,
                       limit: int | None = None, model: Path | None = None) -> dict:
    truth_of = {r.incident_id: t for r, t in load_dataset(data_dir)}
    missing = [e.incident_id for e in store.experiences if e.incident_id not in truth_of]
    if missing:
        raise ValueError(f"{len(missing)} experiences are not in {data_dir} (e.g. {missing[0]}): use the dataset "
                         "the memory was built from")
    fault_ids = defaultdict(set)
    for e in store.experiences:
        fault_ids[truth_of[e.incident_id]["fault_id"]].add(e.incident_id)
    classifier = None
    if model:
        from controller.learned import LogRegClassifier

        classifier = LogRegClassifier.load(model)
    queries = [(r, t) for r, t in load_dataset(data_dir) if t.get("split") == split][:limit]
    n = Counter()
    sims, lofo_sims = [], []
    per_fault = defaultdict(Counter)
    lofo_lookalike = defaultdict(Counter)

    def vote(neighbours) -> str | None:
        weight = Counter()
        for score, e in neighbours:
            if e.owner:
                weight[e.owner] += score
        return weight.most_common(1)[0][0] if weight else None

    for report, truth in queries:
        fault, owner = truth["fault_id"], truth["owner"]
        f = extract(report)
        near = store.similar(f, k=k)
        lofo = store.similar(f, k=k, exclude=fault_ids.get(fault, set()))
        if near:
            top = near[0][1]
            sims.append(near[0][0])
            n["same_fault"] += truth_of[top.incident_id]["fault_id"] == fault
            n["owner_top1"] += top.owner == owner
            n["owner_vote"] += vote(near) == owner
            per_fault[fault]["same_fault"] += truth_of[top.incident_id]["fault_id"] == fault
        if lofo:
            top = lofo[0][1]
            lofo_sims.append(lofo[0][0])
            n["lofo_owner_top1"] += top.owner == owner
            n["lofo_owner_vote"] += vote(lofo) == owner
            per_fault[fault]["lofo_owner"] += top.owner == owner
            lofo_lookalike[fault][truth_of[top.incident_id]["fault_id"]] += 1
        if classifier is not None:
            n["classifier"] += classifier.predict(f).ranked()[0][0] == owner
        per_fault[fault]["n"] += 1
    total = len(queries) or 1
    return {
        "queries": len(queries), "split": split, "k": k, "memory": len(store),
        "same_fault_top1": n["same_fault"] / total, "owner_top1": n["owner_top1"] / total,
        "owner_vote": n["owner_vote"] / total, "lofo_owner_top1": n["lofo_owner_top1"] / total,
        "lofo_owner_vote": n["lofo_owner_vote"] / total,
        "classifier": n["classifier"] / total if classifier is not None else None,
        "mean_top1_similarity": sum(sims) / len(sims) if sims else None,
        "mean_lofo_top1_similarity": sum(lofo_sims) / len(lofo_sims) if lofo_sims else None,
        "per_fault": {f: {"n": c["n"], "same_fault": c["same_fault"] / c["n"], "lofo_owner": c["lofo_owner"] / c["n"],
                          "lofo_lookalike": lofo_lookalike[f].most_common(1)[0][0] if lofo_lookalike[f] else None,
                          "owner": next(t["owner"] for _, t in queries if t["fault_id"] == f)}
                      for f, c in sorted(per_fault.items())},
    }
