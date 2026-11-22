"""Calibrate the router and compare team strategies (the C1 experiment). Lab code: reads labels.

Data discipline:
  development (default): calibrate on one half of `val`, report on the other half;
  final (--final):       calibrate on all of `val`, report on `test`. Do this once, at the end.
The classifier was trained on `train`, so neither half has been seen in training.

Strategies compared on the report split:
  top-1             only the most likely domain (cheapest; misses the owner whenever it is wrong)
  fixed             choose_team's hand-picked thresholds (0.8 confidence, 90% mass)
  lac a / mondrian a  conformal teams over a sweep of alpha values (one cut-off / one per domain)
  everyone          all four domains (always covers the owner; most expensive)

Compare at *equal coverage*: `cheapest_reaching` picks, for each coverage target, the cheapest
strategy that reaches it overall and in its worst domain.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from controller import DOMAINS
from controller.classifier import RuleClassifier
from controller.conformal import ConformalCalibrator
from controller.contracts import IncidentReport
from controller.intake import extract
from controller.pool import default_pool
from controller.registry import AgentRegistry
from controller.router import Router, RoutingPolicy

from .evaluate import load_dataset

ALPHAS = (0.01, 0.02, 0.03, 0.05, 0.10, 0.20)
METHODS = ("lac", "mondrian")
TARGETS = (0.90, 0.95, 0.99)


def _half(incident_id: str) -> int:
    return int(hashlib.sha1(incident_id.encode()).hexdigest(), 16) % 2


def split_for_calibration(pairs: list[tuple[IncidentReport, dict]], final: bool):
    if final:
        return ([p for p in pairs if p[1].get("split") == "val"],
                [p for p in pairs if p[1].get("split") == "test"], "val -> test (final)")
    val = [p for p in pairs if p[1].get("split") == "val"]
    return ([p for p in val if _half(p[0].incident_id) == 0],
            [p for p in val if _half(p[0].incident_id) == 1], "val half A -> val half B (development)")


def _registry() -> AgentRegistry:
    reg = AgentRegistry(clock=lambda: 0.0)
    for card in default_pool():
        reg.register(card, now=0.0)
    return reg


def measure(router: Router, pairs) -> dict:
    n = len(pairs)
    covered = primary_right = size = cost = notify = 0
    by_domain = {d: [0, 0] for d in DOMAINS}  # owner domain -> [covered, total]
    for report, truth in pairs:
        decision = router.route(report, now=0.0)
        hit = truth["owner"] in decision.team
        covered += hit
        by_domain[truth["owner"]][0] += hit
        by_domain[truth["owner"]][1] += 1
        primary_right += bool(decision.primary and decision.primary.domain == truth["owner"])
        size += len(decision.team)
        cost += decision.expected_cost
        notify += decision.notify_human
    per_domain = {d: (c / t if t else None) for d, (c, t) in by_domain.items()}
    present = [v for v in per_domain.values() if v is not None]
    return {"n": n, "owner_on_team": covered / n, "per_domain": per_domain, "worst_domain": min(present),
            "primary_right": primary_right / n, "mean_team_size": size / n, "mean_cost": cost / n,
            "human_notified": notify / n}


def cheapest_reaching(rows: dict[str, dict], targets=TARGETS) -> list[dict]:
    """For each target: the cheapest strategy whose overall AND worst-domain coverage reach it."""
    out = []
    for target in targets:
        ok = [(r["mean_cost"], name) for name, r in rows.items()
              if r["owner_on_team"] >= target and r["worst_domain"] >= target]
        cost, name = min(ok) if ok else (None, None)
        out.append({"target": target, "strategy": name, "mean_cost": cost})
    return out


def calibrate_and_compare(data_dir: Path, model_path: Path | None, out: Path, alpha: float = 0.05,
                          method: str = "mondrian", alphas=ALPHAS, final: bool = False) -> dict:
    classifier = RuleClassifier()
    if model_path:
        from controller.learned import LogRegClassifier

        classifier = LogRegClassifier.load(model_path)
    pairs = load_dataset(data_dir)
    cal, report, label = split_for_calibration(pairs, final)
    if not cal or not report:
        raise ValueError(f"empty calibration or report split in {data_dir}")
    cal_probs = [classifier.predict(extract(r)).owner_probs for r, _ in cal]
    cal_owners = [t["owner"] for _, t in cal]

    registry = _registry()
    uncapped = RoutingPolicy(max_team=len(DOMAINS))  # judge the team rule itself, without the size cap
    rows = {"top-1": measure(Router(classifier, registry, policy=RoutingPolicy(team_rule="top1")), report),
            "fixed (0.8 / 90%)": measure(Router(classifier, registry), report)}
    saved = None
    for m in METHODS:
        for a in sorted(set(alphas) | ({alpha} if m == method else set())):
            cal_m = ConformalCalibrator.fit(cal_probs, cal_owners, a, classifier.name, method=m)
            rows[f"{m} a={a:.2f}"] = measure(Router(classifier, registry, cal_m, uncapped), report)
            if m == method and a == alpha:
                saved = cal_m
    rows["everyone"] = measure(Router(classifier, registry, policy=RoutingPolicy(team_rule="all")), report)

    saved.save(out)
    return {"split": label, "calibration_n": len(cal), "calibration_by_domain": saved.n_by_domain,
            "report_n": len(report), "classifier": classifier.name, "saved": str(out), "method": method,
            "alpha": alpha, "saved_description": saved.describe(), "rows": rows,
            "cheapest": cheapest_reaching(rows)}
