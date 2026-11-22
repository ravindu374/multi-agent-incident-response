"""Score the controller's domain classification on a labelled incident set (synthetic or sandbox).

This is lab code: it reads ground truth. The controller itself never does.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from controller import DOMAINS
from controller.classifier import RuleClassifier, choose_team
from controller.contracts import IncidentReport
from controller.intake import extract


def load_dataset(data_dir: Path) -> list[tuple[IncidentReport, dict]]:
    data_dir = Path(data_dir)
    truth = {}
    for line in (data_dir / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            truth[record["incident_id"]] = record
    pairs = []
    for line in (data_dir / "incidents.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record["incident_id"] in truth:
                pairs.append((IncidentReport.from_lab_record(record), truth[record["incident_id"]]))
    return pairs


def evaluate_classifier(data_dir: Path, split: str | None = None, classifier=None) -> dict:
    classifier = classifier or RuleClassifier()
    rows = []
    for report, truth in load_dataset(data_dir):
        if split and truth.get("split") != split:
            continue
        team = choose_team(classifier.predict(extract(report)))
        rows.append({"fault": truth["fault_id"], "owner": truth["owner"], "predicted": team.owner,
                     "covered": truth["owner"] in team.team, "team_size": len(team.team), "confident": team.confident})
    if not rows:
        raise ValueError(f"no incidents for split {split!r} in {data_dir}")

    def summarise(subset: list[dict]) -> dict:
        n = len(subset)
        confident = [r for r in subset if r["confident"]]
        return {
            "n": n,
            "owner_accuracy": sum(r["predicted"] == r["owner"] for r in subset) / n,
            "owner_on_team": sum(r["covered"] for r in subset) / n,
            "mean_team_size": sum(r["team_size"] for r in subset) / n,
            "confident_share": len(confident) / n,
            "accuracy_when_confident": (sum(r["predicted"] == r["owner"] for r in confident) / len(confident)
                                        if confident else None),
        }

    per_fault = defaultdict(list)
    for r in rows:
        per_fault[r["fault"]].append(r)
    confusion = {owner: Counter() for owner in DOMAINS}
    for r in rows:
        confusion[r["owner"]][r["predicted"]] += 1
    return {
        "method": classifier.name,
        "split": split or "all",
        "overall": summarise(rows),
        "per_fault": {f: summarise(v) for f, v in sorted(per_fault.items())},
        "confusion": {owner: {p: confusion[owner][p] for p in DOMAINS} for owner in DOMAINS},
    }
