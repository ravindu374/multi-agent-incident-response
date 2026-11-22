"""Generate a labelled synthetic incident set.

Output (same shapes as the lab writes during `validate`):
    incidents.jsonl     what the controller sees (alerts, logs, job runs, evidence bundle)
    ground_truth.jsonl  fault, owner, involved domains, valid fixes/intents, variant, noise, split
    manifest.json       seed, generator version, counts
Incident ids are neutral and assigned after shuffling, so neither ids nor file order reveal the fault.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from ..faults import FAULTS
from ..intents import intent_of
from ..settings import Settings
from .noise import draw_noise
from .scenarios import SCENARIOS
from .simulate import simulate
from .world import make_world

GENERATOR_VERSION = "2"  # 2: ground truth gains fix_targets (incidents unchanged)
SPLITS = (("train", 0.6), ("val", 0.2), ("test", 0.2))


def _split_for(rank: int, n: int) -> str:
    position, edge = rank / n, 0.0
    for name, share in SPLITS:
        edge += share
        if position < edge - 1e-9:
            return name
    return SPLITS[-1][0]


def generate(per_fault: int = 30, seed: int = 7, fault_ids: list[str] | None = None,
             settings: Settings | None = None, max_attempts_factor: int = 6) -> tuple[list[dict], list[dict], dict]:
    settings = settings or Settings()
    master = random.Random(seed)
    fault_ids = fault_ids or list(FAULTS)
    produced: list[tuple[dict, dict]] = []
    resampled: Counter = Counter()

    for fault_id in fault_ids:
        fault = FAULTS[fault_id]
        made = 0
        for _attempt in range(per_fault * max_attempts_factor):
            if made == per_fault:
                break
            rng = random.Random(master.getrandbits(64))
            world = make_world(rng)
            effects = SCENARIOS[fault_id](world)
            noise = draw_noise(world)
            result = simulate(world, effects, settings, noise)
            if result is None:
                resampled[fault_id] += 1
                continue
            made += 1
            known_pids = len(world._pids)
            fix_targets = effects.fix_targets(world) if effects.fix_targets else {}
            if len(world._pids) != known_pids:  # would change the random stream, i.e. the incidents
                raise RuntimeError(f"{fault_id}: fix_targets looked up a pid the evidence never created")
            truth = {
                **fault.ground_truth(),
                "valid_intents": sorted({intent_of(a) for a in fault.valid_fixes}),
                "variant": effects.variant,
                "noise": noise.labels(),
                "detect_after_s": round(result.t_open - effects.onset_s, 1),
                "fix_targets": fix_targets,  # what a correct fix must act on (see scenarios.FixTargetsFn)
            }
            produced.append((result.record, truth))
        if made < per_fault:
            raise RuntimeError(f"{fault_id}: only {made}/{per_fault} incidents were detected; check the scenario")

    # stratified split per fault, then a global shuffle and neutral ids
    by_fault: dict[str, list[int]] = {}
    for i, (_, truth) in enumerate(produced):
        by_fault.setdefault(truth["fault_id"], []).append(i)
    for indices in by_fault.values():
        master.shuffle(indices)
        for rank, i in enumerate(indices):
            produced[i][1]["split"] = _split_for(rank, len(indices))
    master.shuffle(produced)

    incidents, truths = [], []
    for n, (record, truth) in enumerate(produced, 1):
        incident_id = f"syn-{seed}-{n:05d}"
        incidents.append({**record, "incident_id": incident_id})
        truths.append({"incident_id": incident_id, "source": "synthetic", **truth})

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "per_fault": per_fault,
        "faults": fault_ids,
        "incidents": len(incidents),
        "splits": dict(Counter(t["split"] for t in truths)),
        "resampled_undetected": dict(resampled),
        "settings": {"ingest_interval_s": settings.ingest_interval_s,
                     "transform_interval_s": settings.transform_interval_s,
                     "freshness_slo_s": settings.freshness_slo_s},
        "note": "Synthetic development data. Report final results on sandbox incidents.",
    }
    return incidents, truths, manifest


def write_dataset(out_dir: Path, incidents: list[dict], truths: list[dict], manifest: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "incidents.jsonl").write_text("".join(json.dumps(i) + "\n" for i in incidents), encoding="utf-8")
    (out_dir / "ground_truth.jsonl").write_text("".join(json.dumps(t) + "\n" for t in truths), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
