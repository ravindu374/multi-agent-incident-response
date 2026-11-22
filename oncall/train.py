"""Train the learned classifier (Block 2b) on a labelled incident set. Lab code: it reads labels.

Fits on the `train` split only, then reports the `val` split next to the rules baseline.
The `test` split is not touched here.
"""
from __future__ import annotations

from pathlib import Path

from controller import DOMAINS
from controller.intake import extract
from controller.learned import LogRegClassifier, TrainConfig

from .evaluate import evaluate_classifier, load_dataset


def train_classifier(data_dir: Path, out: Path, device: str = "auto", config: TrainConfig | None = None) -> dict:
    train = [(report, truth) for report, truth in load_dataset(data_dir) if truth.get("split") == "train"]
    if not train:
        raise ValueError(f"no `train` split in {data_dir}")
    model = LogRegClassifier(config)
    info = model.fit([extract(r) for r, _ in train], [t["owner"] for _, t in train],
                     [t["involved"] for _, t in train], device=device)
    model.save(out)
    return {
        "info": info,
        "model_path": str(out),
        "val": evaluate_classifier(data_dir, split="val", classifier=model),
        "rules_val": evaluate_classifier(data_dir, split="val"),
        "top_features": {d: model.top_features(d, 6) for d in DOMAINS},
    }
