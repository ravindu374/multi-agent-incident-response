"""Split conformal prediction for the owner domain: a team that contains the true owner with a
chosen probability (e.g. 90%), instead of a hand-picked confidence threshold.

Two methods:

LAC ("least ambiguous set-valued classifier") - one cut-off for everything.
  1. On held-out labelled incidents, score each by how surprised the classifier was: s = 1 - P(true owner).
  2. q = the ceil((n+1)(1-alpha))-th smallest score.
  3. Team = every domain with P(domain) >= 1 - q.
  Promise: owner on team >= 1 - alpha *on average over all incidents*. A domain the classifier
  is often wrong about (e.g. data, for heavy-query incidents) can still be missed far more often,
  hidden by the easy domains.

Mondrian (class-conditional) - one cut-off per owner domain.
  Same recipe, but q_d is computed only from calibration incidents whose true owner is d, and
  domain d joins the team when 1 - P(d) <= q_d. Promise: owner on team >= 1 - alpha *for every
  owner domain separately*. Costs a little more team size; needs enough calibration incidents
  per domain (a domain with too few is always included).

Caveat for both: the promise needs exchangeable data (calibration and new incidents from the same
distribution). Recalibrate when the data changes, e.g. from synthetic to sandbox incidents.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import DOMAINS


def _enough(n: int, alpha: float) -> bool:
    """A finite conformal quantile needs ceil((n+1)(1-alpha)) <= n, i.e. roughly n >= 1/alpha - 1."""
    return n > 0 and math.ceil((n + 1) * (1 - alpha)) <= n


def _quantile(scores: list[float], alpha: float) -> float:
    """The conformal quantile: the ceil((n+1)(1-alpha))-th smallest score, or 1.0 if n is too small."""
    n = len(scores)
    return sorted(scores)[math.ceil((n + 1) * (1 - alpha)) - 1] if _enough(n, alpha) else 1.0


class ConformalCalibrator(BaseModel):
    method: Literal["lac", "mondrian"] = "lac"
    alpha: float = Field(gt=0.0, lt=1.0)
    qhat: float = 1.0  # LAC: the single cut-off
    qhats: dict[str, float] = Field(default_factory=dict)  # Mondrian: one cut-off per owner domain
    n: int
    n_by_domain: dict[str, int] = Field(default_factory=dict)
    pooled_domains: list[str] = Field(default_factory=list)  # Mondrian domains that fell back to the shared cut-off
    classifier: str

    @classmethod
    def fit(cls, owner_probs: list[dict[str, float]], owners: list[str], alpha: float, classifier: str,
            method: str = "lac", small_domains: str = "pool") -> ConformalCalibrator:
        """small_domains (Mondrian only), for a domain with too few calibration incidents for this alpha:
          "pool"    use the shared (LAC) cut-off for it: average promise only, flagged in pooled_domains;
          "include" always put it on the team: keeps its promise, at a large cost in team size."""
        if not owner_probs or len(owner_probs) != len(owners):
            raise ValueError("need one probability dict per labelled incident")
        scores = [1.0 - probs[owner] for probs, owner in zip(owner_probs, owners)]
        n_by = {d: sum(o == d for o in owners) for d in DOMAINS}
        pooled_q = _quantile(scores, alpha)
        if method == "lac":
            return cls(method="lac", alpha=alpha, qhat=pooled_q, n=len(scores), n_by_domain=n_by,
                       classifier=classifier)
        if method != "mondrian":
            raise ValueError(f"unknown conformal method {method!r}")
        if small_domains not in ("pool", "include"):
            raise ValueError("small_domains must be 'pool' or 'include'")
        qhats, pooled = {}, []
        for d in DOMAINS:
            own = [s for s, o in zip(scores, owners) if o == d]
            if _enough(len(own), alpha):
                qhats[d] = _quantile(own, alpha)
            elif small_domains == "pool":
                qhats[d], pooled = pooled_q, pooled + [d]
            else:
                qhats[d] = 1.0
        return cls(method="mondrian", alpha=alpha, qhat=max(qhats.values()), qhats=qhats, n=len(scores),
                   n_by_domain=n_by, pooled_domains=pooled, classifier=classifier)

    def threshold(self, domain: str) -> float:
        """Minimum P(domain) for `domain` to join the team."""
        q = self.qhats.get(domain, 1.0) if self.method == "mondrian" else self.qhat
        return 1.0 - q

    def prediction_set(self, owner_probs: dict[str, float]) -> list[str]:
        """The most likely domain, plus every other domain whose probability clears its cut-off,
        most likely first. Always containing the top guess can only add coverage, so the promise
        still holds, and the primary agent is always the classifier's best guess."""
        ranked = sorted(owner_probs, key=lambda d: (-owner_probs[d], DOMAINS.index(d)))
        return [d for i, d in enumerate(ranked) if i == 0 or owner_probs[d] >= self.threshold(d) - 1e-12]

    def describe(self) -> str:
        if self.method == "lac":
            return f"conformal (lac) set at alpha={self.alpha} (P >= {self.threshold(DOMAINS[0]):.2f})"
        cuts = ", ".join(f"{d} >= {self.threshold(d):.2f}" + (" (pooled)" if d in self.pooled_domains else "")
                         for d in DOMAINS)
        return f"conformal (mondrian) set at alpha={self.alpha}: top guess + ({cuts})"

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ConformalCalibrator:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
