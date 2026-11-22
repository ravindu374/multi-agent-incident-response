"""Block 2b - learned domain classifier: text + structured features -> logistic regression.

Features (built from Block 1's IncidentFeatures):
  * the masked incident text as TF-IDF word unigrams and bigrams (sublinear tf, L2-normalised)
  * one 0/1 feature per signal, alert name and component ("sig:disk_full", "alert:DashboardSlow", ...)

Two heads share those features:
  owner     softmax over the 4 domains  -> owner_probs    (who must fix it)
  involved  one sigmoid per domain      -> involved_probs (who holds useful evidence)

Training minimises class-weighted cross-entropy (+ binary cross-entropy for `involved`) with an
L2 penalty, using full-batch Adam. Backends: NumPy on the CPU, or PyTorch on a CUDA GPU when it is
installed. Both optimise the same loss with the same settings, so they give the same model up to
floating-point noise. At this data size the CPU is as fast as the GPU; the GPU path is for
larger datasets later.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import DOMAINS
from .classifier import DomainScores
from .contracts import IncidentFeatures

FORMAT = "a2a-oncall/logreg"
TOKEN = re.compile(r"[a-z][a-z0-9_]+")
STOPWORDS = frozenset({"the", "a", "an", "of", "in", "to", "for", "is", "at", "on", "and", "or", "by", "from",
                       "as", "be", "no", "it", "if"})
GPU_HINT = ("PyTorch with CUDA is not installed. For an RTX GPU on Windows:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
            "(about 2.5 GB). Or train on the CPU with device='cpu'.")


SECTION_HEADERS = frozenset({"title", "alerts", "evidence"})  # layout words from Block 1's text


def tokens(text: str) -> list[str]:
    """Unigrams + bigrams, line by line: a bigram never joins the end of one line to the next."""
    out: list[str] = []
    for line in text.lower().splitlines():
        words = [w for w in TOKEN.findall(line) if w not in STOPWORDS and w not in SECTION_HEADERS]
        out += words + [f"{a}__{b}" for a, b in zip(words, words[1:])]
    return out


def structured(f: IncidentFeatures) -> list[str]:
    return ([f"sig:{s}" for s in f.signals] + [f"alert:{a}" for a in f.alerts]
            + [f"comp:{c}" for c in f.components])


@dataclass
class TrainConfig:
    l2: float = 1e-3
    epochs: int = 400
    lr: float = 0.05
    min_df: int = 2  # a word must appear in at least this many training incidents
    balanced: bool = True  # weight classes so rarer owners count as much as common ones


def resolve_backend(device: str = "auto") -> str:
    """'cpu' -> NumPy. 'cuda' -> PyTorch on the GPU (error if unavailable). 'auto' -> GPU if possible."""
    if device == "cpu":
        return "numpy"
    if device not in ("auto", "cuda", "gpu"):
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    try:
        import torch
    except ImportError:
        if device == "auto":
            return "numpy"
        raise RuntimeError(GPU_HINT) from None
    if torch.cuda.is_available():
        return "cuda"
    if device == "auto":
        return "numpy"
    raise RuntimeError("PyTorch is installed but cannot see a CUDA GPU (check the driver or install a CUDA build).\n"
                       + GPU_HINT)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _fit_numpy(X, y, Y_inv, sample_w, cfg: TrainConfig) -> dict[str, np.ndarray]:
    n, d = X.shape
    k = len(DOMAINS)
    params = {"W": np.zeros((d, k)), "b": np.zeros(k), "V": np.zeros((d, k)), "c": np.zeros(k)}
    m = {key: np.zeros_like(p) for key, p in params.items()}
    v = {key: np.zeros_like(p) for key, p in params.items()}
    onehot = np.eye(k)[y]
    w = (sample_w / sample_w.sum())[:, None]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(1, cfg.epochs + 1):
        G = (_softmax(X @ params["W"] + params["b"]) - onehot) * w  # d(weighted CE)/d(logits)
        H = (_sigmoid(X @ params["V"] + params["c"]) - Y_inv) / n  # d(BCE)/d(logits)
        grads = {"W": X.T @ G + 2 * cfg.l2 * params["W"], "b": G.sum(axis=0),
                 "V": X.T @ H + 2 * cfg.l2 * params["V"], "c": H.sum(axis=0)}
        for key, g in grads.items():
            m[key] = beta1 * m[key] + (1 - beta1) * g
            v[key] = beta2 * v[key] + (1 - beta2) * g * g
            m_hat = m[key] / (1 - beta1 ** step)
            v_hat = v[key] / (1 - beta2 ** step)
            params[key] -= cfg.lr * m_hat / (np.sqrt(v_hat) + eps)
    return params


def _fit_torch(X, y, Y_inv, sample_w, cfg: TrainConfig) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    Yt = torch.tensor(Y_inv, dtype=torch.float32, device=device)
    wt = torch.tensor(sample_w, dtype=torch.float32, device=device)
    n, d = X.shape
    k = len(DOMAINS)
    W = torch.zeros(d, k, device=device, requires_grad=True)
    b = torch.zeros(k, device=device, requires_grad=True)
    V = torch.zeros(d, k, device=device, requires_grad=True)
    c = torch.zeros(k, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([W, b, V, c], lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(cfg.epochs):
        optimiser.zero_grad()
        ce = (F.cross_entropy(Xt @ W + b, yt, reduction="none") * wt).sum() / wt.sum()
        bce = F.binary_cross_entropy_with_logits(Xt @ V + c, Yt, reduction="sum") / n
        loss = ce + bce + cfg.l2 * ((W ** 2).sum() + (V ** 2).sum())
        loss.backward()
        optimiser.step()
    return {name: t.detach().cpu().numpy().astype(np.float64) for name, t in (("W", W), ("b", b), ("V", V), ("c", c))}


class LogRegClassifier:
    name = "logreg"

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0)
        self.params: dict[str, np.ndarray] = {}

    # -- features ---------------------------------------------------------------------------
    def _vector(self, f: IncidentFeatures) -> np.ndarray:
        x = np.zeros(len(self.vocab))
        for name in structured(f):
            i = self.vocab.get(name)
            if i is not None:
                x[i] = 1.0
        counts = Counter(t for t in tokens(f.text) if t in self.vocab)
        if counts:
            idx = np.fromiter((self.vocab[t] for t in counts), dtype=int, count=len(counts))
            vals = np.array([1.0 + math.log(c) for c in counts.values()]) * self.idf[idx]
            x[idx] = vals / (np.linalg.norm(vals) or 1.0)
        return x

    # -- training ---------------------------------------------------------------------------
    def fit(self, features: list[IncidentFeatures], owners: list[str], involved: list[list[str]],
            device: str = "auto") -> dict:
        if not features or not (len(features) == len(owners) == len(involved)):
            raise ValueError("features, owners and involved must be non-empty and the same length")
        unknown = set(owners) - set(DOMAINS)
        if unknown:
            raise ValueError(f"unknown owner domain(s) {unknown}")
        cfg = self.config
        backend = resolve_backend(device)
        started = time.perf_counter()

        doc_counts = [Counter(tokens(f.text)) for f in features]
        df = Counter()
        for counts in doc_counts:
            df.update(counts.keys())
        struct_names = sorted({s for f in features for s in structured(f)})
        text_names = sorted(t for t, n in df.items() if n >= cfg.min_df)
        self.vocab = {name: i for i, name in enumerate(struct_names + text_names)}
        n = len(features)
        self.idf = np.ones(len(self.vocab))
        for t in text_names:
            self.idf[self.vocab[t]] = math.log((1 + n) / (1 + df[t])) + 1.0

        X = np.stack([self._vector(f) for f in features])
        y = np.array([DOMAINS.index(o) for o in owners])
        Y_inv = np.array([[1.0 if d in inv else 0.0 for d in DOMAINS] for inv in involved])
        counts = np.bincount(y, minlength=len(DOMAINS)).astype(float)
        class_w = n / (len(DOMAINS) * np.maximum(counts, 1.0)) if cfg.balanced else np.ones(len(DOMAINS))
        sample_w = class_w[y]

        self.params = (_fit_torch if backend == "cuda" else _fit_numpy)(X, y, Y_inv, sample_w, cfg)
        train_acc = float((self._owner_probs(X).argmax(axis=1) == y).mean())
        return {"backend": backend, "incidents": n, "features": len(self.vocab), "structured": len(struct_names),
                "text_terms": len(text_names), "epochs": cfg.epochs, "train_accuracy": train_acc,
                "seconds": round(time.perf_counter() - started, 3)}

    # -- prediction -------------------------------------------------------------------------
    def _owner_probs(self, X: np.ndarray) -> np.ndarray:
        return _softmax(X @ self.params["W"] + self.params["b"])

    def predict(self, features: IncidentFeatures) -> DomainScores:
        if not self.params:
            raise RuntimeError("model is not trained or loaded")
        x = self._vector(features)
        owner = self._owner_probs(x[None, :])[0]
        involved = _sigmoid(x @ self.params["V"] + self.params["c"])
        names = list(self.vocab)
        evidence = {}
        for j in np.argsort(-owner)[:2]:  # why the two most likely domains scored as they did
            contrib = x * self.params["W"][:, j]
            top = [names[i] for i in np.argsort(-contrib)[:3] if contrib[i] > 0]
            if top:
                evidence[DOMAINS[j]] = top
        return DomainScores(
            incident_id=features.incident_id, method=self.name,
            owner_probs={d: float(p) for d, p in zip(DOMAINS, owner)},
            involved_probs={d: float(p) for d, p in zip(DOMAINS, involved)},
            evidence=evidence,
        )

    def top_features(self, domain: str, k: int = 8) -> list[tuple[str, float]]:
        """The features that push hardest towards `domain` (a quick look inside the model)."""
        j = DOMAINS.index(domain)
        names = list(self.vocab)
        weights = self.params["W"][:, j] - self.params["W"].mean(axis=1)
        return [(names[i], round(float(weights[i]), 3)) for i in np.argsort(-weights)[:k]]

    # -- persistence (plain JSON: readable, no pickle) --------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": FORMAT, "version": 1, "domains": list(DOMAINS), "config": asdict(self.config),
                   "vocab": list(self.vocab), "idf": self.idf.tolist(),
                   **{key: value.tolist() for key, value in self.params.items()}}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LogRegClassifier:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT or tuple(payload["domains"]) != DOMAINS:
            raise ValueError(f"{path} is not a compatible {FORMAT} model")
        model = cls(TrainConfig(**payload["config"]))
        model.vocab = {name: i for i, name in enumerate(payload["vocab"])}
        model.idf = np.array(payload["idf"])
        model.params = {key: np.array(payload[key]) for key in ("W", "b", "V", "c")}
        return model
