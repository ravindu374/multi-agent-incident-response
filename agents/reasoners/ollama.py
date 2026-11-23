"""LLM reasoner: a local Ollama model reads the agent's tool results and answers as a Proposal.

The model is forced into the Proposal shape with Ollama's structured outputs (`format` = JSON
Schema), at temperature 0. The schema also limits finding tools to the tools the agent called and
action targets to real platform components. Its answer is then checked by the agent (agents/specialist.py):
allowed intents only, every action executable and grounded in the evidence, risk never lowered.

What the LLM runs taught:
  - The model writes the fields in the schema's order, so it must observe before it judges:
    findings -> root_cause -> verdict -> points_to -> confidence -> actions (and reason -> stance for a review).
    (Both models had answered "not mine" while describing their own domain's fault in the same answer.)
  - The model must be told what its domain covers. That text is the domain's skills as published on
    its specialists' Agent Cards (controller/pool.py SKILLS), so nothing is added outside the cards.
  - A fix that fails the checks gets ONE repair turn: the model sees exactly what was rejected and why
    (pid 1 is the warehouse's main process; the source sends total_amount, not "<field ...>") and answers
    again. The agent still checks the second answer; nothing passes that the checks would not pass.
  - Answers are bounded (num_predict, maxItems, maxLength): a small model under a JSON grammar can keep
    writing until the 180 s timeout. Calls to one Ollama server go one at a time (it runs them one after
    the other on a small GPU anyway), so a call's timeout measures that call, not the queue before it.
  - A call that still fails (after one retry on a 500) raises ReasonerFailed: the agent fails its task,
    and the controller asks the domain's fallback agent instead of reading a fake "unsure".

Needs the Ollama server running (`ollama serve`) and the model pulled (e.g. `ollama pull qwen2.5:3b`).
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
from pydantic import BaseModel

from controller import DOMAINS
from controller.cards import DOMAIN_TOOLS, SHARED_TOOLS, RoutingProfile
from controller.checks import CHECK, action_problems, evidence_text, location_problems
from controller.history import describe_fix
from controller.plan import COMPONENTS, INTENTS, PARAM_SCHEMAS, PlannedAction
from controller.pool import SKILLS
from controller.proposal import Finding, InvestigationTask, Proposal, Review

from ..tools import Toolbox
from . import ReasonerFailed

LLM_ERRORS = (httpx.HTTPError, KeyError, ValueError)  # transport, HTTP status, unparsable answer

PLATFORM = ("an on-call team for a data platform "
            "(orders API -> ingest job -> Postgres warehouse -> dbt transform -> dashboard API)")

# Who owns what where two domains see the same symptom. A platform rule, the same for every agent.
BOUNDARIES = """- A query or session running in the warehouse belongs to the data domain, even though it also shows \
up as a "postgres: ..." process in the warehouse container's process list.
- A job that fails or waits is usually where the problem shows, not where it is."""


def domain_scope(domain: str) -> str:
    """What a domain owns, in its specialists' own published words (their Agent Card skills)."""
    lines = []
    for s in SKILLS.get(domain, []):
        signs = f" Typical signs: {'; '.join(s.examples)}." if s.examples else ""
        lines.append(f"- {s.name}: {s.description}{signs}")
    return "\n".join(lines) or "- (no skills published)"


SYSTEM_PROMPT = """You are the {domain} agent on {platform}.

WHAT THE {upper} DOMAIN COVERS
{scope}

WHO OWNS WHAT
{boundaries}

Answer in this order, using only the evidence given:
1. findings: one observed fact per item, with the tool whose result shows it. Read YOUR TOOL RESULTS first.
2. root_cause: one full sentence naming the underlying cause (the session, process, configuration, client
   or network path responsible), not a component name and not the error message the failing jobs report.
3. verdict:
   - "owner": the cause you named is one of the problems the {domain} domain covers (listed above) and your
     tool results show it. This holds even when the failure shows up in another component's job or alert.
   - "not_mine": your tool results look healthy, or they show the cause belongs to another domain.
   - "unsure": your evidence is missing or does not settle it.
4. points_to: the domain the evidence points to if your verdict is not_mine, otherwise "none".
5. confidence: the probability that your verdict is right (0-1).
6. actions ({which}): each action must say exactly what to act on, with the params listed for its intent.
   Copy identifiers (pid, IP address, application name, user, role, mount, field names) from YOUR TOOL
   RESULTS; never invent them. target must be one of the platform components.
Answer with JSON only."""

WHICH_ACTIONS = {
    False: "owner only",
    True: "always at least one: the controller chose you for this incident, so give your best fix in your "
          "domain even when your verdict is not_mine or unsure. Your verdict tells how strongly your evidence "
          "backs it, and the controller weighs every agent's fix before anything runs",
}


def system_prompt(domain: str, must_propose: bool = False) -> str:
    return SYSTEM_PROMPT.format(domain=domain, upper=domain.upper(), platform=PLATFORM, scope=domain_scope(domain),
                                boundaries=BOUNDARIES, which=WHICH_ACTIONS[must_propose])


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


CITABLE = ("alerts", *SHARED_TOOLS)  # sources every agent has: the alerts, logs and job runs


def action_schema(intent: str, required: tuple[str, ...]) -> dict:
    """One allowed shape of an action: this intent, one of its targets, and the params it needs (required,
    so the model has to fill them in rather than leave params empty)."""
    spec = INTENTS[intent]
    keys = list(required) + [k for k in spec.optional if k not in required]
    return {"type": "object", "properties": {
        "intent": {"type": "string", "enum": [intent]},
        "target": {"type": "string", "enum": list(spec.targets)},
        "params": {"type": "object", "properties": {k: PARAM_SCHEMAS[k] for k in keys}, "required": list(required),
                   "additionalProperties": False},
        "rationale": {"type": "string", "maxLength": 300}},
        "required": ["intent", "target", "params", "rationale"]}


def response_schema(allowed_intents: list[str], tools: list[str] | None = None, must_propose: bool = False) -> dict:
    shapes = [action_schema(i, alt) for i in allowed_intents for alt in INTENTS[i].alternatives()]
    actions = {"type": "array", "items": {"anyOf": shapes} if shapes else {"type": "object"},
               "maxItems": 3 if shapes else 0}
    if must_propose and shapes:
        actions["minItems"] = 1  # chosen by the classifier: an answer without a fix is not allowed
    properties = {  # in the order the model writes them: observe, explain, then judge, then act
        "findings": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "object", "properties": {
            "tool": {"type": "string", "enum": [*(tools or []), *CITABLE]},
            "fact": {"type": "string", "maxLength": 300}}, "required": ["tool", "fact"]}},
        "root_cause": {"type": "string", "minLength": 20, "maxLength": 400},  # a sentence, not a component name
        "verdict": {"type": "string", "enum": ["owner", "not_mine", "unsure"]},
        "points_to": {"type": "string", "enum": [*DOMAINS, "none"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "actions": actions,
    }
    return {"type": "object", "properties": properties, "required": list(properties)}


def build_prompt(task: InvestigationTask, profile: RoutingProfile, results: dict[str, Any]) -> str:
    intents = "\n".join(
        f"- {i} (risk {s.risk}; target: {' or '.join(s.targets) if s.targets != COMPONENTS else 'any component'}): "
        f"{s.summary}\n    params: {s.needs_text()}"
        for i, s in ((i, INTENTS[i]) for i in profile.allowed_intents))
    alerts = "\n".join(f"- {a.name}: {a.detail}" for a in task.incident.alerts)
    logs = "\n".join(f"--- {svc} ---\n{text.strip()[-1500:]}" for svc, text in task.incident.logs.items() if text.strip())
    evidence = json.dumps({t: (v if v is not None else "NO DATA") for t, v in results.items()}, indent=1, default=str)
    question = f"\nThe primary agent asks: {task.question}\n" if task.question else ""
    return (f"Incident {task.incident.incident_id} (you are the {task.role}).{question}\n\nALERTS\n{alerts}\n\n"
            f"RECENT LOGS\n{logs}\n\nYOUR TOOL RESULTS ({task.domain})\n{evidence}\n\n{past_cases_text(task)}"
            f"ALLOWED INTENTS\n{intents}\n\nPLATFORM COMPONENTS (valid targets)\n{', '.join(COMPONENTS)}")


PAST_CASES_MAX = 3


def past_cases_text(task: InvestigationTask) -> str:
    """Block 6c: the similar past incidents, as kinds of fix (no identifiers to copy). Empty when there are none."""
    if not task.past_cases:
        return ""
    lines = ["SIMILAR PAST INCIDENTS (verified outcomes; a hint of where to look, not evidence about today. Their "
             "identifiers belong to those incidents: take today's from YOUR TOOL RESULTS)"]
    for c in task.past_cases[:PAST_CASES_MAX]:
        fix = "; ".join(describe_fix(c.fix_that_worked)) or "a person (no recorded fix)"
        said = ", ".join(f"{x.agent_id} said {x.verdict} ({x.label})" for x in c.answers if x.label in ("right", "wrong"))
        lines.append(f"- {c.similarity:.2f} alike: owner {c.owner or 'unknown'}; fixed by {fix}"
                     + (f"; then {said}" if said else ""))
    return "\n".join(lines) + "\n\n"


REPAIR_PROMPT = """The controller checked your actions against your tool results and REJECTED:
{problems}
Answer again in the same JSON format. Keep the findings that are right. Replace each rejected action with \
one that passes these checks, copying identifiers from YOUR TOOL RESULTS.{must}"""

REPAIR_MUST = {
    False: " If no action in your domain passes them, give no action and set the verdict to match your evidence.",
    True: " If the checks show the cause is in another domain, say so in verdict and points_to, and still give "
          "your best fix in your own domain.",
}


def rejected_actions(actions: list[PlannedAction], task: InvestigationTask, toolbox: Toolbox) -> list[str]:
    """The agent's own checks (agents/specialist.py), run early to decide on a repair turn: one line per
    rejected action, saying why."""
    seen = evidence_text(toolbox.results, task.incident)
    out = []
    for a in actions:
        problems = action_problems(a, seen) + location_problems(a, toolbox.results)
        if problems:
            out.append(f"- {a.intent} on {a.target} {json.dumps(a.params)}: {'; '.join(problems)}")
    return out


REVIEW_PROMPT = """You are the {domain} agent on {platform}.

WHAT THE {upper} DOMAIN COVERS
{scope}

WHO OWNS WHAT
{boundaries}

Another agent claims to own this incident. Judge the claim ONLY from YOUR tool results ({domain}).
Answer in this order:
1. reason: one sentence quoting the fact from your tool results that bears on the claim.
2. stance:
   - "supports": that fact agrees with the claim, or points to the claimed domain.
   - "contradicts": that fact is inconsistent with the claim, or shows a different root cause that the
     {domain} domain covers.
   - "cannot_tell": your tool results say nothing either way (e.g. your domain is simply healthy).
3. confidence: the probability that your stance is right (0-1).
Answer with JSON only."""


def review_prompt(domain: str) -> str:
    return REVIEW_PROMPT.format(domain=domain, upper=domain.upper(), platform=PLATFORM, scope=domain_scope(domain),
                                boundaries=BOUNDARIES)


REVIEW_SCHEMA = {  # reason first: the model states the fact before it takes a side
    "type": "object",
    "properties": {"reason": {"type": "string", "minLength": 20, "maxLength": 500},
                   "stance": {"type": "string", "enum": ["supports", "contradicts", "cannot_tell"]},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["reason", "stance", "confidence"],
}


def build_review_prompt(task: InvestigationTask, results: dict[str, Any]) -> str:
    c = task.claim
    fixes = "; ".join(f"{a.intent} on {a.target} {a.params}" for a in c.actions) or "none"
    alerts = "\n".join(f"- {a.name}: {a.detail}" for a in task.incident.alerts)
    evidence = json.dumps({t: (v if v is not None else "NO DATA") for t, v in results.items()}, indent=1, default=str)
    return (f"Incident {task.incident.incident_id}.\n\nTHE CLAIM ({c.agent_id}, {c.domain} domain, confidence "
            f"{c.confidence:.2f})\nroot cause: {c.root_cause}\nproposed fix: {fixes}\n\nALERTS\n{alerts}\n\n"
            f"YOUR TOOL RESULTS ({task.domain})\n{evidence}")


_GATES: dict[str, threading.Semaphore] = {}  # one per Ollama server, shared by every agent in this process
_GATES_LOCK = threading.Lock()


def _gate(base_url: str, max_parallel: int) -> threading.Semaphore:
    with _GATES_LOCK:
        return _GATES.setdefault(base_url, threading.BoundedSemaphore(max_parallel))


class OllamaReasoner:
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str = "http://localhost:11434",
                 client: httpx.Client | None = None, timeout_s: float = 180.0, retry_wait_s: float = 3.0,
                 max_answer_tokens: int = 1024, max_parallel: int = 1, repair: bool = True):
        self.model, self.base_url, self.client, self.timeout_s = model, base_url.rstrip("/"), client, timeout_s
        self.retry_wait_s, self.max_answer_tokens, self.repair = retry_wait_s, max_answer_tokens, repair
        self.gate = _gate(self.base_url, max_parallel)

    def _model_for(self, profile: RoutingProfile) -> str:
        return self.model or profile.model.split("ollama/")[-1]

    @staticmethod
    def _worth_retrying(exc: httpx.HTTPError) -> bool:
        # Ollama answers 500 when a model runner dies while models are swapped on a small GPU; a second
        # try usually works. A timeout is not retried: it would only double the wait.
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        return isinstance(exc, httpx.TransportError) and not isinstance(exc, httpx.TimeoutException)

    def _chat(self, model: str, messages: list[dict], schema: dict) -> tuple[dict, dict]:
        """One structured-output call: (parsed JSON answer, raw payload). Retries once on a server or
        connection error, then raises (LLM_ERRORS)."""
        body = {"model": model, "stream": False, "format": schema, "messages": messages,
                "options": {"temperature": 0, "num_predict": self.max_answer_tokens}}
        post = self.client.post if self.client else httpx.post
        for attempt in (1, 2):
            try:
                with self.gate:  # the timeout starts when it is this call's turn
                    resp = post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
                resp.raise_for_status()
                break
            except httpx.HTTPError as exc:
                if attempt == 2 or not self._worth_retrying(exc):
                    raise
                time.sleep(self.retry_wait_s)
        payload = resp.json()
        if payload.get("done_reason") == "length":
            raise ValueError(f"the answer was cut off after {self.max_answer_tokens} tokens (the model kept writing)")
        return json.loads(payload["message"]["content"]), payload

    @staticmethod
    def _failed(what: str, model: str, exc: Exception) -> ReasonerFailed:
        return ReasonerFailed(f"{what} with {model} failed: {type(exc).__name__}: {str(exc)[:200]}")

    @staticmethod
    def _usage(payload: dict) -> dict:
        return {"prompt_tokens": payload.get("prompt_eval_count"), "completion_tokens": payload.get("eval_count"),
                "seconds": round((payload.get("total_duration") or 0) / 1e9, 2)}

    @staticmethod
    def _tools(task: InvestigationTask, profile: RoutingProfile) -> list[str]:
        return [t for t in DOMAIN_TOOLS.get(task.domain, {}) if t in profile.tools]

    def investigate(self, task: InvestigationTask, agent_id: str, profile: RoutingProfile,
                    toolbox: Toolbox) -> Proposal:
        tools = self._tools(task, profile)
        results = {t: _jsonable(toolbox.try_call(t)) for t in tools}
        model = self._model_for(profile)
        base = dict(incident_id=task.incident.incident_id, agent_id=agent_id, domain=task.domain,
                    reasoner=f"{self.name}:{model}")
        messages = [{"role": "system", "content": system_prompt(task.domain, task.must_propose)},
                    {"role": "user", "content": build_prompt(task, profile, results)}]
        schema = response_schema(profile.allowed_intents, tools, task.must_propose)
        try:
            data, payload = self._chat(model, messages, schema)
            proposal = self._proposal(data, payload, base, profile)
        except LLM_ERRORS as exc:
            raise self._failed("investigation", model, exc) from exc

        # repair turn, only where the fixes count: a claim, or any answer when the agent must propose
        counts = task.must_propose or proposal.verdict == "owner"
        rejected = rejected_actions(proposal.actions, task, toolbox) if self.repair and counts else []
        if not rejected:
            return proposal
        messages += [{"role": "assistant", "content": payload["message"]["content"]},
                     {"role": "user", "content": REPAIR_PROMPT.format(problems="\n".join(rejected),
                                                                      must=REPAIR_MUST[task.must_propose])}]
        try:
            data2, payload2 = self._chat(model, messages, schema)
            revised = self._proposal(data2, payload2, base, profile)
        except LLM_ERRORS:
            return proposal  # keep the first answer; the agent's checks drop the rejected actions
        revised.findings.append(Finding(tool=CHECK, fact="revised after the checks rejected: "
                                                         + " | ".join(r[2:] for r in rejected)))
        first, second = proposal.usage or {}, revised.usage or {}
        revised.usage = {k: round((first.get(k) or 0) + (second.get(k) or 0), 2) for k in second} | {"calls": 2}
        return revised

    def _proposal(self, data: dict, payload: dict, base: dict, profile: RoutingProfile) -> Proposal:
        findings = [Finding(tool=str(f.get("tool", "")), fact=str(f.get("fact", ""))) for f in data.get("findings", [])]
        actions = []
        for a in data.get("actions", []):
            intent = a.get("intent")
            if intent not in profile.allowed_intents:
                findings.append(Finding(tool=CHECK, fact=f"dropped action with disallowed intent {intent!r}"))
                continue
            actions.append(PlannedAction(intent=intent, target=str(a.get("target", "")), params=a.get("params") or {},
                                         risk=INTENTS[intent].risk, rationale=str(a.get("rationale", ""))))
        verdict = data.get("verdict", "unsure")  # owner without a usable fix -> unsure (SpecialistAgent)
        points = data.get("points_to")
        return Proposal(
            **base, verdict=verdict if verdict in ("owner", "not_mine", "unsure") else "unsure",
            root_cause=str(data.get("root_cause", "")), confidence=min(1.0, max(0.0, float(data.get("confidence", 0)))),
            findings=findings, actions=actions, points_to=points if points in DOMAINS else None,
            usage=self._usage(payload),
        )

    def review(self, task: InvestigationTask, agent_id: str, profile: RoutingProfile, toolbox: Toolbox) -> Review:
        results = {t: _jsonable(toolbox.try_call(t)) for t in self._tools(task, profile)}
        model = self._model_for(profile)
        base = dict(incident_id=task.incident.incident_id, reviewer=agent_id, domain=task.domain,
                    claim_agent=task.claim.agent_id, claim_domain=task.claim.domain, reasoner=f"{self.name}:{model}")
        messages = [{"role": "system", "content": review_prompt(task.domain)},
                    {"role": "user", "content": build_review_prompt(task, results)}]
        try:
            data, payload = self._chat(model, messages, REVIEW_SCHEMA)
        except LLM_ERRORS as exc:
            raise self._failed("review", model, exc) from exc
        stance = data.get("stance")
        return Review(**base, stance=stance if stance in ("supports", "contradicts", "cannot_tell") else "cannot_tell",
                      reason=str(data.get("reason", "")),
                      confidence=min(1.0, max(0.0, float(data.get("confidence", 0)))), usage=self._usage(payload))
