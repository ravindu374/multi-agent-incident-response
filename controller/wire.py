"""Block 5a-2: how tasks and proposals travel over A2A (shared by agent servers and the orchestrator).

    InvestigationTask --(A2A message: text summary + JSON data part)--> agent server
    agent server --(A2A task, state COMPLETED, artifact "proposal": JSON data part)--> caller
    review task (role "review", carries the claim) --> agent server --> artifact "review"
    PolicyRequest (a RemediationPlan + priority) --> policy agent --> artifact "policy" (PolicyDecision)

Uses the official a2a-sdk (pip install -e ".[agents]"). Our pydantic Agent Cards are converted to
the SDK's AgentCard protobuf *strictly*: an unknown field or a missing REQUIRED field is an error,
so the cards stay compliant with the A2A v1.0 specification.

One protocol detail to know: A2A data parts are google.protobuf.Value, which stores every number
as a double (pid 401 arrives as 401.0). read_task / read_proposal turn whole-number floats back
into ints before validation, so a round trip returns the same objects.
"""
from __future__ import annotations

from typing import Any

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import get_data_parts, new_data_part, new_message, new_text_part
from a2a.types import AgentCard as SdkAgentCard
from a2a.types import Message, Part, Role, SendMessageRequest, Task, TaskState
from a2a.utils.errors import InvalidParamsError
from a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.json_format import ParseDict
from pydantic import ValidationError

from .cards import AgentCard
from .plan import PolicyDecision
from .policy import PolicyRequest
from .proposal import InvestigationTask, Proposal, Review

JSON = "application/json"
PROPOSAL_ARTIFACT = "proposal"
REVIEW_ARTIFACT = "review"
POLICY_ARTIFACT = "policy"
DEFAULT_TIMEOUT_S = 120.0  # an LLM-backed strong agent can take ~20-60 s


class CardNotCompliant(ValueError):
    pass


class AgentCallFailed(RuntimeError):
    """The agent answered, but not with a proposal (rejected, failed, cancelled, ...)."""

    def __init__(self, agent: str, state: str, detail: str):
        super().__init__(f"{agent}: {state}: {detail}")
        self.agent, self.state, self.detail = agent, state, detail


# -- Agent Cards ---------------------------------------------------------------------------------

def to_sdk_card(card: AgentCard) -> SdkAgentCard:
    """Our card -> the official protobuf AgentCard. Raises CardNotCompliant if it does not fit."""
    try:
        sdk = ParseDict(card.to_json(), SdkAgentCard())  # strict: unknown fields raise
    except Exception as e:  # google.protobuf.json_format.ParseError
        raise CardNotCompliant(f"{card.name}: {e}") from e
    try:
        validate_proto_required_fields(sdk)
    except InvalidParamsError as e:
        fields = [err.get("field") for err in (getattr(e, "data", None) or {}).get("errors", [])]
        raise CardNotCompliant(f"{card.name}: REQUIRED field(s) missing: {fields}") from e
    return sdk


# -- payloads ------------------------------------------------------------------------------------

def _restore_ints(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _restore_ints(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_ints(v) for v in value]
    return value


def task_message(task: InvestigationTask) -> Message:
    """The A2A message that asks an agent to investigate (human-readable text + machine-readable JSON)."""
    if task.role == "review" and task.claim is not None:
        summary = (f"Review {task.claim.agent_id}'s claim that incident {task.incident.incident_id} is a "
                   f"{task.claim.domain} problem, using your {task.domain} evidence.")
    else:
        summary = f"Investigate incident {task.incident.incident_id} in the {task.domain} domain ({task.role})."
        if task.must_propose:
            summary += " Propose your best fix, and say how strongly your evidence backs it."
    if task.question:
        summary += f" Question: {task.question}"
    return new_message([new_text_part(summary), new_data_part(task.model_dump(mode="json"), media_type=JSON)],
                       role=Role.ROLE_USER)


def _first_valid(parts, model):
    errors = []
    for data in get_data_parts(parts):
        try:
            return model.model_validate(_restore_ints(data))
        except ValidationError as e:
            errors.append(str(e).splitlines()[0])
    raise ValueError(f"no {model.__name__} in the message's data parts" + (f" ({errors[0]})" if errors else ""))


def read_task(message: Message | None) -> InvestigationTask:
    if message is None:
        raise ValueError("the request carries no message")
    return _first_valid(message.parts, InvestigationTask)


def policy_message(request: PolicyRequest) -> Message:
    plan = request.plan
    summary = (f"Check the {plan.kind} plan for incident {plan.incident_id} ({len(plan.actions)} action(s), "
               f"priority {request.priority}) against the policy rules.")
    return new_message([new_text_part(summary), new_data_part(request.model_dump(mode="json"), media_type=JSON)],
                       role=Role.ROLE_USER)


def read_policy_request(message: Message | None) -> PolicyRequest:
    if message is None:
        raise ValueError("the request carries no message")
    return _first_valid(message.parts, PolicyRequest)


def read_policy(task: Task, agent: str = "policy") -> PolicyDecision:
    return _read(task, agent, POLICY_ARTIFACT, PolicyDecision)


def result_parts(result: Proposal | Review | PolicyDecision) -> list[Part]:
    return [new_data_part(result.model_dump(mode="json"), media_type=JSON)]


proposal_parts = result_parts  # backwards-compatible name


def _read(task: Task, agent: str, name: str, model):
    state = TaskState.Name(task.status.state).removeprefix("TASK_STATE_").lower()
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        detail = " ".join(p.text for p in task.status.message.parts if p.HasField("text")) or "no detail"
        raise AgentCallFailed(agent, state, detail)
    for artifact in task.artifacts:
        if artifact.name == name:
            return _first_valid(artifact.parts, model)
    raise AgentCallFailed(agent, state, f"completed without a {name!r} artifact")


def read_proposal(task: Task, agent: str = "agent") -> Proposal:
    """Pull the Proposal out of a finished A2A task, or say why there is none."""
    return _read(task, agent, PROPOSAL_ARTIFACT, Proposal)


def read_review(task: Task, agent: str = "agent") -> Review:
    return _read(task, agent, REVIEW_ARTIFACT, Review)


# -- calling an agent ----------------------------------------------------------------------------

async def ask(card: AgentCard, task: InvestigationTask, http: httpx.AsyncClient | None = None,
              timeout: float = DEFAULT_TIMEOUT_S) -> Proposal:
    """Send one InvestigationTask to the agent described by `card` and wait for its Proposal.

    Uses the card's first interface (JSON-RPC). Pass `http` to reuse a connection pool (the caller
    closes it); otherwise a client with `timeout` is created for this call.
    """
    return read_proposal(await _send(card, task_message(task), http, timeout), card.name)


async def ask_policy(card: AgentCard, request: PolicyRequest, http: httpx.AsyncClient | None = None,
                     timeout: float = DEFAULT_TIMEOUT_S) -> PolicyDecision:
    """Ask the policy agent to approve, reject or escalate a plan."""
    return read_policy(await _send(card, policy_message(request), http, timeout), card.name)


async def ask_review(card: AgentCard, task: InvestigationTask, http: httpx.AsyncClient | None = None,
                     timeout: float = DEFAULT_TIMEOUT_S) -> Review:
    """Ask the agent to review task.claim; returns its Review."""
    if task.role != "review" or task.claim is None:
        raise ValueError("ask_review needs a task with role 'review' and a claim")
    return read_review(await _send(card, task_message(task), http, timeout), card.name)


async def _send(card: AgentCard, message: Message, http: httpx.AsyncClient | None, timeout: float) -> Task:
    own = http is None
    http = http or httpx.AsyncClient(timeout=timeout)
    try:
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(to_sdk_card(card))
        final: Task | None = None
        async for response in client.send_message(SendMessageRequest(message=message)):
            if response.HasField("task"):
                final = response.task
            elif response.HasField("message"):
                raise AgentCallFailed(card.name, "message", "replied with a message instead of a task")
        if final is None:
            raise AgentCallFailed(card.name, "no_response", "the agent returned nothing")
        return final
    finally:
        if own:
            await http.aclose()
