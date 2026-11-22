"""Block 5a-2: each expert agent as a real A2A server (official a2a-sdk, JSON-RPC over HTTP).

What one server does:
    GET  /.well-known/agent-card.json   -> the agent's card (discovery)
    POST /a2a  (JSON-RPC SendMessage)   -> runs SpecialistAgent.handle(task) and returns an A2A
                                           task: submitted -> working -> completed, with the
                                           Proposal as artifact "proposal" (or, for a task with
                                           role "review", SpecialistAgent.review -> artifact "review")
A message without a valid InvestigationTask is *rejected*; a crash inside the agent *fails* the
task. Either way the caller gets a clear state, never a made-up proposal.

The policy agent is served the same way: it takes a PolicyRequest (plan + priority) and answers with
artifact "policy" (a PolicyDecision from controller/policy.py's deterministic rules).

The reasoner is ordinary blocking code (the Ollama reasoner waits on HTTP), so it runs in a worker
thread, and at most `maxConcurrency` (from the card) investigations run at once; extra requests wait.

    python -m agents serve data/synthetic/dev/incidents.jsonl          # 9 investigators + the policy agent
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from urllib.parse import urlparse

from a2a.helpers import new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import TaskState
from starlette.applications import Starlette

from controller.plan import PolicyDecision
from controller.proposal import Proposal, Review
from controller.wire import (POLICY_ARTIFACT, PROPOSAL_ARTIFACT, REVIEW_ARTIFACT, read_policy_request, read_task,
                             result_parts, to_sdk_card)

from .policy_agent import PolicyAgent
from .specialist import SpecialistAgent

log = logging.getLogger(__name__)


def summary(r: Proposal | Review | PolicyDecision) -> str:
    if isinstance(r, PolicyDecision):
        verdict = "approved" if r.approved else "not approved"
        return f"policy: {verdict}{', needs a human' if r.needs_human else ''} - {'; '.join(r.reasons)}"
    if isinstance(r, Review):
        return f"{r.reviewer} {r.stance} {r.claim_agent}'s claim ({r.confidence:.2f}) - {r.reason}"
    head = f"{r.agent_id}: {r.verdict} ({r.confidence:.2f})"
    return head + (f" - {r.root_cause}" if r.root_cause else "")


class InvestigationExecutor(AgentExecutor):
    """Bridges A2A requests to SpecialistAgent.handle."""

    def __init__(self, agent: SpecialistAgent):
        self.agent = agent
        self.slots = asyncio.Semaphore(max(1, agent.profile.max_concurrency))

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is None:  # A2A: publish the Task before any status update
            await event_queue.enqueue_event(new_task(context.task_id, context.context_id,
                                                     TaskState.TASK_STATE_SUBMITTED, history=[context.message]))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        try:
            task = read_task(context.message)
            if task.role == "review" and task.claim is None:
                raise ValueError("a review task must carry the claim to review")
        except ValueError as e:
            await updater.reject(updater.new_agent_message([new_text_part(f"rejected: {e}")]))
            return
        reviewing = task.role == "review"
        async with self.slots:
            await updater.start_work(updater.new_agent_message([new_text_part(
                f"{self.agent.agent_id} {'reviewing a claim on' if reviewing else 'investigating'} "
                f"{task.incident.incident_id} ({task.domain})")]))
            try:
                work = self.agent.review if reviewing else self.agent.handle
                result = await asyncio.to_thread(work, task)
            except Exception as e:  # the agent broke: say so, do not guess
                log.exception("%s failed on %s", self.agent.agent_id, task.incident.incident_id)
                await updater.failed(updater.new_agent_message([new_text_part(f"agent error: {e!r}")]))
                return
        await updater.add_artifact(result_parts(result), name=REVIEW_ARTIFACT if reviewing else PROPOSAL_ARTIFACT)
        await updater.complete(updater.new_agent_message([new_text_part(summary(result))]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


class PolicyExecutor(AgentExecutor):
    """Bridges A2A requests to PolicyAgent.decide."""

    def __init__(self, agent: PolicyAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is None:
            await event_queue.enqueue_event(new_task(context.task_id, context.context_id,
                                                     TaskState.TASK_STATE_SUBMITTED, history=[context.message]))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        try:
            request = read_policy_request(context.message)
        except ValueError as e:
            await updater.reject(updater.new_agent_message([new_text_part(f"rejected: {e}")]))
            return
        decision = self.agent.decide(request)  # pure rules: fast, no thread needed
        await updater.add_artifact(result_parts(decision), name=POLICY_ARTIFACT)
        await updater.complete(updater.new_agent_message([new_text_part(summary(decision))]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def build_app(agent: SpecialistAgent | PolicyAgent) -> Starlette:
    """A Starlette app serving one agent's card and its JSON-RPC endpoint (path taken from the card)."""
    card = to_sdk_card(agent.card)
    executor = PolicyExecutor(agent) if isinstance(agent, PolicyAgent) else InvestigationExecutor(agent)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
    rpc_path = urlparse(agent.card.supported_interfaces[0].url).path or "/"
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url=rpc_path))


def port_of(agent: SpecialistAgent | PolicyAgent) -> int:
    url = urlparse(agent.card.supported_interfaces[0].url)
    return url.port or (443 if url.scheme == "https" else 80)


async def serve(agents: list[SpecialistAgent | PolicyAgent], host: str = "127.0.0.1", log_level: str = "warning") -> None:
    """Run several agent servers in one process, each on the port from its card. Ctrl+C stops all."""
    import uvicorn

    class _Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):  # one Ctrl+C must stop every server, not just the last one started
            yield

    servers = [_Server(uvicorn.Config(build_app(a), host=host, port=port_of(a), log_level=log_level))
               for a in agents]
    await asyncio.gather(*(s.serve() for s in servers))
