"""Block 5a-2: Agent Cards vs the official a2a-sdk types, and agents served over real A2A (in-process HTTP)."""
import asyncio
import json
import unittest
import warnings

import httpx

try:
    from agents.server import build_app
    from controller.wire import AgentCallFailed, ask, read_task, task_message, to_sdk_card
except ImportError:  # a2a-sdk not installed: pip install -e ".[agents]"
    build_app = None

from agents.reasoners.rules import RuleReasoner
from agents.specialist import SpecialistAgent
from agents.tools import RecordedEvidence
from controller.cards import AgentCard, fetch_card
from controller.contracts import IncidentReport
from controller.evidence import EvidenceBundle
from controller.pool import default_pool
from controller.proposal import InvestigationTask
from oncall.settings import ROOT
from oncall.synth import generate

CARDS = {c.name: c for c in default_pool()}


@unittest.skipIf(build_app is None, "a2a-sdk is not installed")
class A2ACase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        incidents, truths, _ = generate(per_fault=2, seed=11)
        cls.reports = [IncidentReport.from_lab_record(r) for r in incidents]
        cls.evidence = RecordedEvidence({r["incident_id"]: EvidenceBundle.model_validate(r["evidence"]["domain"])
                                         for r in incidents})

    def agent(self, name: str) -> SpecialistAgent:
        return SpecialistAgent(CARDS[name], RuleReasoner(), self.evidence)

    def over_a2a(self, agent: SpecialistAgent, tasks: list[InvestigationTask]):
        """Send tasks through the agent's real A2A app: JSON-RPC over HTTP, served in-process."""
        async def run():
            app = build_app(agent)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, timeout=30) as http:
                return [await ask(agent.card, t, http=http) for t in tasks]
        return asyncio.run(run())


class CardComplianceTest(A2ACase):
    def test_every_pool_card_is_a_valid_official_agent_card(self):
        for card in default_pool():
            with self.subTest(card=card.name):
                sdk = to_sdk_card(card)  # strict: unknown fields or missing REQUIRED fields raise
                self.assertEqual(sdk.name, card.name)
                self.assertEqual(sdk.supported_interfaces[0].protocol_version, "1.0")
                self.assertEqual(sdk.capabilities.extensions[0].uri, "urn:a2a-oncall:routing-profile:v1")

    def test_exported_card_files_are_valid_too(self):
        for path in sorted((ROOT / "agents" / "cards").glob("*.json")):
            with self.subTest(card=path.name):
                to_sdk_card(AgentCard.model_validate(json.loads(path.read_text("utf-8"))))

    def test_served_card_is_discoverable_and_keeps_the_routing_profile(self):
        agent = self.agent("data-strong")
        with warnings.catch_warnings():  # starlette's TestClient nags about its own httpx use
            warnings.simplefilter("ignore")
            from starlette.testclient import TestClient

            with TestClient(build_app(agent), base_url="http://localhost:9312") as client:
                card = fetch_card("http://localhost:9312", client=client)
        self.assertEqual(card.name, "data-strong")
        self.assertEqual(card.routing_profile(), agent.profile)  # survives the protobuf round trip


class A2AServerTest(A2ACase):
    def test_proposals_over_a2a_equal_in_process_proposals(self):
        for name in ("data-strong", "infra-fast", "network-strong", "generalist"):
            agent = self.agent(name)
            tasks = [InvestigationTask(incident=r, domain=agent.profile.domains[0]) for r in self.reports[::3]]
            with self.subTest(agent=name):
                remote = self.over_a2a(agent, tasks)
                local = [agent.handle(t) for t in tasks]
                self.assertEqual([p.model_dump() for p in remote], [p.model_dump() for p in local])

    def test_whole_numbers_come_back_as_ints(self):
        agent = self.agent("data-strong")
        tasks = [InvestigationTask(incident=r, domain="data") for r in self.reports]
        params = [a.params for p in self.over_a2a(agent, tasks) for a in p.actions if "pid" in a.params]
        self.assertTrue(params)
        self.assertTrue(all(type(p["pid"]) is int for p in params))

    def test_task_message_round_trips(self):
        task = InvestigationTask(incident=self.reports[0], domain="infra", role="consult", question="CPU hog?")
        self.assertEqual(read_task(task_message(task)), task)

    def test_a_review_over_a2a_equals_the_in_process_review(self):
        from controller.wire import ask_review

        data, infra = self.agent("data-strong"), self.agent("infra-strong")
        report = self.reports[0]
        claim = data.handle(InvestigationTask(incident=report, domain="data"))
        task = InvestigationTask(incident=report, domain="infra", role="review", claim=claim)
        self.assertEqual(read_task(task_message(task)), task)  # the claim travels inside the message

        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(infra)), timeout=30) as http:
                return await ask_review(infra.card, task, http=http)

        self.assertEqual(asyncio.run(run()), infra.review(task))

    def test_a_message_without_a_task_is_rejected(self):
        from a2a.client import ClientConfig, ClientFactory
        from a2a.helpers import new_text_message
        from a2a.types import Role, SendMessageRequest, TaskState

        async def run():
            agent = self.agent("data-fast")
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(agent))) as http:
                client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(to_sdk_card(agent.card))
                msg = new_text_message("please look at the database", role=Role.ROLE_USER)
                return [r.task async for r in client.send_message(SendMessageRequest(message=msg))][-1]

        task = asyncio.run(run())
        self.assertEqual(task.status.state, TaskState.TASK_STATE_REJECTED)

    def test_an_agent_crash_fails_the_task_instead_of_guessing(self):
        class Broken(RuleReasoner):
            def investigate(self, *a, **k):
                raise RuntimeError("tool backend down")

        agent = SpecialistAgent(CARDS["infra-strong"], Broken(), self.evidence)
        with self.assertRaises(AgentCallFailed) as ctx:
            self.over_a2a(agent, [InvestigationTask(incident=self.reports[0], domain="infra")])
        self.assertEqual(ctx.exception.state, "failed")
        self.assertIn("tool backend down", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
