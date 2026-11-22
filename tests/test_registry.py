"""Block 3: Agent Cards, the default pool, and the registry's hard filter."""
import json
import unittest

import httpx

from controller import DOMAINS
from controller.cards import (
    DOMAIN_INTENTS,
    DOMAIN_TOOLS,
    ROUTING_EXTENSION_URI,
    WELL_KNOWN_CARD_PATH,
    AgentCard,
    fetch_card,
)
from controller.evidence import EvidenceBundle
from controller.pool import default_pool
from controller.registry import BREAKER_COOLDOWN_S, BREAKER_THRESHOLD, HEARTBEAT_TTL_S, AgentRegistry
from oncall.actions import ACTIONS
from oncall.faults import FAULTS
from oncall.intents import INTENT_ACTIONS
from oncall.settings import ROOT


def registry() -> AgentRegistry:
    reg = AgentRegistry(clock=lambda: 0.0)
    for card in default_pool():
        reg.register(card, now=0.0)
    return reg


class CardTest(unittest.TestCase):
    def test_published_cards_match_the_pool(self):
        for card in default_pool():
            with self.subTest(agent=card.name):
                published = json.loads((ROOT / "agents" / "cards" / f"{card.name}.json").read_text("utf-8"))
                self.assertEqual(published, card.to_json(), "run: python -m controller agents --export agents/cards")

    def test_cards_use_a2a_camel_case_and_carry_the_routing_extension(self):
        card = default_pool()[0].to_json()
        for key in ("supportedInterfaces", "defaultInputModes", "defaultOutputModes", "skills", "capabilities"):
            self.assertIn(key, card)
        self.assertEqual(card["supportedInterfaces"][0]["protocolBinding"], "JSONRPC")
        self.assertEqual(card["supportedInterfaces"][0]["protocolVersion"], "1.0")
        (ext,) = card["capabilities"]["extensions"]
        self.assertEqual(ext["uri"], ROUTING_EXTENSION_URI)
        self.assertIn("allowedIntents", ext["params"])
        self.assertEqual(AgentCard.model_validate(card).routing_profile().domains, ["network"])

    def test_card_without_routing_extension_is_refused(self):
        card = AgentCard(name="x", description="x", supported_interfaces=[{"url": "http://x"}], version="1")
        with self.assertRaises(ValueError):
            AgentRegistry().register(card)

    def test_a2a_discovery_fetches_the_well_known_card(self):
        card = default_pool()[2]

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, WELL_KNOWN_CARD_PATH)
            return httpx.Response(200, json=card.to_json())

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            fetched = fetch_card("http://data-fast:9311", client=client)
        self.assertEqual(fetched, card)


class PoolTest(unittest.TestCase):
    def test_two_variants_per_domain_plus_generalist_and_policy(self):
        profiles = {c.name: c.routing_profile() for c in default_pool()}
        for domain in DOMAINS:
            tiers = {p.tier for p in profiles.values() if p.role == "specialist" and p.domains == [domain]}
            self.assertEqual(tiers, {"fast", "strong"}, domain)
        self.assertEqual(sorted(p.domains for p in profiles.values() if p.role == "generalist"), [list(DOMAINS)])
        self.assertEqual(sum(p.role == "policy" for p in profiles.values()), 1)

    def test_strong_costs_more_than_fast(self):
        profiles = {c.name: c.routing_profile() for c in default_pool()}
        for domain in DOMAINS:
            self.assertGreater(profiles[f"{domain}-strong"].cost_units, profiles[f"{domain}-fast"].cost_units)

    def test_every_tool_returns_a_real_evidence_section(self):
        for domain, tools in DOMAIN_TOOLS.items():
            for tool, path in tools.items():
                section, field = path.split(".")
                with self.subTest(tool=tool):
                    self.assertEqual(section, domain)
                    self.assertIn(field, EvidenceBundle.model_fields[section].annotation.model_fields)


class PermissionsTest(unittest.TestCase):
    def test_domain_intents_agree_with_the_sandbox_action_permissions(self):
        for domain, intents in DOMAIN_INTENTS.items():
            for intent in intents:
                with self.subTest(domain=domain, intent=intent):
                    self.assertTrue(any(domain in ACTIONS[a].domains for a in INTENT_ACTIONS[intent]))

    def test_every_fault_is_fixable_by_its_owners_agents(self):
        from oncall.intents import intent_of

        for fault in FAULTS.values():
            with self.subTest(fault=fault.id):
                self.assertTrue({intent_of(a) for a in fault.valid_fixes} & set(DOMAIN_INTENTS[fault.owner]))


class HardFilterTest(unittest.TestCase):
    def test_domain_coverage_and_order(self):
        ok, excluded = registry().eligible("data", now=0.0)
        self.assertEqual([c.agent_id for c in ok], ["data-fast", "data-strong", "generalist"])
        self.assertEqual(excluded, [])

    def test_policy_agent_never_investigates(self):
        reg = registry()
        self.assertIn("policy", reg.check("policy", now=0.0))
        self.assertEqual(reg.policy_agents(now=0.0), ["policy"])

    def test_down_agent_is_skipped_with_a_reason(self):
        reg = registry()
        reg.mark_down("data-strong")
        ok, excluded = reg.eligible("data", now=0.0)
        self.assertNotIn("data-strong", [c.agent_id for c in ok])
        self.assertEqual([(x.agent_id, x.reason) for x in excluded], [("data-strong", "reported down")])
        reg.heartbeat("data-strong", now=1.0)
        self.assertIn("data-strong", [c.agent_id for c in reg.eligible("data", now=1.0)[0]])

    def test_missing_heartbeat_counts_as_gone(self):
        reg = registry()
        later = HEARTBEAT_TTL_S + 1
        for agent in reg.agents():
            if agent != "network-fast":
                reg.heartbeat(agent, now=later)
        ok, excluded = reg.eligible("network", now=later)
        self.assertEqual([c.agent_id for c in ok], ["network-strong", "generalist"])
        self.assertIn("no heartbeat", excluded[0].reason)

    def test_circuit_breaker_opens_half_opens_and_closes(self):
        reg = registry()
        for _ in range(BREAKER_THRESHOLD):
            reg.begin_task("infra-fast")
            reg.end_task("infra-fast", ok=False, latency_s=5.0, now=10.0)
        self.assertIn("circuit breaker open", reg.check("infra-fast", "infra", now=11.0))
        after = 10.0 + BREAKER_COOLDOWN_S + 1
        for agent in reg.agents():
            reg.heartbeat(agent, now=after)
        self.assertIsNone(reg.check("infra-fast", "infra", now=after))  # half-open: one trial allowed
        reg.begin_task("infra-fast")
        reg.end_task("infra-fast", ok=True, latency_s=3.0, now=after)
        state = reg.state("infra-fast")
        self.assertEqual((state.consecutive_failures, state.breaker_open_until), (0, None))
        self.assertAlmostEqual(state.latency_ewma_s, 0.3 * 3.0 + 0.7 * 5.0)

    def test_capacity_limit(self):
        reg = registry()
        for _ in range(reg.profile("security-strong").max_concurrency):
            reg.begin_task("security-strong")
        self.assertIn("at capacity", reg.check("security-strong", "security", now=0.0))

    def test_intent_permission_and_generalist_switch(self):
        reg = registry()
        ok, excluded = reg.eligible("network", intent="rotate_credential", now=0.0)
        self.assertEqual([c.agent_id for c in ok], ["generalist"])  # network specialists may not rotate credentials
        self.assertEqual({x.reason for x in excluded}, {"may not propose rotate_credential"})
        ok, _ = reg.eligible("security", intent="rotate_credential", include_generalist=False, now=0.0)
        self.assertEqual([c.agent_id for c in ok], ["security-fast", "security-strong"])

    def test_agents_can_join_and_leave_at_runtime(self):
        reg = AgentRegistry(clock=lambda: 0.0)
        for card in default_pool():
            if not card.name.startswith("security"):
                reg.register(card, now=0.0)
        self.assertEqual([c.agent_id for c in reg.eligible("security", now=0.0)[0]], ["generalist"])
        reg.register(next(c for c in default_pool() if c.name == "security-fast"), now=0.0)
        self.assertEqual([c.agent_id for c in reg.eligible("security", now=0.0)[0]], ["security-fast", "generalist"])
        reg.unregister("security-fast")
        self.assertNotIn("security-fast", reg.agents())


if __name__ == "__main__":
    unittest.main()
