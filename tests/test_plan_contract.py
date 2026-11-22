"""The output contract (RemediationPlan / ExecutionReport) and its mapping onto the sandbox."""
import unittest

from pydantic import ValidationError

from controller.__main__ import schema_documents
from controller.plan import INTENTS, RISK_ORDER, ExecutionReport, PlannedAction, RemediationPlan
from oncall.actions import ACTIONS
from oncall.faults import FAULTS
from oncall.intents import INTENT_ACTIONS, intent_of
from oncall.settings import ROOT


def plan(**overrides) -> RemediationPlan:
    base = dict(
        incident_id="inc-1", owner="infra", team=["infra", "data"],
        root_cause="unknown process using 98% CPU in the warehouse container", confidence=0.86,
        actions=[PlannedAction(intent="stop_process", target="warehouse", params={"pids": [41, 42]},
                               risk="medium", rationale="process list shows two busy loops")],
    )
    base.update(overrides)
    return RemediationPlan(**base)


class PlanModelTest(unittest.TestCase):
    def test_round_trips_through_json(self):
        p = plan()
        self.assertEqual(RemediationPlan.model_validate_json(p.model_dump_json()), p)

    def test_unknown_intent_is_rejected(self):
        with self.assertRaises(ValidationError):
            PlannedAction(intent="drop_database", target="warehouse", risk="high", rationale="x")

    def test_risk_cannot_be_downgraded(self):
        with self.assertRaises(ValidationError):
            PlannedAction(intent="rotate_credential", target="warehouse", risk="low", rationale="x")
        PlannedAction(intent="reroute_connection", target="ingest", risk="high", rationale="raising is fine")

    def test_owner_must_lead_the_team(self):
        with self.assertRaises(ValidationError):
            plan(team=["data", "infra"])
        with self.assertRaises(ValidationError):
            plan(owner="storage", team=["storage"])

    def test_execution_report(self):
        report = ExecutionReport(incident_id="inc-1", outcomes=[{"intent": "stop_process", "status": "done"}],
                                 resolved=True)
        self.assertTrue(report.resolved)


class SandboxMappingTest(unittest.TestCase):
    def test_every_intent_maps_to_real_sandbox_actions(self):
        self.assertEqual(set(INTENT_ACTIONS), set(INTENTS))
        for intent, actions in INTENT_ACTIONS.items():
            for action in actions:
                with self.subTest(intent=intent, action=action):
                    self.assertIn(action, ACTIONS)

    def test_every_sandbox_action_is_reachable_from_exactly_one_intent(self):
        covered = [a for actions in INTENT_ACTIONS.values() for a in actions]
        self.assertEqual(sorted(covered), sorted(ACTIONS))

    def test_intent_risk_matches_the_riskiest_action_behind_it(self):
        for intent, actions in INTENT_ACTIONS.items():
            with self.subTest(intent=intent):
                riskiest = max(RISK_ORDER[ACTIONS[a].risk] for a in actions)
                self.assertEqual(RISK_ORDER[INTENTS[intent].risk], riskiest)

    def test_every_correct_fix_can_be_expressed_as_an_intent(self):
        for fault in FAULTS.values():
            for action in fault.valid_fixes:
                with self.subTest(fault=fault.id, action=action):
                    self.assertIn(intent_of(action), INTENTS)


class SchemaFilesTest(unittest.TestCase):
    def test_published_schemas_are_up_to_date(self):
        for name, text in schema_documents().items():
            with self.subTest(file=name):
                published = (ROOT / "docs" / "contract" / name).read_text(encoding="utf-8")
                self.assertEqual(published, text, "run: python -m controller schema")


if __name__ == "__main__":
    unittest.main()
