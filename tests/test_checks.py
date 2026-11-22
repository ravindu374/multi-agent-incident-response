"""Executable / grounded action checks (controller/checks.py) and the culprit check (oncall/agent_eval.py)."""
import unittest

from controller.checks import (action_problems, evidence_text, executable_problems, grounding_problems, identifiers,
                               location_problems)
from controller.evidence import ProcessInfo
from controller.plan import INTENTS, PlannedAction
from oncall.agent_eval import hits_fix_target


def act(intent: str, target: str = "warehouse", **params) -> PlannedAction:
    return PlannedAction(intent=intent, target=target, params=params, risk=INTENTS[intent].risk, rationale="test")


class ExecutableTest(unittest.TestCase):
    def test_actions_must_name_what_they_act_on(self):
        self.assertTrue(executable_problems(act("stop_session")))
        self.assertTrue(executable_problems(act("block_client", "metrics-api")))
        self.assertTrue(executable_problems(act("reroute_connection", "ingest", route="source")))  # no endpoint
        for ok in (act("stop_session", pid=6079), act("stop_session", application_name="psql"),
                   act("block_client", "metrics-api", ip="172.18.0.5"), act("restore_config", "metrics-api"),
                   act("reroute_connection", "ingest", route="source", endpoint="secondary")):
            with self.subTest(action=ok.intent):
                self.assertEqual(executable_problems(ok), [])

    def test_target_must_be_a_platform_component(self):
        self.assertIn("not a platform component", executable_problems(act("stop_session", "ingest session", pid=1))[0])

    def test_target_must_suit_the_intent(self):
        self.assertIn("acts on warehouse", executable_problems(act("stop_session", "prometheus", pid=1))[0])
        self.assertEqual(executable_problems(act("stop_process", "transform", pids=[3])), [])  # any container

    def test_every_alternative_is_listed(self):
        self.assertEqual(len(INTENTS["stop_session"].alternatives()), 5)
        self.assertEqual(INTENTS["reroute_connection"].alternatives(), [("route", "endpoint")])
        self.assertEqual(INTENTS["restore_config"].alternatives(), [()])

    def test_empty_values_do_not_count(self):
        self.assertTrue(executable_problems(act("stop_session", pid=None, pids=[], application_name="")))

    def test_a_mapping_that_renames_nothing_is_not_a_fix(self):
        self.assertIn("changes nothing", executable_problems(act("map_schema_fields", "ingest",
                                                                 mapping={"amount": "amount"}))[0])
        self.assertEqual(executable_problems(act("map_schema_fields", "ingest",
                                                 mapping={"total_amount": "amount", "order_id": "order_id"})), [])


class GroundingTest(unittest.TestCase):
    TEXT = evidence_text({"locks": [{"pid": 6079, "granted": True}], "failed_logins": {"172.18.0.55": 212}},
                         {"logs": "ingest: client=10.0.0.7 failed"})

    def test_ids_must_appear_in_what_the_agent_saw(self):
        self.assertEqual(grounding_problems(act("stop_session", pid=6079), self.TEXT), [])
        self.assertEqual(grounding_problems(act("block_client", "metrics-api", ip="10.0.0.7"), self.TEXT), [])
        self.assertTrue(grounding_problems(act("stop_session", pid=401), self.TEXT))

    def test_whole_tokens_only(self):
        self.assertTrue(grounding_problems(act("stop_session", pid=60), self.TEXT))  # not a prefix of 6079
        self.assertTrue(grounding_problems(act("block_client", "metrics-api", ip="172.18.0.5"), self.TEXT))

    def test_non_identifying_params_are_not_grounded(self):
        a = act("resize_compute", "transform", memory_mb=512)
        self.assertEqual(action_problems(a, self.TEXT), [])

    def test_identifiers_are_normalised(self):
        ids = identifiers(act("stop_session", pid=7.0, pids=[8, 9.0], exclude_application_name="ingest"))
        self.assertEqual(ids["pid"], ["7", "8", "9"])
        self.assertEqual(identifiers(act("map_schema_fields", "ingest", mapping={"total_amount": "amount"}))["mapping"],
                         ["total_amount"])


class LocationTest(unittest.TestCase):
    RESULTS = {"top_processes": {
        "warehouse": [ProcessInfo(pid=4629, cpu_pct=30, command="sh /usr/local/lib/maint/backup-compress.sh 6"),
                      ProcessInfo(pid=5118, cpu_pct=40, command="postgres: analyst warehouse 172.18.0.25(41234) SELECT"),
                      ProcessInfo(pid=1, cpu_pct=0.4, command="postgres"),
                      ProcessInfo(pid=3181, cpu_pct=0.2, command="postgres: checkpointer")],
        "transform": [ProcessInfo(pid=77, cpu_pct=3, command="dbt build")]}}

    def test_a_process_must_be_in_the_targets_process_list(self):
        self.assertEqual(location_problems(act("stop_process", "warehouse", pids=[4629]), self.RESULTS), [])
        # what llama3 proposed: a warehouse query's pid, "stopped" on transform
        self.assertIn("not in transform's process list",
                      location_problems(act("stop_process", "transform", pid=5118), self.RESULTS)[0])

    def test_never_stop_the_component_itself(self):
        # what both LLMs did on I1: every pid in the warehouse's list, the burners and the database alike
        problems = " ".join(location_problems(act("stop_process", "warehouse", pids=[4629, 1, 3181]), self.RESULTS))
        self.assertIn("pid 1 is warehouse's main process", problems)
        self.assertIn("3181 are warehouse's own database processes", problems)
        # what llama3 did on D5: a notebook's queries killed as processes; they are sessions (data domain)
        backend = location_problems(act("stop_process", "warehouse", pids=[5118]), self.RESULTS)
        self.assertIn("ended with stop_session", backend[0])
        self.assertEqual(location_problems(act("stop_session", pid=5118), self.RESULTS), [])
        # a process list without the database's processes in it: nothing to protect beyond pid 1
        self.assertEqual(location_problems(act("stop_process", "transform", pid=77), self.RESULTS), [])

    def test_no_process_list_no_stop_process(self):
        self.assertTrue(location_problems(act("stop_process", "warehouse", pids=[5118]), {}))
        self.assertEqual(location_problems(act("stop_session", pid=5118), {}), [])  # sessions: not checked here

    def test_a_field_to_map_must_be_one_the_source_sends(self):
        sample = {"source_sample": ["created_at", "customer_id", "order_id", "status", "store_id", "total_amount"]}
        right = act("map_schema_fields", "ingest", mapping={"total_amount": "amount"})
        backwards = act("map_schema_fields", "ingest", mapping={"amount": "total_amount"})  # the old prompt's reading
        self.assertEqual(location_problems(right, sample), [])
        self.assertIn("not sent by the source", location_problems(backwards, sample)[0])
        self.assertTrue(location_problems(right, {}))  # never looked at the source's fields

    def test_fields_mapped_onto_themselves_change_nothing(self):
        mapping = {"amount_lkr": "amount", "created_at": "created_at", "order_id": "order_id"}
        a = act("map_schema_fields", "ingest", mapping=mapping)
        self.assertEqual(identifiers(a)["mapping"], ["amount_lkr"])
        self.assertTrue(hits_fix_target(a, {"map_schema_fields": [{"mapping": ["amount_lkr"]}]}))


class FixTargetTest(unittest.TestCase):
    D3 = {"stop_session": [{"pid": [6079]}, {"application_name": ["adhoc-maintenance"]}, {"usename": ["maint"]}]}
    I1 = {"stop_process": [{"target": ["warehouse"], "pid": [11, 12, 13]}]}

    def test_the_lock_holder_is_the_culprit_not_the_waiter(self):
        self.assertTrue(hits_fix_target(act("stop_session", pid=6079), self.D3))
        self.assertTrue(hits_fix_target(act("stop_session", application_name="adhoc-maintenance"), self.D3))
        self.assertFalse(hits_fix_target(act("stop_session", pid=4211), self.D3))  # the ingest session
        self.assertFalse(hits_fix_target(act("stop_session", application_name="ingest"), self.D3))

    def test_every_named_thing_must_be_a_culprit(self):
        self.assertTrue(hits_fix_target(act("stop_process", pids=[11, 12]), self.I1))
        self.assertFalse(hits_fix_target(act("stop_process", pids=[11, 99]), self.I1))  # 99 is innocent
        self.assertFalse(hits_fix_target(act("stop_process", "transform", pids=[11]), self.I1))  # wrong container

    def test_no_known_target_means_not_checked(self):
        self.assertTrue(hits_fix_target(act("quarantine_records"), {}))
        self.assertTrue(hits_fix_target(act("stop_session", pid=1), None))


if __name__ == "__main__":
    unittest.main()
