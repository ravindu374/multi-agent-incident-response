"""Synthetic incident generator: determinism, format, label hygiene, and built-in ambiguity."""
import json
import re
import unittest
from collections import Counter

from controller.contracts import IncidentReport
from controller.evidence import EvidenceBundle
from controller.intake import extract
from oncall.faults import FAULTS
from oncall.oracle import ALERT_NAMES
from oncall.synth import generate
from oncall.synth.scenarios import SCENARIOS

PER_FAULT = 5


class SynthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.incidents, cls.truths, cls.manifest = generate(per_fault=PER_FAULT, seed=11)
        cls.truth = {t["incident_id"]: t for t in cls.truths}

    def test_every_fault_has_a_scenario(self):
        self.assertEqual(set(SCENARIOS), set(FAULTS))

    def test_counts_and_stratified_splits(self):
        self.assertEqual(Counter(t["fault_id"] for t in self.truths), {f: PER_FAULT for f in FAULTS})
        for fault_id in FAULTS:
            splits = Counter(t["split"] for t in self.truths if t["fault_id"] == fault_id)
            self.assertEqual(splits, {"train": 3, "val": 1, "test": 1})

    def test_deterministic_for_a_seed(self):
        again, truths_again, _ = generate(per_fault=PER_FAULT, seed=11)
        self.assertEqual(json.dumps(again), json.dumps(self.incidents))
        self.assertEqual(truths_again, self.truths)
        other, _, _ = generate(per_fault=PER_FAULT, seed=12)
        self.assertNotEqual(json.dumps(other), json.dumps(self.incidents))

    def test_records_match_the_lab_format(self):
        alert_names = set(ALERT_NAMES.values())
        for record in self.incidents:
            with self.subTest(incident=record["incident_id"]):
                report = IncidentReport.from_lab_record(record)
                self.assertTrue(report.alerts)
                self.assertLessEqual({a.name for a in report.alerts}, alert_names)
                self.assertEqual(report.title, report.alerts[0].name)
                self.assertEqual(set(report.logs), {"ingest", "transform", "metrics-api"})
                EvidenceBundle.model_validate(record["evidence"]["domain"])
                extract(report)  # Block 1 must handle every generated incident

    def test_nothing_in_an_incident_reveals_its_label(self):
        fault_id = re.compile(r"\b[NDIS][1-5]\b")
        for record in self.incidents:
            text = json.dumps(record)
            with self.subTest(incident=record["incident_id"]):
                self.assertRegex(record["incident_id"], r"^syn-11-\d{5}$")
                self.assertIsNone(fault_id.search(text))
                for word in ("inject", "fault", "ground_truth", "owner", "valid_fix"):
                    self.assertNotIn(word, text.lower())

    def test_ids_do_not_follow_fault_order(self):
        order = [self.truth[r["incident_id"]]["fault_id"] for r in self.incidents]
        self.assertNotEqual(order, sorted(order, key=list(FAULTS).index))

    def test_cpu_faults_overlap_at_intake_but_evidence_tells_them_apart(self):
        """Same symptoms (data's runaway query vs infra's rogue process), different deep evidence."""
        def of(fault):
            return [r for r in self.incidents if self.truth[r["incident_id"]]["fault_id"] == fault]

        d5_signals = {frozenset(extract(IncidentReport.from_lab_record(r)).signals) for r in of("D5")}
        i1_signals = {frozenset(extract(IncidentReport.from_lab_record(r)).signals) for r in of("I1")}
        self.assertTrue(d5_signals & i1_signals, "expected at least one identical signal set")
        for record in of("I1"):
            top = record["evidence"]["domain"]["infra"]["top_processes"]
            if top:  # unless that evidence went missing on purpose
                self.assertTrue(any("backup-compress" in p["command"] for p in top["warehouse"]))
        for record in of("D5"):
            queries = record["evidence"]["domain"]["data"]["active_queries"]
            if queries:
                self.assertTrue(any(q["application_name"] == "jupyter-notebook" for q in queries))

    def test_ground_truth_is_complete(self):
        for truth in self.truths:
            with self.subTest(incident=truth["incident_id"]):
                self.assertEqual(truth["owner"], FAULTS[truth["fault_id"]].owner)
                self.assertTrue(truth["valid_intents"])
                self.assertIn("variant", truth)
                self.assertGreaterEqual(truth["detect_after_s"], 0)


if __name__ == "__main__":
    unittest.main()
