"""Block 1 against the hand-written sample set (samples/incidents.jsonl, one incident per fault)."""
import json
import unittest

from controller.contracts import load_incidents
from controller.intake import extract
from oncall.settings import ROOT

SAMPLES = ROOT / "samples"


class SampleSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads((SAMPLES / "expected_intake.json").read_text("utf-8"))
        cls.features = {r.incident_id: extract(r) for r in load_incidents(SAMPLES / "incidents.jsonl")}

    def test_every_sample_has_expectations(self):
        self.assertEqual(set(self.features), set(self.expected))
        self.assertEqual(len(self.features), 14)

    def test_alerts_components_and_signals(self):
        for incident_id, exp in self.expected.items():
            f = self.features[incident_id]
            with self.subTest(incident=incident_id, imitates=exp["imitates"]):
                self.assertEqual(f.alerts, exp["alerts"])
                self.assertEqual(f.components, exp["components"])
                missing = set(exp["must_have_signals"]) - set(f.signals)
                self.assertFalse(missing, f"missing signals {missing}; got {f.signals}")
                wrong = set(exp["must_not_have_signals"]) & set(f.signals)
                self.assertFalse(wrong, f"unexpected signals {wrong}; got {f.signals}")

    def test_masked_text_has_no_volatile_tokens(self):
        for incident_id, f in self.features.items():
            with self.subTest(incident=incident_id):
                self.assertNotRegex(f.text, r"\d{4}-\d{2}-\d{2}")  # timestamps
                self.assertNotRegex(f.text, r"\b\d{1,3}(\.\d{1,3}){3}\b")  # IP addresses

    def test_duplicate_log_lines_collapse(self):
        by_fault = {exp["imitates"]: self.features[i] for i, exp in self.expected.items()}
        self.assertEqual(len(by_fault["N1"].error_lines), 1)  # 4 identical timeouts -> 1 line

    def test_cpu_faults_look_the_same_at_intake(self):
        """D5 (data) and I1 (infra) are indistinguishable here, by design: only deeper evidence
        from specialist agents can tell a heavy query from a rogue process."""
        by_fault = {exp["imitates"]: self.features[i] for i, exp in self.expected.items()}
        self.assertEqual(set(by_fault["D5"].signals), set(by_fault["I1"].signals))
        self.assertEqual(by_fault["D5"].components, by_fault["I1"].components)


if __name__ == "__main__":
    unittest.main()
