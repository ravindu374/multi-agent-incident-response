"""Block 2a: the rules classifier (Baseline A) and team selection."""
import json
import unittest

from controller import DOMAINS
from controller.classifier import RULE_WEIGHTS, DomainScores, RuleClassifier, choose_team
from controller.contracts import IncidentFeatures, load_incidents
from controller.intake import SIGNAL_PATTERNS, extract
from oncall.settings import ROOT


def features(**signals) -> IncidentFeatures:
    return IncidentFeatures(incident_id="t", alerts=[], signals=signals, components=[], error_lines=[], text="")


def scores(**owner_probs) -> DomainScores:
    return DomainScores(incident_id="t", method="test", owner_probs=owner_probs,
                        involved_probs={d: 0.0 for d in owner_probs})


class RuleClassifierTest(unittest.TestCase):
    def test_rule_table_only_uses_known_signals_and_domains(self):
        self.assertLessEqual(set(RULE_WEIGHTS), set(SIGNAL_PATTERNS))
        for weights in RULE_WEIGHTS.values():
            self.assertLessEqual(set(weights), set(DOMAINS))

    def test_probabilities_are_well_formed(self):
        s = RuleClassifier().predict(features(disk_full=3, stale_data=1))
        self.assertAlmostEqual(sum(s.owner_probs.values()), 1.0)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in s.involved_probs.values()))
        self.assertEqual(s.ranked()[0][0], "infra")
        self.assertEqual(s.evidence["infra"], ["disk_full"])

    def test_counts_do_not_matter_only_presence(self):
        once = RuleClassifier().predict(features(lock_timeout=1)).owner_probs
        many = RuleClassifier().predict(features(lock_timeout=9)).owner_probs
        self.assertEqual(once, many)

    def test_no_evidence_means_uniform_and_unsure(self):
        s = RuleClassifier().predict(features())
        self.assertEqual(set(round(p, 6) for p in s.owner_probs.values()), {0.25})
        self.assertFalse(choose_team(s).confident)


class ChooseTeamTest(unittest.TestCase):
    def test_confident_owner_works_alone(self):
        team = choose_team(scores(network=0.9, data=0.05, infra=0.03, security=0.02))
        self.assertEqual((team.team, team.confident), (["network"], True))

    def test_unsure_owner_brings_the_other_plausible_owners(self):
        team = choose_team(scores(infra=0.6, data=0.35, network=0.03, security=0.02))
        self.assertEqual((team.owner, team.team, team.confident), ("infra", ["infra", "data"], False))

    def test_team_size_is_capped(self):
        team = choose_team(scores(network=0.25, data=0.25, infra=0.25, security=0.25), max_team=3)
        self.assertEqual(len(team.team), 3)


class SampleSetTest(unittest.TestCase):
    """Behaviour on the hand-written samples. These are not an accuracy benchmark: the rules and
    the samples were written from the same knowledge, so the rules look better here than they will
    on real sandbox incidents."""

    @classmethod
    def setUpClass(cls):
        samples = ROOT / "samples"
        cls.owner = {k: v["owner"] for k, v in json.loads((samples / "expected_intake.json").read_text()).items()}
        cls.fault = {k: v["imitates"] for k, v in json.loads((samples / "expected_intake.json").read_text()).items()}
        classifier = RuleClassifier()
        cls.teams = {r.incident_id: choose_team(classifier.predict(extract(r)))
                     for r in load_incidents(samples / "incidents.jsonl")}

    def test_true_owner_is_always_on_the_team(self):
        for incident_id, team in self.teams.items():
            with self.subTest(incident=incident_id, fault=self.fault[incident_id]):
                self.assertIn(self.owner[incident_id], team.team)

    def test_cpu_faults_are_flagged_as_unsure(self):
        """D5 (data) and I1 (infra) have identical features, so no classifier can be confident;
        the right move is to put both domains on the team."""
        for incident_id, fault in self.fault.items():
            if fault in ("D5", "I1"):
                team = self.teams[incident_id]
                self.assertFalse(team.confident)
                self.assertLessEqual({"data", "infra"}, set(team.team))


if __name__ == "__main__":
    unittest.main()
