"""Block 4: conformal calibration and the router (toy inputs only)."""
import random
import tempfile
import unittest
from pathlib import Path

from controller import DOMAINS
from controller.classifier import DomainScores
from controller.conformal import ConformalCalibrator
from controller.contracts import AlertIn, IncidentReport
from controller.pool import default_pool
from controller.registry import AgentRegistry
from controller.router import Router, RoutingPolicy, priority_of
from oncall.route_eval import split_for_calibration


class FixedClassifier:
    """Returns preset probabilities per incident id, so routing logic can be tested in isolation."""

    name = "fixed"

    def __init__(self, probs: dict[str, dict[str, float]]):
        self.probs = probs

    def predict(self, features):
        p = self.probs[features.incident_id]
        return DomainScores(incident_id=features.incident_id, method=self.name, owner_probs=p,
                            involved_probs={d: 0.0 for d in DOMAINS})


def probs(**kw) -> dict[str, float]:
    rest = (1.0 - sum(kw.values())) / (len(DOMAINS) - len(kw))
    return {d: kw.get(d, rest) for d in DOMAINS}


def incident(incident_id: str, *alerts: str) -> IncidentReport:
    return IncidentReport(incident_id=incident_id, title=alerts[0],
                          alerts=[AlertIn(name=a, detail="x") for a in alerts])


def registry(*down: str) -> AgentRegistry:
    reg = AgentRegistry(clock=lambda: 0.0)
    for card in default_pool():
        reg.register(card, now=0.0)
    for agent in down:
        reg.mark_down(agent)
    return reg


class ConformalTest(unittest.TestCase):
    def test_quantile_follows_the_formula(self):
        p = [probs(data=x) for x in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)]
        cal = ConformalCalibrator.fit(p, ["data"] * 9, alpha=0.2, classifier="t")
        # scores 0.1..0.9, k = ceil(10 * 0.8) = 8 -> qhat = 0.8 -> threshold 0.2
        self.assertAlmostEqual(cal.qhat, 0.8)
        self.assertEqual(cal.prediction_set(probs(data=0.25, infra=0.6)), ["infra", "data"])

    def test_too_few_incidents_means_everyone(self):
        cal = ConformalCalibrator.fit([probs(data=0.9)] * 3, ["data"] * 3, alpha=0.05, classifier="t")
        self.assertEqual(cal.qhat, 1.0)
        self.assertEqual(len(cal.prediction_set(probs(data=0.97))), 4)

    def test_never_returns_an_empty_team(self):
        cal = ConformalCalibrator(alpha=0.1, qhat=0.05, n=100, classifier="t")
        self.assertEqual(cal.prediction_set(probs(network=0.4, data=0.3)), ["network"])

    def test_coverage_promise_holds_on_exchangeable_data(self):
        rng = random.Random(0)

        def draw():
            raw = [rng.gammavariate(0.6, 1.0) for _ in DOMAINS]
            p = {d: x / sum(raw) for d, x in zip(DOMAINS, raw)}
            owner = rng.choices(DOMAINS, weights=[p[d] for d in DOMAINS])[0]  # labels follow the probabilities
            return p, owner

        cal_data = [draw() for _ in range(800)]
        cal = ConformalCalibrator.fit([p for p, _ in cal_data], [o for _, o in cal_data], alpha=0.1, classifier="t")
        test = [draw() for _ in range(3000)]
        coverage = sum(o in cal.prediction_set(p) for p, o in test) / len(test)
        self.assertGreaterEqual(coverage, 0.88)  # promise: >= 0.90 on average (small sampling slack)
        self.assertLess(sum(len(cal.prediction_set(p)) for p, _ in test) / len(test), 4.0)  # not just "everyone"

    def test_save_and_load(self):
        for cal in (ConformalCalibrator(alpha=0.1, qhat=0.42, n=50, classifier="logreg"),
                    ConformalCalibrator(method="mondrian", alpha=0.05, qhats={d: 0.3 for d in DOMAINS}, n=50,
                                        classifier="logreg")):
            with tempfile.TemporaryDirectory() as tmp:
                cal.save(Path(tmp) / "c.json")
                self.assertEqual(ConformalCalibrator.load(Path(tmp) / "c.json"), cal)


class MondrianTest(unittest.TestCase):
    @staticmethod
    def biased_world(seed: int, n: int):
        """A classifier that is systematically too unsure about `data` incidents (like D5)."""
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            owner = rng.choice(DOMAINS)
            raw = {d: rng.gammavariate(0.5, 1.0) for d in DOMAINS}
            raw[owner] += rng.uniform(1.0, 3.0) * (0.25 if owner == "data" else 1.0)
            total = sum(raw.values())
            out.append(({d: v / total for d, v in raw.items()}, owner))
        return out

    def test_one_cut_off_per_domain(self):
        p = [probs(data=0.9), probs(data=0.4), probs(infra=0.95), probs(infra=0.85)]
        cal = ConformalCalibrator.fit(p, ["data", "data", "infra", "infra"], alpha=0.4, classifier="t",
                                      method="mondrian")
        # k = ceil(3 * 0.6) = 2 of 2 -> each domain's worst score
        self.assertAlmostEqual(cal.qhats["data"], 0.6)
        self.assertAlmostEqual(cal.qhats["infra"], 0.15)
        self.assertEqual(cal.n_by_domain["data"], 2)
        # network/security had no calibration incidents -> pooled cut-off over all 4 scores (k = 3 -> 0.15)
        self.assertEqual(cal.pooled_domains, ["network", "security"])
        self.assertAlmostEqual(cal.qhats["network"], 0.15)
        strict = ConformalCalibrator.fit(p, ["data", "data", "infra", "infra"], alpha=0.4, classifier="t",
                                         method="mondrian", small_domains="include")
        self.assertEqual((strict.qhats["network"], strict.pooled_domains), (1.0, []))

    def test_top_guess_is_always_on_the_team_and_first(self):
        # the sample-01 situation: network is most likely but below its own high cut-off
        cal = ConformalCalibrator(method="mondrian", alpha=0.05, n=100, classifier="t",
                                  qhats={"network": 0.09, "data": 0.64, "infra": 0.58, "security": 0.5})
        team = cal.prediction_set({"network": 0.83, "data": 0.12, "infra": 0.04, "security": 0.01})
        self.assertEqual(team, ["network"])

    def test_mondrian_protects_the_weak_domain_that_lac_neglects(self):
        cal_data, test = self.biased_world(1, 1600), self.biased_world(2, 4000)
        p, o = [x for x, _ in cal_data], [y for _, y in cal_data]
        lac = ConformalCalibrator.fit(p, o, alpha=0.1, classifier="t", method="lac")
        mondrian = ConformalCalibrator.fit(p, o, alpha=0.1, classifier="t", method="mondrian")

        def coverage(cal, domain):
            cases = [(x, y) for x, y in test if y == domain]
            return sum(y in cal.prediction_set(x) for x, y in cases) / len(cases)

        self.assertLess(coverage(lac, "data"), 0.85)  # the average promise hides the weak domain
        for domain in DOMAINS:
            self.assertGreaterEqual(coverage(mondrian, domain), 0.87, domain)  # >= 0.90 promise, sampling slack

    def test_cheapest_reaching_respects_the_worst_domain(self):
        from oncall.route_eval import cheapest_reaching

        rows = {"cheap": {"owner_on_team": 0.96, "worst_domain": 0.80, "mean_cost": 1.5},
                "fair": {"owner_on_team": 0.97, "worst_domain": 0.96, "mean_cost": 1.9},
                "all": {"owner_on_team": 1.0, "worst_domain": 1.0, "mean_cost": 4.6}}
        picks = {c["target"]: c["strategy"] for c in cheapest_reaching(rows, targets=(0.95, 0.99))}
        self.assertEqual(picks, {0.95: "fair", 0.99: "all"})


class RouterTest(unittest.TestCase):
    def route(self, probs_by_id, report, *down, calibrator=None, policy=None):
        return Router(FixedClassifier(probs_by_id), registry(*down), calibrator, policy).route(report, now=0.0)

    def test_priority_comes_from_the_most_severe_alert(self):
        self.assertEqual(priority_of(incident("i", "IngestJobFailing", "DashboardDown")), "P1")
        self.assertEqual(priority_of(incident("i", "TransformJobFailing")), "P3")

    def test_clear_low_priority_incident_gets_one_cheap_agent(self):
        d = self.route({"i": probs(data=0.95)}, incident("i", "TransformJobFailing"))
        self.assertEqual((d.team, d.confident, d.primary.agent_id, d.consults), (["data"], True, "data-fast", []))
        self.assertIn("data-strong", d.primary.fallbacks)
        self.assertFalse(d.notify_human)

    def test_high_priority_incident_gets_the_strong_agent(self):
        d = self.route({"i": probs(security=0.95)}, incident("i", "LoginFailureSpike"))
        self.assertEqual(d.primary.agent_id, "security-strong")

    def test_unsure_incident_brings_consults(self):
        d = self.route({"i": probs(infra=0.56, data=0.43)}, incident("i", "DashboardSlow"))
        self.assertEqual(d.team, ["infra", "data"])
        self.assertEqual((d.primary.domain, [c.domain for c in d.consults]), ("infra", ["data"]))
        self.assertEqual(d.expected_cost, d.primary.cost_units + d.consults[0].cost_units)

    def test_unsure_p1_incident_notifies_a_human(self):
        d = self.route({"i": probs(infra=0.56, data=0.43)}, incident("i", "DashboardDown"))
        self.assertTrue(d.notify_human)
        self.assertEqual(d.primary.tier, "strong")  # high stakes justify the deeper agent

    def test_down_agent_is_replaced(self):
        d = self.route({"i": probs(data=0.95)}, incident("i", "TransformJobFailing"), "data-fast")
        self.assertEqual(d.primary.agent_id, "data-strong")
        self.assertTrue(any("data-fast" in r and "reported down" in r for r in d.reasons))

    def test_no_agent_left_for_a_domain(self):
        d = self.route({"i": probs(data=0.95)}, incident("i", "TransformJobFailing"),
                       "data-fast", "data-strong", "generalist")
        self.assertIsNone(d.primary)
        self.assertEqual(d.unassigned, ["data"])
        self.assertTrue(d.notify_human)

    def test_generalist_covering_two_domains_is_paid_once(self):
        d = self.route({"i": probs(network=0.5, data=0.45)}, incident("i", "IngestJobFailing"),
                       "network-fast", "network-strong", "data-fast", "data-strong")
        self.assertEqual(d.agents, ["generalist", "generalist"])
        self.assertEqual(d.expected_cost, 6.0)

    def test_calibrated_team_and_baseline_rules(self):
        cal = ConformalCalibrator(alpha=0.1, qhat=0.7, n=100, classifier="fixed")  # keep P >= 0.3
        p = {"i": probs(infra=0.5, data=0.35, network=0.1)}
        report = incident("i", "DashboardSlow")
        self.assertEqual(self.route(p, report, calibrator=cal).team, ["infra", "data"])
        self.assertEqual(self.route(p, report, policy=RoutingPolicy(team_rule="top1")).team, ["infra"])
        self.assertEqual(len(self.route(p, report, policy=RoutingPolicy(team_rule="all")).team), 4)


class CalibrationSplitTest(unittest.TestCase):
    def test_development_split_uses_two_disjoint_halves_of_val(self):
        pairs = [(incident(f"syn-1-{i:05d}", "DashboardSlow"), {"split": s})
                 for i, s in enumerate(["val"] * 40 + ["test"] * 10 + ["train"] * 10)]
        cal, report, label = split_for_calibration(pairs, final=False)
        self.assertEqual(len(cal) + len(report), 40)
        self.assertFalse({r.incident_id for r, _ in cal} & {r.incident_id for r, _ in report})
        self.assertIn("development", label)
        cal, report, label = split_for_calibration(pairs, final=True)
        self.assertEqual((len(cal), len(report)), (40, 10))


if __name__ == "__main__":
    unittest.main()
