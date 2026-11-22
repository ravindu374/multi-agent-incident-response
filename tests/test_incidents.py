import unittest

from oncall.incidents import IncidentDetector
from oncall.oracle import Check, HealthReport


def report(*failing: str) -> HealthReport:
    names = ["freshness", "ingest_job", "transform_job", "api_up"]
    return HealthReport(
        taken_at=0,
        checks=[Check(name=n, ok=n not in failing, detail=f"{n} detail") for n in names],
    )


class DetectorTest(unittest.TestCase):
    def test_quiet_when_healthy(self):
        self.assertEqual(IncidentDetector().observe(report()), [])

    def test_open_update_resolve(self):
        detector = IncidentDetector(resolve_after=2)

        (opened,) = detector.observe(report("ingest_job"))
        self.assertEqual(opened.kind, "opened")
        self.assertEqual(opened.incident.title, "IngestJobFailing")

        self.assertEqual(detector.observe(report("ingest_job")), [])  # nothing new

        (updated,) = detector.observe(report("ingest_job", "freshness"))
        self.assertEqual(updated.kind, "updated")
        self.assertEqual([a.name for a in updated.new_alerts], ["OrdersMartStale"])

        self.assertEqual(detector.observe(report()), [])  # one healthy report is not enough
        (resolved,) = detector.observe(report())
        self.assertEqual(resolved.kind, "resolved")
        self.assertIsNotNone(resolved.incident.resolved_at)
        self.assertIsNone(detector.current)

    def test_flap_resets_the_healthy_streak(self):
        detector = IncidentDetector(resolve_after=2)
        detector.observe(report("api_up"))
        detector.observe(report())
        detector.observe(report("api_up"))
        self.assertEqual(detector.observe(report()), [])
        self.assertIsNotNone(detector.current)

    def test_title_uses_most_user_facing_alert(self):
        (opened,) = IncidentDetector().observe(report("transform_job", "freshness"))
        self.assertEqual(opened.incident.title, "OrdersMartStale")
        self.assertIn("OrdersMartStale: freshness detail", opened.incident.summary)


if __name__ == "__main__":
    unittest.main()
