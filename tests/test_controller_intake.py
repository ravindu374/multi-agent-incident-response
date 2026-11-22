"""Block 1 (intake) on incidents shaped like the sandbox's real output."""
import unittest

from controller.contracts import IncidentReport
from controller.intake import extract, mask


def incident(alerts, logs=None, runs=None, **extra) -> IncidentReport:
    record = {
        "incident_id": "inc-test",
        "title": alerts[0][0],
        "alerts": [{"name": n, "detail": d, "check": "x"} for n, d in alerts],
        "evidence": {"logs": logs or {}, "recent_job_runs": runs or []},
        **extra,
    }
    return IncidentReport.from_lab_record(record)


SOURCE_LATENCY = incident(
    [("IngestJobFailing", "ingest failed 2 runs in a row; last error: ReadTimeout: timed out"),
     ("OrdersMartStale", "orders_mart is 145s behind the source (SLO 120s)")],
    logs={
        "ingest": "2026-11-19 10:00:00,123 ingest INFO run ok: 75 rows (batch 41)\n"
                  "2026-11-19 10:00:20,456 ingest ERROR run failed: ReadTimeout: timed out\n"
                  "2026-11-19 10:00:40,789 ingest ERROR run failed: ReadTimeout: timed out\n",
        "transform": "2026-11-19 10:00:30,000 transform INFO dbt build ok in 6.2s\n",
    },
    runs=[{"job": "ingest", "status": "failed", "finished_at": "2026-11-19 10:00:40", "error": "ReadTimeout: timed out"}],
)

TRANSFORM_OOM = incident(
    [("TransformJobFailing", "transform failed 2 runs in a row; last error: dbt build was killed "
                             "(SIGKILL, exit 137); the container may be out of memory")],
    logs={"transform": "2026-11-19 10:01:00,000 transform ERROR dbt build failed in 3.1s: dbt build was killed "
                       "(SIGKILL, exit 137); the container may be out of memory\n"},
)

LOGIN_SPAM = incident(
    [("DashboardSlow", "dashboard p95 latency 2.40s over 5 probes (SLO 1.0s)"),
     ("LoginFailureSpike", "212 failed dashboard logins in the last 60s (limit 30)")],
    logs={"metrics-api": "2026-11-19 10:02:00,001 metrics-api WARNING failed login user=admin client=172.18.0.12\n"
                         "2026-11-19 10:02:00,002 metrics-api WARNING failed login user=admin client=172.18.0.12\n"
                         "2026-11-19 10:02:00,003 metrics-api WARNING failed login user=root client=172.18.0.12\n"},
)

SPOOL_FULL = incident(
    [("IngestJobFailing", "ingest failed 3 runs in a row; last error: OSError: [Errno 28] No space left on device")],
    logs={"ingest": "2026-11-19 10:03:00,000 ingest ERROR run failed: OSError: [Errno 28] No space left on "
                    "device: '/spool/batch-1234.json'\n"},
)


class MaskTest(unittest.TestCase):
    def test_volatile_tokens_are_masked(self):
        self.assertEqual(
            mask("2026-11-19 10:00:20,456 client=172.18.0.12:5432 pid 4711 id deadbeefcafe took 2.5s"),
            "<TS> client=<IP> pid <N> id <HEX> took <N>s",
        )

    def test_identifiers_with_digits_survive(self):
        self.assertEqual(mask("p95 latency, stg_orders, pg16"), "p95 latency, stg_orders, pg16")


class IntakeTest(unittest.TestCase):
    def test_repeated_failures_collapse_to_one_line(self):
        f = extract(SOURCE_LATENCY)
        self.assertEqual(f.error_lines, ["[ingest] ingest ERROR run failed: ReadTimeout: timed out"])
        self.assertEqual(f.alerts, ["IngestJobFailing", "OrdersMartStale"])
        self.assertEqual(f.components, ["ingest", "marts"])
        self.assertGreaterEqual(f.signals["network_timeout"], 1)
        self.assertEqual(f.signals["stale_data"], 1)

    def test_signals_come_from_raw_text_not_masked_text(self):
        self.assertIn("oom_killed", extract(TRANSFORM_OOM).signals)
        f = extract(SPOOL_FULL)
        self.assertIn("disk_full", f.signals)
        self.assertIn("Errno <N>", f.error_lines[0])  # ...while the stored line stays masked

    def test_security_incident(self):
        f = extract(LOGIN_SPAM)
        self.assertEqual(set(f.signals), {"high_latency", "login_failures"})
        self.assertEqual(f.components, ["metrics-api"])
        self.assertEqual(len(f.error_lines), 2)  # admin + root; the duplicate admin line is dropped
        self.assertNotIn("172.18", f.text)

    def test_text_is_masked_and_ordered(self):
        text = extract(SOURCE_LATENCY).text
        self.assertTrue(text.startswith("title: IngestJobFailing\nalerts:\n- IngestJobFailing:"))
        self.assertNotIn("2026-", text)
        self.assertIn("orders_mart is <N>s behind the source (SLO <N>s)", text)

    def test_no_signal_for_a_healthy_looking_report(self):
        report = incident([("OrdersMartStale", "orders_mart is 130s behind the source (SLO 120s)")],
                          logs={"ingest": "2026-11-19 10:00:00,000 ingest INFO run ok: 75 rows (batch 9)\n"})
        f = extract(report)
        self.assertEqual(f.signals, {"stale_data": 1})
        self.assertEqual(f.error_lines, [])


class ContractTest(unittest.TestCase):
    def test_ground_truth_fields_never_enter_the_controller(self):
        report = incident([("DashboardDown", "dashboard API down: ConnectError")],
                          owner="infra", fault_id="I4", valid_fixes=["rollback_service_config"])
        dumped = report.model_dump_json()
        for leaked in ("owner", "fault_id", "valid_fixes", "rollback_service_config", "I4"):
            self.assertNotIn(leaked, dumped)


if __name__ == "__main__":
    unittest.main()
