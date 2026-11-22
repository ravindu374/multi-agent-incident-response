import unittest

from oncall.oracle import JobStatus, Snapshot, evaluate
from oncall.settings import Settings

S = Settings()


def healthy_snapshot(**overrides) -> Snapshot:
    base = dict(
        freshness_lag_s=45,
        jobs={
            "ingest": JobStatus(failures_since_success=0, last_success_age_s=10),
            "transform": JobStatus(failures_since_success=0, last_success_age_s=25),
        },
        api_up=True,
        api_latency_p95_s=0.12,
        api_probes=5,
        db_connections=6,
        db_max_connections=40,
        login_failures_60s=0,
    )
    base.update(overrides)
    return Snapshot(**base)


def failing(report):
    return {c.name: c for c in report.failing}


class OracleTest(unittest.TestCase):
    def test_baseline_is_healthy(self):
        report = evaluate(healthy_snapshot(), S)
        self.assertTrue(report.healthy, [c.detail for c in report.failing])

    def test_stale_mart(self):
        report = evaluate(healthy_snapshot(freshness_lag_s=300), S)
        checks = failing(report)
        self.assertEqual(set(checks), {"freshness"})
        self.assertEqual(checks["freshness"].alert, "OrdersMartStale")
        self.assertIn("300s behind", checks["freshness"].detail)

    def test_missing_mart_fails_freshness(self):
        self.assertIn("freshness", failing(evaluate(healthy_snapshot(freshness_lag_s=None), S)))

    def test_one_failed_run_is_tolerated_two_are_not(self):
        jobs = {
            "ingest": JobStatus(failures_since_success=1, last_success_age_s=20, last_error="ReadTimeout: timed out"),
            "transform": JobStatus(last_success_age_s=25),
        }
        self.assertTrue(evaluate(healthy_snapshot(jobs=jobs), S).healthy)
        jobs["ingest"] = JobStatus(failures_since_success=2, last_success_age_s=35, last_error="ReadTimeout: timed out")
        detail = failing(evaluate(healthy_snapshot(jobs=jobs), S))["ingest_job"].detail
        self.assertIn("failed 2 runs in a row", detail)
        self.assertIn("ReadTimeout", detail)

    def test_job_that_cannot_record_failures_still_alerts_by_age(self):
        jobs = {"ingest": JobStatus(last_success_age_s=500), "transform": JobStatus(last_success_age_s=25)}
        self.assertIn("ingest_job", failing(evaluate(healthy_snapshot(jobs=jobs), S)))

    def test_api_down_skips_latency(self):
        checks = failing(evaluate(healthy_snapshot(api_up=False, api_detail="ConnectError: refused"), S))
        self.assertIn("api_up", checks)
        self.assertNotIn("api_latency", checks)

    def test_slow_and_erroring_dashboard(self):
        self.assertIn("api_latency", failing(evaluate(healthy_snapshot(api_latency_p95_s=2.5), S)))
        checks = failing(evaluate(healthy_snapshot(api_errors=3, api_detail="last error: HTTP 503"), S))
        self.assertIn("3/5", checks["api_latency"].detail)

    def test_security_signals(self):
        checks = failing(evaluate(healthy_snapshot(unexpected_clients={"psql": 6}, login_failures_60s=200), S))
        self.assertIn("psql x6", checks["unexpected_db_clients"].detail)
        self.assertEqual(checks["login_failures"].alert, "LoginFailureSpike")

    def test_connection_saturation(self):
        checks = failing(evaluate(healthy_snapshot(db_connections=39), S))
        self.assertIn("39/40", checks["db_connections"].detail)

    def test_collection_errors_raise_monitoring_gap(self):
        checks = failing(evaluate(healthy_snapshot(collection_errors=["warehouse: OperationalError: x"]), S))
        self.assertEqual(checks["monitoring"].alert, "MonitoringGap")


if __name__ == "__main__":
    unittest.main()
