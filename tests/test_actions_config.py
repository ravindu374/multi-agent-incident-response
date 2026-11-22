"""Actions that only edit runtime config, exercised against a temp copy of runtime/defaults."""
import shutil
import tempfile
import unittest
from pathlib import Path

from oncall.actions import ActionError, execute
from oncall.lab import Lab
from oncall.settings import ROOT, Settings


class ConfigActionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(ROOT / "runtime" / "defaults", self.tmp / "runtime" / "defaults")
        self.lab = Lab(Settings(runtime_dir=self.tmp / "runtime", lab_dir=self.tmp / "lab"))
        self.lab.runtime.init()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_switch_endpoint_and_audit_log(self):
        execute(self.lab, "switch_source_endpoint", {"endpoint": "secondary"}, actor="test")
        self.assertEqual(self.lab.runtime.load("ingest")["source"]["active"], "secondary")
        (entry,) = self.lab.ledger.read("actions")
        self.assertEqual((entry["action"], entry["actor"], entry["risk"]), ("switch_source_endpoint", "test", "low"))

    def test_switch_endpoint_rejects_unknown_path(self):
        with self.assertRaises(ActionError):
            execute(self.lab, "switch_warehouse_endpoint", {"endpoint": "tertiary"}, actor="test")

    def test_field_mapping_guard(self):
        execute(self.lab, "set_field_mapping", {"mapping": {"total_amount": "amount"}}, actor="test")
        self.assertEqual(self.lab.runtime.load("ingest")["field_map"], {"total_amount": "amount"})
        with self.assertRaises(ActionError):
            execute(self.lab, "set_field_mapping", {"mapping": {"x": "password"}}, actor="test")

    def test_block_ip_is_idempotent_and_validated(self):
        for _ in range(2):
            execute(self.lab, "block_ip", {"ip": "172.18.0.9"}, actor="test")
        self.assertEqual(self.lab.runtime.load("metrics_api")["blocked_ips"], ["172.18.0.9"])
        with self.assertRaises(ActionError):
            execute(self.lab, "block_ip", {"ip": "not-an-ip"}, actor="test")

    def test_rollback_restores_defaults(self):
        self.lab.runtime.update("ingest", lambda c: c["source"]["endpoints"].__setitem__("primary", "http://bad"))
        execute(self.lab, "rollback_service_config", {"service": "ingest"}, actor="test")
        self.assertEqual(self.lab.runtime.load("ingest"), self.lab.runtime.default("ingest"))

    def test_unknown_action(self):
        with self.assertRaises(ActionError):
            execute(self.lab, "rm_rf", {}, actor="test")


if __name__ == "__main__":
    unittest.main()
