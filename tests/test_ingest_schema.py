"""The ingest job's schema contract (the mechanism behind fault D1)."""
import importlib.util
import unittest

from oncall.settings import ROOT

_spec = importlib.util.spec_from_file_location("ingest_schema", ROOT / "sandbox" / "ingest" / "schema.py")
schema = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(schema)

ROW = {"order_id": 1, "created_at": "2026-11-19T00:00:00+00:00", "customer_id": 7, "store_id": 3,
       "amount": 12.5, "status": "placed"}


class IngestSchemaTest(unittest.TestCase):
    def test_passes_through_valid_rows(self):
        self.assertEqual(schema.normalise([ROW], {}), [ROW])

    def test_drifted_payload_is_rejected_with_a_useful_message(self):
        drifted = {**{k: v for k, v in ROW.items() if k != "amount"}, "total_amount": 12.5}
        with self.assertRaises(schema.SchemaMismatch) as ctx:
            schema.normalise([drifted], {})
        self.assertIn("['amount']", str(ctx.exception))
        self.assertIn("total_amount", str(ctx.exception))

    def test_field_map_repairs_drift(self):
        drifted = {**{k: v for k, v in ROW.items() if k != "amount"}, "total_amount": 12.5}
        self.assertEqual(schema.normalise([drifted], {"total_amount": "amount"}), [ROW])


if __name__ == "__main__":
    unittest.main()
