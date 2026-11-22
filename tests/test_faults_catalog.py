import unittest

from oncall.actions import ACTIONS, DOMAINS
from oncall.faults import FAULTS

GENERIC_TAGS = {"data_stale"}


class FaultCatalogTest(unittest.TestCase):
    def test_every_domain_owns_at_least_one_fault(self):
        self.assertEqual({f.owner for f in FAULTS.values()}, set(DOMAINS))

    def test_metadata_is_consistent(self):
        for fault_id, fault in FAULTS.items():
            with self.subTest(fault_id):
                self.assertEqual(fault.id, fault_id)
                self.assertIn(fault.owner, DOMAINS)
                self.assertIn(fault.owner, fault.involved)
                self.assertLessEqual(set(fault.involved), set(DOMAINS))
                self.assertTrue(fault.valid_fixes)
                self.assertTrue(fault.symptom_tags)

    def test_valid_fixes_are_allowlisted_for_the_owner(self):
        for fault in FAULTS.values():
            for name in fault.valid_fixes:
                with self.subTest(fault=fault.id, action=name):
                    self.assertIn(name, ACTIONS)
                    self.assertIn(fault.owner, ACTIONS[name].domains)

    def test_static_reference_fixes_use_valid_fixes(self):
        for fault in FAULTS.values():
            if fault.fix_needs_lookup:
                continue
            with self.subTest(fault=fault.id):
                calls = fault.reference_fix(lab=None, details={})
                self.assertTrue(calls)
                for name, params in calls:
                    self.assertIn(name, fault.valid_fixes)
                    self.assertIsInstance(params, dict)

    def test_symptoms_alone_never_identify_the_owner(self):
        """Each fault shares a specific symptom with a fault owned by another domain."""
        for fault in FAULTS.values():
            tags = set(fault.symptom_tags) - GENERIC_TAGS
            peers = [g.id for g in FAULTS.values()
                     if g.owner != fault.owner and tags & set(g.symptom_tags)]
            with self.subTest(fault=fault.id):
                self.assertTrue(peers, f"{fault.id} is identifiable from its symptoms alone")


if __name__ == "__main__":
    unittest.main()
