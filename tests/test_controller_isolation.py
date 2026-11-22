"""Information barrier: the controller package must never import the lab (which holds ground truth)."""
import re
import unittest

from oncall.settings import ROOT

LAB_IMPORT = re.compile(r"^\s*(from|import)\s+oncall\b", re.MULTILINE)


class IsolationTest(unittest.TestCase):
    def test_controller_and_agents_do_not_import_the_lab(self):
        for package in ("controller", "agents"):
            with self.subTest(package=package):
                offenders = [p.name for p in (ROOT / package).rglob("*.py") if LAB_IMPORT.search(p.read_text("utf-8"))]
                self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
