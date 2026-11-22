"""Block 2b logic on a tiny hand-made problem (milliseconds of CPU; no experiment is run here)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from controller import DOMAINS
from controller.classifier import choose_team
from controller.contracts import IncidentFeatures
from controller.learned import LogRegClassifier, TrainConfig, resolve_backend, structured, tokens

HAS_TORCH = importlib.util.find_spec("torch") is not None
TEXT = {
    "network": "ingest failed: ReadTimeout timed out on the source path",
    "data": "dbt test not_null_stg_orders_amount failed with null amounts",
    "infra": "dbt killed exit 137 out of memory in transform container",
    "security": "failed dashboard logins spike from one client address",
}
SIGNAL = {"network": "network_timeout", "data": "data_test_failed", "infra": "oom_killed",
          "security": "login_failures"}


def toy(domain: str, i: int) -> IncidentFeatures:
    return IncidentFeatures(incident_id=f"{domain}-{i}", alerts=["OrdersMartStale"] if i % 2 else [],
                            signals={SIGNAL[domain]: 1}, components=[], error_lines=[],
                            text=f"{TEXT[domain]} run {i}")


def toy_set(per_domain: int = 6):
    feats = [toy(d, i) for d in DOMAINS for i in range(per_domain)]
    owners = [f.incident_id.split("-")[0] for f in feats]
    involved = [[o] for o in owners]
    return feats, owners, involved


class FeatureTest(unittest.TestCase):
    def test_tokens_add_bigrams_and_drop_stopwords(self):
        self.assertEqual(tokens("The lock timeout <N>"), ["lock", "timeout", "lock__timeout"])

    def test_bigrams_never_cross_lines_or_include_section_headers(self):
        text = "alerts:\n- OrdersMartStale: behind (SLO <N>s)\nevidence:\n- [ingest] disk full"
        toks = tokens(text)
        self.assertNotIn("slo__evidence", toks)
        self.assertNotIn("evidence", toks)
        self.assertIn("disk__full", toks)

    def test_structured_names(self):
        self.assertEqual(structured(toy("infra", 1)), ["sig:oom_killed", "alert:OrdersMartStale"])


class LogRegTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feats, cls.owners, cls.involved = toy_set()
        cls.model = LogRegClassifier(TrainConfig(epochs=200))
        cls.info = cls.model.fit(cls.feats, cls.owners, cls.involved, device="cpu")

    def test_learns_a_separable_problem(self):
        self.assertEqual(self.info["backend"], "numpy")
        self.assertEqual(self.info["train_accuracy"], 1.0)
        unseen = toy("security", 99)
        self.assertEqual(self.model.predict(unseen).ranked()[0][0], "security")

    def test_outputs_are_well_formed_and_plug_into_choose_team(self):
        scores = self.model.predict(toy("data", 42))
        self.assertAlmostEqual(sum(scores.owner_probs.values()), 1.0, places=6)
        self.assertTrue(all(0 <= p <= 1 for p in scores.involved_probs.values()))
        self.assertEqual(scores.method, "logreg")
        self.assertIn("data", choose_team(scores).team)
        self.assertTrue(scores.evidence)  # names the features behind the top domains

    def test_unknown_words_are_ignored(self):
        f = IncidentFeatures(incident_id="x", alerts=[], signals={}, components=[], error_lines=[],
                             text="zebra quantum flux")
        self.assertAlmostEqual(sum(self.model.predict(f).owner_probs.values()), 1.0, places=6)

    def test_save_and_load_give_identical_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            self.model.save(path)
            loaded = LogRegClassifier.load(path)
        for f in self.feats[:4]:
            self.assertEqual(self.model.predict(f).owner_probs, loaded.predict(f).owner_probs)

    def test_top_features_point_at_the_right_words(self):
        names = [name for name, _ in self.model.top_features("infra", 5)]
        self.assertTrue(any("oom" in n or "memory" in n or "137" in n or "killed" in n for n in names), names)

    def test_rejects_bad_labels(self):
        with self.assertRaises(ValueError):
            LogRegClassifier().fit(self.feats[:2], ["network", "storage"], [[], []], device="cpu")


class BackendTest(unittest.TestCase):
    def test_cpu_is_always_numpy(self):
        self.assertEqual(resolve_backend("cpu"), "numpy")

    @unittest.skipIf(HAS_TORCH, "PyTorch is installed")
    def test_cuda_without_torch_explains_how_to_install(self):
        self.assertEqual(resolve_backend("auto"), "numpy")
        with self.assertRaises(RuntimeError) as ctx:
            resolve_backend("cuda")
        self.assertIn("download.pytorch.org", str(ctx.exception))

    @unittest.skipUnless(HAS_TORCH and resolve_backend("auto") == "cuda", "needs PyTorch with a CUDA GPU")
    def test_gpu_and_cpu_train_the_same_model(self):
        feats, owners, involved = toy_set()
        cpu = LogRegClassifier(TrainConfig(epochs=200))
        cpu.fit(feats, owners, involved, device="cpu")
        gpu = LogRegClassifier(TrainConfig(epochs=200))
        gpu.fit(feats, owners, involved, device="cuda")
        for f in feats[:6]:
            a = np.array(list(cpu.predict(f).owner_probs.values()))
            b = np.array(list(gpu.predict(f).owner_probs.values()))
            self.assertLess(float(np.abs(a - b).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
