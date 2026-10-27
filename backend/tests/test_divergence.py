"""
Unit Tests for Jensen-Shannon Divergence and Diversity Metrics
"""
import unittest
import math
from backend.services.divergence_service import DivergenceService

class TestDivergenceService(unittest.TestCase):
    def setUp(self):
        self.vocab = ["DevOps", "Kubernetes", "Linux", "News", "Politics"]
        self.service = DivergenceService(vocabulary=self.vocab)

    def test_identical_distributions_zero_jsd(self):
        p = {"DevOps": 0.5, "Kubernetes": 0.5, "Linux": 0.0, "News": 0.0, "Politics": 0.0}
        q = {"DevOps": 0.5, "Kubernetes": 0.5, "Linux": 0.0, "News": 0.0, "Politics": 0.0}
        jsd = self.service.compute_jsd(p, q)
        self.assertAlmostEqual(jsd, 0.0, places=4, msg="Identical distributions must have JSD ~ 0.0")

    def test_orthogonal_distributions_unit_jsd(self):
        # Completely disjoint distributions
        p = {"DevOps": 1.0, "Kubernetes": 0.0, "Linux": 0.0, "News": 0.0, "Politics": 0.0}
        q = {"DevOps": 0.0, "Kubernetes": 0.0, "Linux": 0.0, "News": 1.0, "Politics": 0.0}
        jsd = self.service.compute_jsd(p, q)
        self.assertAlmostEqual(jsd, 1.0, places=3, msg="Completely disjoint distributions must have JSD ~ 1.0")

    def test_jsd_symmetry(self):
        p = {"DevOps": 0.6, "Kubernetes": 0.3, "Linux": 0.1, "News": 0.0, "Politics": 0.0}
        q = {"DevOps": 0.1, "Kubernetes": 0.2, "Linux": 0.2, "News": 0.3, "Politics": 0.2}
        jsd_pq = self.service.compute_jsd(p, q)
        jsd_qp = self.service.compute_jsd(q, p)
        self.assertAlmostEqual(jsd_pq, jsd_qp, places=5, msg="JSD must be symmetric: JSD(P||Q) == JSD(Q||P)")

    def test_boundedness(self):
        p = {"DevOps": 0.4, "Kubernetes": 0.6, "Linux": 0.0, "News": 0.0, "Politics": 0.0}
        q = {"DevOps": 0.2, "Kubernetes": 0.1, "Linux": 0.1, "News": 0.4, "Politics": 0.2}
        jsd = self.service.compute_jsd(p, q)
        self.assertTrue(0.0 <= jsd <= 1.0, f"JSD must be strictly bounded in [0, 1], got {jsd}")

    def test_topic_diversity_entropy(self):
        # Uniform distribution should have maximum normalized entropy (1.0)
        uniform = {t: 0.2 for t in self.vocab}
        h_uniform = self.service.compute_topic_diversity(uniform)
        self.assertAlmostEqual(h_uniform, 1.0, places=2)

        # Concentrated single-topic distribution should have minimum entropy (~0.0)
        concentrated = {t: 0.0 for t in self.vocab}
        concentrated["DevOps"] = 1.0
        h_conc = self.service.compute_topic_diversity(concentrated)
        self.assertAlmostEqual(h_conc, 0.0, places=2)

if __name__ == "__main__":
    unittest.main()
