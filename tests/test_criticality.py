import unittest
import numpy as np
from src.metrics.criticality import (
    compute_layer_branching_ratios,
    compute_criticality_index,
    classify_dynamical_regime
)


class TestCriticalityMetrics(unittest.TestCase):

    def test_branching_ratios(self):
        norms = [10.0, 10.0, 10.0, 10.0]
        ratios = compute_layer_branching_ratios(norms)
        self.assertEqual(len(ratios), 3)
        for r in ratios:
            self.assertAlmostEqual(r, 1.0, places=4)

    def test_criticality_index_ideal(self):
        ratios = [1.0, 1.0, 1.0]
        kappa, mean_sigma, stability = compute_criticality_index(ratios)
        self.assertAlmostEqual(kappa, 1.0, places=4)
        self.assertAlmostEqual(mean_sigma, 1.0, places=4)
        self.assertAlmostEqual(stability, 1.0, places=4)

    def test_regime_classification(self):
        self.assertEqual(classify_dynamical_regime(0.5), "Subcritical (Frozen / Comatose Attractor)")
        self.assertEqual(classify_dynamical_regime(1.0), "Critical (Edge of Chaos / Conscious Operating Point)")
        self.assertEqual(classify_dynamical_regime(1.8), "Supercritical (Runaway / Seizure-like Attractor)")


if __name__ == "__main__":
    unittest.main()
