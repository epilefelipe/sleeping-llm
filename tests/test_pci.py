import unittest
import numpy as np
from src.metrics.pci import compute_casali_pci, compute_evoked_response, lempel_ziv_76


class TestCasaliPCI(unittest.TestCase):

    def test_lz76_complexity(self):
        # Uniform pattern
        c_ones = lempel_ziv_76("1111111111")
        self.assertLess(c_ones, 5)

        # Alternating pattern
        c_alt = lempel_ziv_76("1010101010")
        self.assertGreaterEqual(c_alt, 2)

    def test_evoked_response_isolation(self):
        shape = (12, 20)
        base = np.zeros(shape)
        pert = np.ones(shape) * 2.0
        delta = compute_evoked_response(base, pert)
        self.assertEqual(delta.shape, (12, 20))
        self.assertAlmostEqual(delta[0, 0], 2.0, places=3)

    def test_pci_zero_perturbation(self):
        evoked = np.zeros((12, 30))
        noise = np.ones((12, 30))
        pci = compute_casali_pci(evoked, baseline_noise=noise, significance_percentile=99.0)
        self.assertEqual(pci, 0.0)

    def test_pci_high_complexity(self):
        np.random.seed(42)
        # 50% exceeding threshold, highly differentiated spatial-temporal pattern
        evoked = np.random.choice([0.0, 10.0], size=(12, 40))
        noise = np.full((12, 40), 1.0)
        pci = compute_casali_pci(evoked, baseline_noise=noise, significance_percentile=99.0)
        self.assertGreater(pci, 0.5)
        self.assertLessEqual(pci, 1.0)


if __name__ == "__main__":
    unittest.main()
