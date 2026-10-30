import unittest
from src.epistemology.beauty_experiment import SleepingBeautyExperiment


class TestSleepingBeauty(unittest.TestCase):

    def test_simulation_protocol(self):
        exp = SleepingBeautyExperiment()
        res = exp.simulate_protocol(num_trials=200)

        self.assertEqual(res["trials"], 200)
        self.assertGreater(res["total_awakenings"], 200)
        # Expected thirder ratio ~0.333
        self.assertAlmostEqual(res["empirical_credence_heads"], 0.333, delta=0.08)
        self.assertEqual(res["theoretical_halfer"], 0.50)
        self.assertAlmostEqual(res["theoretical_thirder"], 0.3333, places=3)


if __name__ == "__main__":
    unittest.main()
