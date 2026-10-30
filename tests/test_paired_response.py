import unittest

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.coma.hooks import ComaHook
from src.metrics.evoked import paired_trajectories


class TestPairedResponse(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.model = GPT2LMHeadModel(GPT2Config(
            n_layer=2, n_head=2, n_embd=32, vocab_size=31, n_positions=16,
            bos_token_id=0, eos_token_id=1
        )).eval()
        self.inputs = {'input_ids': torch.tensor([[1, 4, 2, 8, 9, 3, 6, 5]])}
        self.layer = self.model.transformer.h[0]

    def measure(self, amplitude):
        return paired_trajectories(
            self.model, self.inputs, self.layer, 3, amplitude, 17
        )

    def test_zero_impulse_under_damping(self):
        intact, _ = self.measure(0.0)
        hook = ComaHook(damping_factor=0.2)
        hook.register(self.model.transformer.h[1])
        baseline, perturbed = self.measure(0.0)
        hook.remove()
        np.testing.assert_array_equal(baseline, perturbed)
        self.assertGreater(np.linalg.norm(perturbed - intact), 0.0)

    def test_pulse_is_causal_and_removed(self):
        self.measure(0.0)
        existing_hooks = set(self.layer._forward_hooks)
        baseline, perturbed = self.measure(1.0)
        np.testing.assert_array_equal(baseline[:, :3], perturbed[:, :3])
        self.assertGreater(np.linalg.norm(perturbed[:, 3:] - baseline[:, 3:]), 0.0)
        restored, _ = self.measure(0.0)
        np.testing.assert_array_equal(baseline, restored)
        self.assertEqual(set(self.layer._forward_hooks), existing_hooks)

    def test_common_random_numbers_and_rng_restoration(self):
        hook = ComaHook(damping_factor=0.5, noise_std=0.1)
        hook.register(self.model.transformer.h[1])
        state = torch.random.get_rng_state().clone()
        baseline, perturbed = self.measure(0.0)
        hook.remove()
        np.testing.assert_array_equal(baseline, perturbed)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
