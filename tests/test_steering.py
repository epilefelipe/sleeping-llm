import unittest
import torch
import torch.nn as nn
import numpy as np
from src.awakening.steering import SteeringVector, ActivationSteeringHook
from src.awakening.criticality import CriticalityLayer


class TestSteeringAndCriticality(unittest.TestCase):

    def test_steering_vector_injection(self):
        v = np.array([1.0, 0.0, 0.0, 0.0])
        svec = SteeringVector(vector=v, layer_idx=2, multiplier=2.0)
        hook = ActivationSteeringHook(steering_vector=svec, chaotic_alpha=0.0)

        layer = nn.Identity()
        hook.register(layer)

        x = torch.zeros(1, 4, 4)
        out = layer(x)
        expected = torch.tensor([2.0, 0.0, 0.0, 0.0]).expand(1, 4, 4)
        self.assertTrue(torch.allclose(out, expected))
        hook.remove()

    def test_criticality_layer_gain(self):
        crit = CriticalityLayer(
            hidden_dim=16,
            homeostatic_gain=1.5,
            stochastic_resonance_std=0.0,
            adaptive=False
        )
        layer = nn.Identity()
        crit.register(layer)

        x = torch.ones(1, 3, 16)
        out = layer(x)
        self.assertTrue(torch.allclose(out, torch.ones(1, 3, 16) * 1.5))
        crit.remove()


if __name__ == "__main__":
    unittest.main()
