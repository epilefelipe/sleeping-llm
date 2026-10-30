import unittest
import torch
import torch.nn as nn
from src.coma.hooks import ComaHook


class TestComaHook(unittest.TestCase):

    def test_coma_hook_damping(self):
        linear = nn.Linear(10, 10, bias=False)
        nn.init.eye_(linear.weight)

        hook = ComaHook(damping_factor=0.25)
        hook.register(linear)

        x = torch.ones(1, 5, 10)
        out = linear(x)
        self.assertTrue(torch.allclose(out, torch.ones(1, 5, 10) * 0.25))

        hook.remove()
        out_unhooked = linear(x)
        self.assertTrue(torch.allclose(out_unhooked, torch.ones(1, 5, 10)))

    def test_tuple_output_compatibility(self):
        class DummyLayer(nn.Module):
            def forward(self, x):
                return (x * 2.0, "extra_info")

        layer = DummyLayer()
        hook = ComaHook(damping_factor=0.5)
        hook.register(layer)

        x = torch.ones(2, 4)
        out, extra = layer(x)
        self.assertTrue(torch.allclose(out, torch.ones(2, 4)))
        self.assertEqual(extra, "extra_info")
        hook.remove()


if __name__ == "__main__":
    unittest.main()
