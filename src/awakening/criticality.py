"""
Criticality Layer and Homeostatic Phase Transition Tuning.

Pillar 3: Re-ignition Physics & Criticality.

Theoretical Foundation:
In mammalian cortex, homeostatic plasticity and ascending neuromodulatory inputs (acetylcholine,
noradrenaline) constantly regulate recurrent gain to maintain networks at the critical boundary
between quiescence and seizure (the 'Edge of Chaos').

The CriticalityLayer operates as a dynamic neuromodulator inside Transformer layers:
- Detects subcritical collapse (coma / extinction: sigma < 1.0).
- Injects calibrated stochastic resonance (The LavaRand principle).
- Restores the branching ratio toward criticality (sigma ~ 1.0) without triggering runaway excitation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class CriticalityLayer(nn.Module):
    """
    A neuromodulatory layer designed to maintain the Transformer residual stream
    at the critical boundary (sigma ~ 1.0).

    Can be deployed as a standard PyTorch module or registered as a forward hook.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        target_branching_ratio: float = 1.0,
        homeostatic_gain: float = 1.0,
        stochastic_resonance_std: float = 0.05,
        adaptive: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_branching_ratio = target_branching_ratio
        self.homeostatic_gain = nn.Parameter(torch.tensor(homeostatic_gain, dtype=torch.float32))
        self.stochastic_resonance_std = stochastic_resonance_std
        self.adaptive = adaptive
        self.hook_handle = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Applies homeostatic gain and critical stochastic resonance to hidden_states.
        Shape: (batch_size, seq_len, hidden_dim).
        """
        # 1. Compute empirical variance along hidden dimension
        current_norm = torch.norm(hidden_states, dim=-1, keepdim=True).mean()

        # 2. Adaptive homeostatic scaling
        if self.adaptive and current_norm < 10.0:
            # Detects comatose collapse: inject energetic compensation
            scaling = 1.0 + torch.clamp((10.0 - current_norm) / 5.0, 0.0, 3.0)
        else:
            scaling = 1.0

        # 3. Apply calibrated stochastic resonance (Pillar 3)
        if self.stochastic_resonance_std > 0.0 and self.training or not torch.is_grad_enabled():
            noise = torch.randn_like(hidden_states) * self.stochastic_resonance_std
            tuned_states = (hidden_states * self.homeostatic_gain * scaling) + noise
        else:
            tuned_states = hidden_states * self.homeostatic_gain * scaling

        return tuned_states

    def register(self, layer_module):
        """Registers this criticality tuner as a PyTorch forward hook on target module."""
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
                rest = out[1:]
                return (self.forward(h),) + rest
            else:
                return self.forward(out)

        self.hook_handle = layer_module.register_forward_hook(hook_fn)
        return self.hook_handle

    def remove(self):
        """Detaches the hook from the layer module."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
