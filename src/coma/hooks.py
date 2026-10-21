"""
Coma induction protocols using PyTorch forward hooks.
"""

from typing import Callable, Optional


class ComaHook:
    """
    Hook to attenuate or disrupt activations in intermediate Transformer layers.
    Simulates thalamocortical decoupling or pharmacological sedation.
    """

    def __init__(self, damping_factor: float = 0.0, noise_std: float = 0.0):
        """
        Args:
            damping_factor: Scale applied to activations (0.0 = complete severance/silence).
            noise_std: Standard deviation of Gaussian noise added to activations.
        """
        self.damping_factor = damping_factor
        self.noise_std = noise_std
        self.hook_handle = None

    def __call__(self, module, input_tensor, output_tensor):
        # If output is a tuple (hidden_states, attention_weights), modify hidden_states
        if isinstance(output_tensor, tuple):
            hidden_states = output_tensor[0]
            rest = output_tensor[1:]
        else:
            hidden_states = output_tensor
            rest = None

        # Apply damping and noise
        modified_states = hidden_states * self.damping_factor
        if self.noise_std > 0.0:
            import torch
            noise = torch.randn_like(modified_states) * self.noise_std
            modified_states = modified_states + noise

        if rest is not None:
            return (modified_states,) + rest
        return modified_states

    def register(self, layer_module):
        self.hook_handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
