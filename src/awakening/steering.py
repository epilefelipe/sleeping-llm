"""
Pillar 4: Representation Steering and In-Silico Deep Brain Stimulation (DBS).
Implements directional activation steering vectors and chaotic entropy injection
to trigger cognitive re-ignition in comatose Transformer hidden states.
"""

from typing import Optional, Union, List, Dict, Any
import numpy as np


class SteeringVector:
    """
    Representation Steering Vector computed from contrastive state representations.
    Inspired by Representation Engineering (Zou et al., 2023).
    """

    def __init__(self, vector: np.ndarray, layer_idx: int, multiplier: float = 1.0):
        """
        Args:
            vector: 1D or 2D array representing the directional vector in activation space.
            layer_idx: The target Transformer layer where the vector will be injected.
            multiplier: Scaling factor (stimulation amplitude alpha).
        """
        self.raw_vector = vector
        self.layer_idx = layer_idx
        self.multiplier = multiplier

        # Compute normalized unit vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            self.unit_vector = vector / norm
        else:
            self.unit_vector = vector

    @classmethod
    def compute_from_states(
        cls,
        awake_activations: np.ndarray,
        comatose_activations: np.ndarray,
        layer_idx: int,
        multiplier: float = 1.0
    ) -> "SteeringVector":
        """
        Computes difference-in-means steering vector:
        v = E[h_awake] - E[h_comatose]
        """
        mean_awake = np.mean(awake_activations, axis=0)
        mean_coma = np.mean(comatose_activations, axis=0)
        diff_vector = mean_awake - mean_coma
        return cls(vector=diff_vector, layer_idx=layer_idx, multiplier=multiplier)

    def get_injection_tensor(self, chaotic_entropy: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the perturbation vector ready for addition:
        v_inject = multiplier * unit_vector (optionally modulated by chaotic entropy).
        """
        base = self.multiplier * self.unit_vector

        if chaotic_entropy is not None:
            # The LavaRand Principle: modulate with non-repeating aperiodic physical noise
            modulation = 1.0 + chaotic_entropy
            return base * modulation

        return base


class ActivationSteeringHook:
    """
    Forward hook for injecting steering vectors directly into PyTorch layer outputs.
    Acts as an in-silico Deep Brain Stimulation (DBS) electrode.
    """

    def __init__(self, steering_vector: SteeringVector, chaotic_alpha: float = 0.0):
        self.steering_vector = steering_vector
        self.chaotic_alpha = chaotic_alpha
        self.hook_handle = None

    def __call__(self, module, input_tensor, output_tensor):
        import torch

        # Handle tuple output (hidden_states, attention_weights)
        if isinstance(output_tensor, tuple):
            hidden_states = output_tensor[0]
            rest = output_tensor[1:]
        else:
            hidden_states = output_tensor
            rest = None

        # Convert steering vector to torch tensor on the same device/dtype
        v_np = self.steering_vector.get_injection_tensor()
        v_torch = torch.tensor(v_np, dtype=hidden_states.dtype, device=hidden_states.device)

        # Apply chaotic entropy modulation if enabled (LavaRand principle)
        if self.chaotic_alpha > 0.0:
            chaos = torch.randn_like(hidden_states) * self.chaotic_alpha
            steered_states = hidden_states + v_torch * (1.0 + chaos)
        else:
            steered_states = hidden_states + v_torch

        if rest is not None:
            return (steered_states,) + rest
        return steered_states

    def register(self, layer_module):
        self.hook_handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
