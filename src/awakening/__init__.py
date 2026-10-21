"""
Awakening and re-ignition protocols for comatose Transformer latent states.
"""

from src.awakening.steering import SteeringVector, ActivationSteeringHook
from src.awakening.criticality import CriticalityLayer

__all__ = ["SteeringVector", "ActivationSteeringHook", "CriticalityLayer"]
