"""
Information-theoretic, complexity, and criticality metrics for Project Sleeping LLM.
"""

from src.metrics.pci import compute_casali_pci, compute_pci, compute_evoked_response
from src.metrics.criticality import (
    compute_layer_branching_ratios,
    compute_temporal_branching_ratio,
    compute_criticality_index,
    classify_dynamical_regime,
    compute_participation_ratio
)

__all__ = [
    "compute_casali_pci",
    "compute_pci",
    "compute_evoked_response",
    "compute_layer_branching_ratios",
    "compute_temporal_branching_ratio",
    "compute_criticality_index",
    "classify_dynamical_regime",
    "compute_participation_ratio"
]
