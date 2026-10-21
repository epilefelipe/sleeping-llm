"""
Criticality Metrics for Neural Networks and In-Silico Disorders of Consciousness.

Pillar 3: Re-ignition Physics & Criticality.

Theoretical Foundation:
The Critical Brain Hypothesis (Beggs & Plenz 2003, Chialvo 2010, Tagliazucchi 2016) posits that
conscious, adaptive neural systems operate near a non-equilibrium phase transition ("the edge of chaos").
- Subcritical regime (sigma < 1.0): Perturbations rapidly decay; system is rigid, frozen, comatose.
- Critical regime (sigma == 1.0): Perturbations propagate with optimal dynamic range and maximal integration.
- Supercritical regime (sigma > 1.0): Perturbations amplify uncontrollably into epileptic seizure-like states.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def compute_layer_branching_ratios(layer_norms: np.ndarray) -> np.ndarray:
    """
    Computes the layer-to-layer branching parameter sigma_l = ||Delta h_{l+1}|| / ||Delta h_l||.

    Args:
        layer_norms: 1D array of perturbation norms for each layer (shape: (num_layers,)).

    Returns:
        ratios: 1D array of branching ratios of length (num_layers - 1).
    """
    layer_norms = np.asarray(layer_norms, dtype=float)
    if len(layer_norms) < 2:
        return np.array([1.0])

    ratios = []
    for l in range(len(layer_norms) - 1):
        prev = layer_norms[l]
        curr = layer_norms[l + 1]
        if prev > 1e-7:
            ratios.append(curr / prev)
        else:
            ratios.append(1.0 if curr <= 1e-7 else 10.0)

    return np.array(ratios)


def compute_temporal_branching_ratio(binary_evoked_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the classic Beggs & Plenz (2003) temporal branching ratio across timesteps:
    sigma(t) = A(t+1) / A(t), where A(t) is the count of active units/layers at time t.

    Args:
        binary_evoked_matrix: 2D binary matrix (layers x timesteps).

    Returns:
        mean_sigma: Average branching parameter across active timesteps.
        temporal_sigmas: Array of instantaneous branching ratios.
    """
    active_per_time = np.sum(binary_evoked_matrix, axis=0)
    sigmas = []

    for t in range(len(active_per_time) - 1):
        a_t = active_per_time[t]
        a_next = active_per_time[t + 1]
        if a_t > 0:
            sigmas.append(a_next / a_t)

    if len(sigmas) == 0:
        return 0.0, np.array([])

    mean_sigma = float(np.mean(sigmas))
    return mean_sigma, np.array(sigmas)


def compute_criticality_index(branching_ratios: np.ndarray) -> Tuple[float, float, float]:
    """
    Quantifies proximity to criticality and dynamical stability across layers.
    Returns:
        kappa: Distance of mean branching ratio from 1.0, in (0, 1].
        mean_sigma: The mean branching ratio.
        stability: Critical stability score exp(-10 * variance(sigma)) in (0, 1].
                   High stability (~ 1.0) indicates smooth critical propagation.
                   Low stability indicates shock/volatility at lesion boundaries.
    """
    branching_ratios = np.asarray(branching_ratios, dtype=float)
    if len(branching_ratios) == 0:
        return 0.0, 0.0, 0.0

    mean_sigma = float(np.mean(branching_ratios))
    var_sigma = float(np.var(branching_ratios))

    kappa = float(np.exp(-abs(mean_sigma - 1.0)))
    stability = float(np.exp(-10.0 * var_sigma))

    return kappa, mean_sigma, stability


def classify_dynamical_regime(branching_ratio: float, tolerance: float = 0.15) -> str:
    """
    Classifies the dynamical phase state of the network.

    Args:
        branching_ratio: The observed branching parameter sigma.
        tolerance: Symmetrical boundary around 1.0 for criticality (default: 0.15).

    Returns:
        Regime label: 'subcritical (frozen/coma)', 'critical (edge-of-chaos)', or 'supercritical (seizure)'.
    """
    if branching_ratio < (1.0 - tolerance):
        return "Subcritical (Frozen / Comatose Attractor)"
    elif branching_ratio > (1.0 + tolerance):
        return "Supercritical (Runaway / Seizure-like Attractor)"
    else:
        return "Critical (Edge of Chaos / Conscious Operating Point)"


def compute_participation_ratio(hidden_matrix: np.ndarray) -> float:
    """
    Computes the Participation Ratio (PR) of covariance eigenvalues.
    PR measures the effective dimensionality of neural representations:
        PR = (sum lambda_i)^2 / sum (lambda_i^2)
    High PR indicates rich, distributed, scale-free integration (criticality).
    Low PR indicates low-rank collapse into degenerate attractors (coma).
    """
    if hidden_matrix.ndim != 2:
        hidden_matrix = hidden_matrix.reshape(hidden_matrix.shape[0], -1)

    centered = hidden_matrix - np.mean(hidden_matrix, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)

    try:
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvals = eigenvals[eigenvals > 1e-10]
        if len(eigenvals) == 0:
            return 1.0
        sum_e = np.sum(eigenvals)
        sum_sq = np.sum(eigenvals ** 2)
        pr = (sum_e ** 2) / sum_sq
        return float(pr)
    except np.linalg.LinAlgError:
        return 1.0
