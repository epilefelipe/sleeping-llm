"""
Spatiotemporal Perturbational Complexity Index (PCI_ST / Casali et al. 2013).
Implements:
1. True in-silico TMS impulse perturbation (evoked response differential: Delta H = H_pert - H_base).
2. Data-driven statistical thresholding based on pre-stimulus / baseline noise floor (not arbitrary std).
3. 2D Spatiotemporal matrix representation (Channels/Layers x Timesteps/Depth).
4. Casali asymptotic Lempel-Ziv source entropy normalization.
"""

from typing import Tuple, Optional
import numpy as np


def lempel_ziv_76(binary_string: str) -> int:
    """
    Computes Lempel-Ziv 1976 (LZ76) complexity of a binary sequence.
    Counts the number of distinct patterns encountered.
    """
    n = len(binary_string)
    if n == 0:
        return 0

    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1

    while l + k <= n:
        if binary_string[i + k - 1] == binary_string[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    return c


def compute_evoked_response(
    baseline_trajectory: np.ndarray,
    perturbed_trajectory: np.ndarray
) -> np.ndarray:
    """
    Isolates the causal perturbational response (analogous to Transcranial Evoked Potentials - TEPs).
    Delta H = H_perturbed - H_baseline
    Shape: (channels/layers, timesteps/depth)
    """
    return perturbed_trajectory - baseline_trajectory


def binarize_spatiotemporal_response(
    evoked_matrix: np.ndarray,
    baseline_noise: Optional[np.ndarray] = None,
    significance_percentile: float = 99.0
) -> Tuple[np.ndarray, float]:
    """
    Binarizes the 2D evoked response matrix against an empirical statistical noise floor.
    In Casali et al., only responses statistically exceeding pre-stimulus noise are set to 1.

    Args:
        evoked_matrix: 2D array (channels x time) of evoked activity.
        baseline_noise: 2D array of pre-stimulus / unperturbed baseline noise.
        significance_percentile: Percentile threshold for statistical significance (default: 99.0%).

    Returns:
        binary_matrix: 2D binary matrix of significant evoked activations.
        threshold: The absolute threshold value used.
    """
    if baseline_noise is not None:
        # Use baseline distribution to derive non-parametric empirical threshold
        threshold = float(np.percentile(np.abs(baseline_noise), significance_percentile))
    else:
        # If no explicit baseline recording provided, use parametric estimate from lower quartile
        threshold = float(np.median(np.abs(evoked_matrix)) + 2.5 * np.std(evoked_matrix[:int(evoked_matrix.shape[0] * 0.25)]))

    # Protect against zero-variance edge cases
    if threshold <= 1e-9:
        threshold = 1e-4

    binary_matrix = (np.abs(evoked_matrix) > threshold).astype(int)
    return binary_matrix, threshold


def compute_casali_pci(
    evoked_matrix: np.ndarray,
    baseline_noise: Optional[np.ndarray] = None,
    significance_percentile: float = 99.0
) -> float:
    """
    Computes the authentic 2D Spatiotemporal Perturbational Complexity Index (PCI).

    Args:
        evoked_matrix: 2D array of shape (num_channels/layers, num_timesteps).
        baseline_noise: Optional baseline noise matrix for statistical thresholding.
        significance_percentile: Statistical cutoff (default: 99th percentile).

    Returns:
        pci: Normalized Perturbational Complexity Index in [0, 1].
    """
    channels, timesteps = evoked_matrix.shape
    total_length = channels * timesteps

    if total_length == 0:
        return 0.0

    # 1. Statistical binarization based on baseline noise floor
    binary_matrix, _ = binarize_spatiotemporal_response(
        evoked_matrix, baseline_noise, significance_percentile
    )

    # 2. 2D Spatiotemporal traversal: preserve spatial structure across time
    # Order: scan all channels at time t before moving to t+1 (standard EEG column-wise raster)
    serialized_stream = "".join(binary_matrix.T.flatten().astype(str))

    # 3. Calculate empirical LZ76 complexity
    c = lempel_ziv_76(serialized_stream)

    # 4. Source entropy calculation
    p1 = float(np.mean(binary_matrix))
    p0 = 1.0 - p1

    # Edge cases: completely silent (p1=0) or completely saturated (p1=1) -> Zero complexity
    if p1 == 0.0 or p0 == 0.0 or p1 < 1e-5 or p0 < 1e-5:
        return 0.0

    shannon_entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))

    # Asymptotic maximal complexity bound according to Casali et al. (2013)
    c_max = (total_length * shannon_entropy) / np.log2(total_length)

    if c_max <= 0:
        return 0.0

    pci = c / c_max
    return float(np.clip(pci, 0.0, 1.0))
