"""
Perturbational Complexity Index (PCI) for Deep Neural Networks.
Adapts the Casali & Massimini (2013) complexity metric to Transformer latent spaces.
"""

import numpy as np


def lempel_ziv_complexity(binary_sequence: str) -> int:
    """
    Computes the Lempel-Ziv (LZ76) complexity of a binary sequence.
    Counts the number of distinct non-overlapping substrings.
    """
    n = len(binary_sequence)
    if n == 0:
        return 0

    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1

    while l + k <= n:
        if binary_sequence[i + k - 1] == binary_sequence[l + k - 1]:
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


def compute_pci(activation_matrix: np.ndarray, threshold_std: float = 1.0) -> float:
    """
    Computes the Perturbational Complexity Index (PCI_LLM) for an activation matrix.

    Args:
        activation_matrix: 2D array of shape (num_layers, hidden_dimension) or (timesteps, channels).
        threshold_std: Number of standard deviations above baseline to set binary threshold.

    Returns:
        pci: Normalized Perturbational Complexity Index in range [0, 1].
    """
    # 1. Binarize activation matrix relative to mean baseline
    mean_val = np.mean(activation_matrix)
    std_val = np.std(activation_matrix)
    threshold = mean_val + threshold_std * std_val

    binary_mat = (activation_matrix > threshold).astype(int)

    # 2. Serialize matrix into a 1D bitstream
    bitstream = "".join(binary_mat.flatten().astype(str))
    total_length = len(bitstream)

    if total_length == 0:
        return 0.0

    # 3. Compute empirical Lempel-Ziv complexity
    c = lempel_ziv_complexity(bitstream)

    # 4. Normalize by source entropy / asymptotic upper bound
    p1 = np.mean(binary_mat)
    p0 = 1.0 - p1

    if p0 == 0 or p1 == 0:
        return 0.0

    # Shannon source entropy
    entropy = - (p1 * np.log2(p1) + p0 * np.log2(p0))
    asymptotic_bound = (total_length * entropy) / np.log2(total_length)

    if asymptotic_bound == 0:
        return 0.0

    pci = c / asymptotic_bound
    return float(np.clip(pci, 0.0, 1.0))
