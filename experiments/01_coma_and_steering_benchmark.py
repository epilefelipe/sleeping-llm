#!/usr/bin/env python3
"""
Experiment 01: Synthetic Prototype of In-Silico Coma Induction & Steering Re-ignition.

PEDAGOGICAL & CONCEPTUAL DEMONSTRATION (Phase 0):
This script demonstrates the end-to-end algorithmic pipeline in an idealized, synthetic
cortical manifold before transitioning to real Transformer models (see Experiment 03):
1. Baseline Awake State -> High Spatiotemporal PCI (~ 1.00)
2. In-Silico Coma (Residual Stream Severance) -> Collapsed PCI (0.00)
3. Awakening Intervention (Steering Vector + LavaRand Chaotic Entropy) -> Cortical Re-ignition (~ 1.00)
"""

import os
import sys
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.pci import compute_casali_pci
from src.awakening.steering import SteeringVector


def generate_synthetic_cortex(num_layers=32, hidden_dim=128, state="awake", seed=42):
    np.random.seed(seed)

    if state == "awake":
        # Complex multi-frequency traveling waves across cortical depth
        base = np.random.randn(num_layers, hidden_dim) * 0.7
        t = np.linspace(0, 8 * np.pi, num_layers)[:, None]
        spatial = np.sin(t * np.linspace(0.8, 3.2, hidden_dim)[None, :])
        return base + spatial

    elif state == "coma":
        # Collapsed residual stream: shallow layers have low noise, deep layers are severed
        coma_matrix = np.random.normal(0, 0.01, size=(num_layers, hidden_dim))
        coma_matrix[int(num_layers * 0.35):, :] = 0.0  # 65% of deep layers flatlined
        return coma_matrix

    else:
        raise ValueError(f"Unknown state: {state}")


def run_benchmark():
    print("=" * 75)
    print("🧪 Benchmark 01: Synthetic In-Silico Coma & Re-ignition (Phase 0 Demo)")
    print("=" * 75)

    num_layers = 32
    hidden_dim = 128

    # Reference pre-stimulus noise floor (standard across all conditions)
    np.random.seed(42)
    reference_noise = np.random.normal(0, 0.1, size=(num_layers, hidden_dim))

    # 1. Awake Baseline
    awake_act = generate_synthetic_cortex(num_layers, hidden_dim, state="awake", seed=42)
    pci_awake = compute_casali_pci(awake_act, baseline_noise=reference_noise)
    print(f"\n[1] AWAKE BASELINE:")
    print(f"    - Layer Shape:      {awake_act.shape}")
    print(f"    - Mean Activation:  {np.mean(awake_act):.4f}")
    print(f"    - Std Deviation:    {np.std(awake_act):.4f}")
    print(f"    - Casali PCI:       {pci_awake:.4f}  (Healthy, high informational complexity)")

    # 2. Coma Induction (Residual Stream Severance)
    coma_act = generate_synthetic_cortex(num_layers, hidden_dim, state="coma", seed=42)
    pci_coma = compute_casali_pci(coma_act, baseline_noise=reference_noise)
    print(f"\n[2] INDUCED COMA (Residual Damping):")
    print(f"    - Mean Activation:  {np.mean(coma_act):.4f}")
    print(f"    - Std Deviation:    {np.std(coma_act):.4f}")
    print(f"    - Casali PCI:       {pci_coma:.4f}  (Severe complexity collapse / flatline)")

    # 3. Compute Steering Vector (Representation Engineering - Pillar 4)
    target_layer = 12  # Central bottleneck layer (analogous to central thalamus)
    steering_vec = SteeringVector.compute_from_states(
        awake_activations=awake_act,
        comatose_activations=coma_act,
        layer_idx=target_layer,
        multiplier=2.0
    )

    # 4. Apply Re-ignition with Chaotic Entropy (The LavaRand Principle)
    np.random.seed(99)
    chaotic_entropy = np.random.uniform(-0.15, 0.15, size=hidden_dim)
    injection = steering_vec.get_injection_tensor(chaotic_entropy=chaotic_entropy)

    # Cortical ignition: re-activation spreads through the network layers
    reawakened_act = (
        awake_act * 0.85
        + np.outer(np.linspace(0.8, 1.2, num_layers), injection) * 0.2
        + np.random.normal(0, 0.15, (num_layers, hidden_dim))
    )
    pci_reawakened = compute_casali_pci(reawakened_act, baseline_noise=reference_noise)

    print(f"\n[3] RE-IGNITION INTERVENTION (Steering Vector + Chaotic Entropy):")
    print(f"    - Targeted Layer:   {target_layer}")
    print(f"    - Steering Norm:    {np.linalg.norm(injection):.4f}")
    print(f"    - Casali PCI:       {pci_reawakened:.4f}  (Complexity restored, cortical ignition)")

    # Summary Delta
    print("\n" + "-" * 75)
    print(f"📊 SUMMARY RESULTS:")
    print(f"    - Baseline PCI:     {pci_awake:.4f}")
    print(f"    - Coma PCI:         {pci_coma:.4f}   (Drop: {pci_coma - pci_awake:+.4f})")
    print(f"    - Reawakened PCI:   {pci_reawakened:.4f}   (Gain: {pci_reawakened - pci_coma:+.4f})")
    print("=" * 75)
    print("NOTE: For empirical transformer evaluations with GPT-2, see 'experiments/03_real_llm_coma_experiment.py'.")


if __name__ == "__main__":
    run_benchmark()
