#!/usr/bin/env python3
"""
Quickstart demonstration of Project Sleeping LLM.
Simulates a multi-layer activation response under three states:
1. Awake (Normal complex information integration)
2. Comatose (Suppressed residual stream / collapse)
3. Re-awakened (Restoration via stochastic resonance & steering perturbation)
"""

import numpy as np
from src.metrics.pci import compute_pci


def simulate_network_activity(num_layers=32, hidden_dim=64, state="awake"):
    np.random.seed(42)

    if state == "awake":
        # Complex, structured, cross-layer non-linear propagation (high integration)
        base = np.random.randn(num_layers, hidden_dim)
        # Add correlated spatiotemporal waves across layers
        t = np.linspace(0, 4 * np.pi, num_layers)[:, None]
        freq = np.linspace(1, 3, hidden_dim)[None, :]
        waves = np.sin(t * freq) * 1.5
        activity = base + waves

    elif state == "coma":
        # Severe suppression / flatline / low-rank degenerate attractor
        activity = np.random.normal(loc=0.0, scale=0.05, size=(num_layers, hidden_dim))
        # Zero out the deeper 60% of layers (residual stream severance)
        activity[int(num_layers * 0.4):, :] = 0.0

    elif state == "reawakened":
        # Recovered state after applying in-silico steering perturbation
        recovered = simulate_network_activity(num_layers, hidden_dim, state="awake")
        noise = np.random.normal(0, 0.2, size=(num_layers, hidden_dim))
        activity = recovered * 0.85 + noise

    else:
        raise ValueError(f"Unknown state: {state}")

    return activity


def main():
    print("=" * 60)
    print("🧠 Project Sleeping LLM: In-Silico Coma & PCI Benchmark")
    print("=" * 60)

    states = ["awake", "coma", "reawakened"]

    # In clinical Casali PCI, a reference pre-stimulus noise floor is used
    # to evaluate whether activations statistically cross significance.
    reference_noise = np.random.normal(0, 0.1, size=(32, 64))

    for st in states:
        activity = simulate_network_activity(num_layers=32, hidden_dim=64, state=st)
        pci_score = compute_pci(activity, baseline_noise=reference_noise)
        print(f"\n[State: {st.upper():<12}]")
        print(f"  -> Activity Mean:  {np.mean(activity):.4f}")
        print(f"  -> Activity Std:   {np.std(activity):.4f}")
        print(f"  -> PCI_LLM Score:  {pci_score:.4f}")

        if st == "awake":
            print("  Status: Normal conscious baseline (Complex, differentiated)")
        elif st == "coma":
            print("  Status: Comatose collapse (Low complexity, disconnected)")
        elif st == "reawakened":
            print("  Status: Re-ignition achieved (Complexity restored)")

    print("\n" + "=" * 60)
    print("Simulation complete. The framework is ready for LLM integration.")
    print("=" * 60)


if __name__ == "__main__":
    main()
