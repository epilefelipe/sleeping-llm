# Project Sleeping LLM 🧠💤

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23536600.svg)](https://doi.org/10.5281/zenodo.23536600)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **In-Silico Modeling of Comatose States and Cognitive Re-ignition Dynamics in Deep Neural Networks**

---

## 📌 Executive Summary

Disorders of Consciousness (DoC)—such as traumatic coma, unresponsive wakefulness syndrome (UWS), and minimally conscious states (MCS)—remain one of the most tragic and unsolved mysteries in modern clinical medicine. Because invasive experimentation on living human brains is ethically restricted, we lack a comprehensive understanding of the physical and informational mechanisms of **cortical awakening (ignition)**.

**Project Sleeping LLM** introduces a novel computational paradigm: using modern autoregressive Transformer architectures (LLMs) as **in-silico model organisms** for cortical disconnection.

1. **Inducing Artificial Coma:** We model functional analogues of coma etiologies (residual stream damping, attention entropy collapse, and latent attractor clamping).
2. **Measuring Consciousness In-Silico:** We adapt the clinical **Perturbational Complexity Index (PCI)** (Massimini & Casali, 2013) to measure internal information integration ($\text{PCI}_{\text{LLM}}$).
3. **Discovering Awakening Interventions:** We systematically test in-silico Deep Brain Stimulation (DBS via representation steering vectors), critical stochastic resonance, and recurrent feedback loops to trigger cognitive re-ignition.

---

## 🔬 Theoretical Foundations

- **Global Neuronal Workspace Theory (GNWT):** Modeled via the multi-head attention broadcasting and the residual stream highway.
- **Integrated Information Theory (IIT) & PCI:** Evaluating the algorithmic compressibility (Lempel-Ziv) of internal hidden state trajectories following localized tensor perturbations.
- **Mechanistic Interpretability:** Utilizing representation engineering to discover directional steering vectors that restore coherence to collapsed attractors.

---

## 📂 Repository Structure

```
sleeping-llm/
├── docs/
│   └── whitepaper.pdf              # Official working paper (Zenodo DOI: 10.5281/zenodo.23536600)
├── src/
│   ├── coma/                       # Protocols for inducing artificial comatose states
│   │   ├── __init__.py
│   │   └── hooks.py                # PyTorch forward hooks for residual stream / attention clamping
│   ├── metrics/                    # Complexity and consciousness metrics
│   │   ├── __init__.py
│   │   └── pci.py                  # Perturbational Complexity Index (PCI) & Lempel-Ziv implementation
│   └── awakening/                  # Algorithmic interventions & re-ignition protocols
│       ├── __init__.py
│       └── steering.py             # Representation steering & activation injection
├── experiments/                    # Benchmark scripts and ablation runs
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/your-username/sleeping-llm.git
cd sleeping-llm
pip install -r requirements.txt
```

---

## 📄 Working Paper Citation

If you use this framework or theoretical architecture in your research, please cite:

```bibtex
@article{abarzua2026sleepingllm,
  title={Project Sleeping LLM: Modeling Disorders of Consciousness via In-Silico Coma Induction and Reawakening Dynamics in Large Language Models},
  author={Abarzua Herrera, Felipe Antonio},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.23536600},
  url={https://doi.org/10.5281/zenodo.23536600}
}
```

---

## 🤝 Collaboration & Contact

Initiator & Project Lead: **Felipe Antonio Abarzua Herrera**  
Zenodo Record: [10.5281/zenodo.23536600](https://doi.org/10.5281/zenodo.23536600)  
Open for collaborations with researchers from computational neuroscience, machine learning, and neurology.
