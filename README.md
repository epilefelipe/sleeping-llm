# Project Sleeping LLM 🧠💤

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23536600.svg)](https://doi.org/10.5281/zenodo.23536600)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **In-Silico Modeling of Comatose States, Cognitive Re-ignition Dynamics, and Self-Locating Epistemology in Deep Neural Networks**

---

## 📌 Executive Summary

Disorders of Consciousness (DoC)—such as traumatic coma, unresponsive wakefulness syndrome (UWS), and minimally conscious states (MCS)—remain one of the most tragic and unsolved mysteries in modern clinical medicine. Because invasive experimentation on living human brains is ethically restricted, we lack a comprehensive understanding of the physical and informational mechanisms of **cortical awakening (ignition)**.

**Project Sleeping LLM** introduces a novel computational paradigm: using modern autoregressive Transformer architectures (LLMs) as **in-silico model organisms** for cortical disconnection.

1. **Inducing Artificial Coma:** We model functional analogues of coma etiologies (residual stream damping, attention entropy collapse, and latent attractor clamping).
2. **Measuring Consciousness In-Silico:** We adapt the clinical **Perturbational Complexity Index (PCI)** (Massimini & Casali, 2013) to measure internal information integration ($\text{PCI}_{\text{LLM}}$).
3. **Discovering Awakening Interventions:** We systematically test in-silico Deep Brain Stimulation (DBS via representation steering vectors), critical stochastic resonance, and recurrent feedback loops to trigger cognitive re-ignition.
4. **Epistemic Self-Location:** We integrate the **Sleeping Beauty Dilemma** to evaluate how a stateless cognitive network re-calibrates temporal priors and subjective probability upon awakening.

---

## 🧭 Epistemological Stance: The Map is Not the Territory

> *"We cannot treat the programming and the hypothesis in the same way. The code is merely a computational telescope; the hypothesis is the biological mystery of the mind."*  
> — **Project Sleeping LLM Foundational Principle**

We explicitly reject **computational reductionism**. The software in this repository does **not** claim that a Large Language Model is a human brain, nor do we claim machine sentience. 

Instead, the codebase serves strictly as an **in-silico laboratory instrument** (analogous to a particle accelerator in physics): a controlled, reproducible testbed to explore non-linear perturbation dynamics and generate rigorous mathematical hypotheses before translating them to clinical neuroscience.

---

The project is structured across six interdisciplinary pillars, establishing clear boundaries between **LLM Engineering**, **Scientific Theory**, and the dynamics of **Sleeping & Awakening**:

| # | Pillar | Primary Domain | Research Focus & Objective |
| :---: | :--- | :--- | :--- |
| **1** | **Decoupling Architecture** | LLM Engineering + Sleeping | Designing surgical PyTorch hooks to clamp the residual stream, ablate attention heads, and simulate pharmacological/structural coma etiologies without destructive weight overwriting. |
| **2** | **In-Silico Consciousness Metrics** | Scientists + LLM Engineering | Adapting the **Perturbational Complexity Index (PCI_LLM)** using Lempel-Ziv algorithmic compressibility and Integrated Information Theory ($\Phi$) over internal latent trajectories. |
| **3** | **Re-ignition Physics & Criticality** | Sleeping + Scientists | Treating the collapsed model as a non-linear dynamical system in a low-energy attractor; evaluating **stochastic resonance** (critical noise thresholds) and recurrent feedback to trigger state phase transitions. |
| **4** | **Representation Steering (In-Silico DBS)** | LLM Engineering + Sleeping | Leveraging **Representation Engineering** to discover directional steering vectors in activation space, acting as mathematical deep brain stimulation to restore reasoning coherence. |
| **5** | **Biomedical & Clinical Translation** | Scientists + Sleeping | Mapping artificial activation matrices against high-density clinical EEG datasets from comatose patients to formulate translational non-invasive stimulation protocols (tACS/tDCS/tFUS). |
| **6** | **The Sleeping Beauty Epistemology** | Epistemology + Sleeping + Scientists | Investigating self-locating belief (*self-locating credence*), temporal amnesia, and uncertainty calibration when a stateless network is re-awakened from dormant intervals. |

---

## 🔬 Theoretical Foundations

- **Global Neuronal Workspace Theory (GNWT):** Modeled via multi-head attention broadcasting and the residual stream highway.
- **Integrated Information Theory (IIT) & PCI:** Evaluating the algorithmic compressibility (Lempel-Ziv) of internal hidden state trajectories following localized tensor perturbations.
- **Mechanistic Interpretability:** Utilizing representation engineering to discover directional steering vectors that restore coherence to collapsed attractors.
- **Decision Theory & Self-Locating Credence:** Addressing the **Sleeping Beauty Problem** within artificial intelligence: how does a network lacking continuous episodic memory establish orientation and calibrate credence upon re-ignition?

---

## 📂 Repository Structure

```
sleeping-llm/
├── docs/
│   ├── whitepaper.pdf              # Official working paper (Zenodo DOI: 10.5281/zenodo.23536600)
│   ├── EXPERIMENT_03_REPORT.md     # Official empirical benchmark technical report
│   ├── EXPERIMENT_03_REPORT.pdf    # Academic technical report (5 pages)
│   └── ROADMAP_AND_ISSUES.md       # Prioritized backlog
├── src/
│   ├── coma/                       # Pillar 1: Protocols for inducing artificial comatose states
│   │   ├── __init__.py
│   │   └── hooks.py                # PyTorch forward hooks for residual stream / attention clamping
│   ├── metrics/                    # Pillar 2: Complexity, criticality & consciousness metrics
│   │   ├── __init__.py
│   │   ├── pci.py                  # Perturbational Complexity Index (PCI) & Lempel-Ziv implementation
│   │   └── criticality.py          # Pillar 3: Branching ratios (σ), criticality index (κ), stability
│   ├── awakening/                  # Pillars 3 & 4: Algorithmic interventions & re-ignition protocols
│   │   ├── __init__.py
│   │   ├── steering.py             # Representation steering & activation injection
│   │   └── criticality.py          # CriticalityLayer: Homeostatic gain & stochastic resonance
│   └── epistemology/               # Pillar 6: The Sleeping Beauty problem & self-locating credence
│       ├── __init__.py
│       └── beauty_experiment.py    # Self-locating prior evaluation in stateless re-awakening
├── experiments/                    # Benchmark scripts and ablation runs
├── quickstart_demo.py              # Minimal simulation script demonstrating the 3 states
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/epilefelipe/sleeping-llm.git
cd sleeping-llm

# Create and activate isolated environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the initial simulation
python quickstart_demo.py
```

---

## 📊 Empirical Benchmark Results (ComaBench v3: Real Transformer In-Silico Coma)

We evaluate in-silico coma induction, data-driven 2D Spatiotemporal Casali PCI, and re-ignition using a real pretrained autoregressive transformer (`GPT-2`, 124M parameters, 12 layers) across 3 diverse cognitive domains (Consciousness, Neurobiology, Logic) with lesion severity sweeps ($\alpha \in [0.50, 0.20, 0.05]$) and iso-energetic sham controls.

- **Full Scientific Report:** [docs/EXPERIMENT_03_REPORT.md](docs/EXPERIMENT_03_REPORT.md) | [PDF Version (5 pages)](docs/EXPERIMENT_03_REPORT.pdf)
- **Official Run UUID:** `3baa4ab0-fc29-4f4b-a719-892ab4c729a2` (`2026-10-21T16:29:04.277-03:00`)
- **Raw Data Artifacts:** [JSON Payload](experiments/results/run_3baa4ab0-fc29-4f4b-a719-892ab4c729a2.json) | [Execution Log](experiments/results/run_3baa4ab0-fc29-4f4b-a719-892ab4c729a2.md)

| Experimental Condition | Lesion Severity / Setup | Mean PCI_2D | Behavioral Signature |
| :--- | :---: | :---: | :--- |
| **Stage 1: Awake Baseline** | *Intact Reference* | **0.6201** | Coherent, fluent syntactic discourse |
| **Stage 2: Mild Sedation (MCS)** | $\alpha = 0.50$ | **0.3324** | Partial syntax preserved (conjunctions, prepositions) |
| **Stage 3: Deep Coma (Severed)** | $\alpha = 0.05$ | **0.3324** | Catatonic mutism (`-------------------------`) |
| **Stage 4: Iso-Energetic Focal Sham** | $\|v\| = 100.0$, Layer 4 | **0.3596** | Clinical boundary ($PCI^* \approx 0.31$), uninformative loops |
| **Stage 5: Unipolar Steering (L4)** | $m = 1.5$, Layer 4 | **0.3324** | Dash attractor breached; punctuation emergence |
| **Stage 6: Multi-Site Steering** | Layers 4 + 7 + 10 | **0.3324** | Distributed manifold activation |
| **Stage 7: Criticality Layer Tuning** | Gateway Layer 5 | **0.3981** | **Morphological recovery (sub-words `able`, `ability`)** |

> **Key Takeaway:** ComaBench v3 eliminates arbitrary thresholds by deriving baseline noise matrices ($\Delta_{\text{baseline}}$) directly from embedding-level micro-jitter. In-silico coma causes an authentic drop in 2D Spatiotemporal PCI from **`0.6201`** to **`0.3324`** ($\Delta = -0.2878$). The **Criticality Layer** yields the highest complexity recovery ($\text{PCI} = 0.3981$, $+0.0657$) and restores morphological English sub-words from total mutism.

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

## 🤝 Contributing & Collaboration

We are in **Phase 0** and actively seeking collaborators across computational neuroscience, machine learning, and neurology. 

- **How to contribute:** Read our [CONTRIBUTING.md](CONTRIBUTING.md) guide for setup, priorities, and co-authorship policy.
- **Open tasks & roadmap:** Check [docs/ROADMAP_AND_ISSUES.md](docs/ROADMAP_AND_ISSUES.md) for prioritized issues.
- **Project Lead:** Felipe Antonio Abarzua Herrera
- **Zenodo Record:** [10.5281/zenodo.23536600](https://doi.org/10.5281/zenodo.23536600)
- **GitHub:** [https://github.com/epilefelipe/sleeping-llm](https://github.com/epilefelipe/sleeping-llm)
