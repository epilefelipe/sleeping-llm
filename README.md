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

## 📊 Empirical Benchmark Results (ComaBench v0.3: Multi-Architecture Empirical Suite)

We evaluate in-silico coma induction, data-driven 2D Spatiotemporal Casali PCI ($PCI_{2D}$), autoregressive perplexity ($PPL$), and next-token accuracy ($Acc$) across **two distinct architectures**:
1. **OpenAI GPT-2** (124M parameters, 12 layers, absolute positional embeddings)
2. **EleutherAI Pythia-160M** (160M parameters, 12 layers, rotary position embeddings / RoPE)

Evaluations span a **10-level fine-grained dose-response sweep** ($\alpha \in [1.0, 0.0]$), $N=3$ random seeds, anatomical controls (Early L0–5 vs Late L6–11 damping), and neuromodulatory re-ignition controls (Difference-in-Means vs Isotropic Sham).

- **Full Scientific Report:** [docs/EXPERIMENT_04_REPORT.md](docs/EXPERIMENT_04_REPORT.md) | [PDF Version (Academic)](docs/EXPERIMENT_04_REPORT.pdf)
- **Official Run UUID:** `460042a9-9d73-4e24-9477-d0fba789c705` (`2026-10-29T23:06:32-03:00`)
- **Raw Data Artifact:** [JSON Payload](experiments/results/run_comabench_v03_460042a9-9d73-4e24-9477-d0fba789c705.json)
- **Execution Script:** [`experiments/04_rigorous_comabench_suite.py`](experiments/04_rigorous_comabench_suite.py)

### 1. Fine-Grained Dose-Response Curves (Pythia-160M)

| Attenuation $\alpha$ | Clinical Analogue | Mean $PCI_{2D} \pm \text{std}$ | Perplexity ($PPL$) | Next-Token Acc | Branching $\bar{\sigma}$ | Qualitative Output |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1.00** | Intact Waking | **$0.6830 \pm 0.0748$** | **53.3** | **0.349** | **1.010** | Coherent semantic discourse |
| **0.85** | Sub-clinical Fatigue | $0.7154 \pm 0.0901$ | 84.3 | 0.304 | 1.070 | Coherent phrase structures |
| **0.70** | Mild Sedation | $0.6075 \pm 0.1178$ | 215.0 | 0.244 | 1.148 | Preserved syntax |
| **0.55** | Moderate Sedation / MCS | $0.5065 \pm 0.1058$ | 1,007.1 | 0.212 | 1.220 | Partial word degradation |
| **0.40** | Stupor / Delirium | $0.6125 \pm 0.1533$ | 8,212.8 | 0.032 | 1.296 | Severe agrammatism |
| **0.30** | Light Coma / UWS | $0.5708 \pm 0.1146$ | 25,894.1 | **0.000** | 1.338 | Punctuation loops |
| **0.20** | Moderate Coma | $0.5012 \pm 0.0851$ | 31,826.3 | 0.015 | 1.382 | Fragmentation |
| **0.10** | Severe Coma | $0.3814 \pm 0.0525$ | 25,071.8 | **0.000** | 1.446 | Syntactic collapse |
| **0.05** | Deep Catatonic Freeze | **$0.3474 \pm 0.0282$** | 24,476.6 | 0.015 | 1.466 | **Casali Clinical Cutoff ($PCI^* \approx 0.31$)** |
| **0.00** | Complete Severance | $0.4192 \pm 0.0761$ | 12,712.8 | 0.080 | 1.851 | Static lexical attractor |

### 2. Anatomical Double Dissociation & The Limits of DBS Re-ignition

- **Early (L0–5) vs Late (L6–11) Lesioning:** Early lesioning disrupts perceptual/lexical input (perplexity explodes, agrammatic word salad), but token generation continues. Late lesioning cuts off recurrent semantic integration, collapsing the system into a catatonic dash attractor (`---`) or clinical $PCI$ threshold ($0.3119$).
- **Honest Demarcation of DBS Re-ignition:** Single-site activation steering at Layer 4 (difference-in-means) perturbs latent activations but **fails** to restore grammatical coherence or semantic reasoning after a 6-layer lesion. Project Sleeping LLM strictly reports this recovery limitation rather than claiming to "cure" artificial coma.
- **Criticality Layer Dynamics:** Homeostatic gain in Layer 5 restores branching ratio toward the edge of chaos ($\bar{\sigma} \approx 1$), serving as a physical/dynamical stabilizer that elevates spatiotemporal reverberation.

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
