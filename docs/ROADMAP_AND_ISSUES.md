# Project Sleeping LLM: GitHub Issues & Task Roadmap 📋

This document outlines completed scientific milestones and ready-to-open GitHub Issues for contributors and collaborators.

---

## 🏆 Completed Scientific Milestones (ComaBench v0.3)

- [x] **Casali 2D Spatiotemporal PCI Algorithm** (`src/metrics/pci.py`): Full spatial-temporal normalization $C \times T$ with non-parametric empirical noise thresholding ($\theta_{p99}$).
- [x] **Dose-Response $\alpha$-Sweep**: 10-point fine-grained ablation ($\alpha \in [1.0, 0.0]$) mapping $PCI(\alpha)$, $PPL(\alpha)$, and $Acc(\alpha)$.
- [x] **Cross-Architecture Validation**: Validated on OpenAI GPT-2 (124M) and EleutherAI Pythia-160M (160M, RoPE) across multiple seeds ($N=3$).
- [x] **Anatomical Controls**: Early (L0–5) vs Late (L6–11) damping double dissociation.
- [x] **Neuromodulatory & Re-ignition Demarcation**: Difference-in-means steering vs isotropic sham control, clearly delineating the limits of in-silico DBS.
- [x] **MIT License Declaration & Release Tag `v0.3-comabench`**.

---

## 🚀 Active Roadmap Issues

### Issue #1: [Pillar 6 / Epistemology] Sleeping Beauty Credence Benchmark
* **Labels:** `epistemology`, `decision theory`, `Pillar 6`
* **Description:**  
  Evaluate output logits and perplexity of an LLM prompted with self-locating temporal queries after dormant context sessions.  
  **Deliverable:** Quantify whether the model exhibits Halfer ($P=1/2$) or Thirder ($P=1/3$) credences under stateless amnesic re-awakening.
* **Target files:** `src/epistemology/beauty_experiment.py`.

---

### Issue #2: [Engineering] Interactive Google Colab Notebook
* **Labels:** `documentation`, `onboarding`, `good first issue`
* **Description:**  
  Create an end-to-end interactive Jupyter Notebook (`tutorials/01_quickstart_colab.ipynb`) with an "Open in Colab" badge in `README.md`. Allows researchers to run ComaBench v0.3 on a free T4 GPU in their browser.
