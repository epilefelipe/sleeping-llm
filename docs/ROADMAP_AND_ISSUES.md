# Project Sleeping LLM: GitHub Issues & Task Roadmap 📋

This document outlines the ready-to-open GitHub Issues for contributors and collaborators.

---

### Issue #1: [Core/Pillar 4] Implement Contrastive Representation Steering on Real Transformer Layers
* **Labels:** `enhancement`, `Pillar 4`, `help wanted`, `good first issue`
* **Description:**  
  We have implemented the initial mathematical scaffold in `src/awakening/steering.py` (`SteeringVector`, `ActivationSteeringHook`).  
  **Deliverable:** Build an adapter to extract difference-in-means vectors directly from open HuggingFace causal LMs (e.g., `meta-llama/Llama-3.2-1B` or `Qwen/Qwen2.5-0.5B`) between coherent and comatose prompt activations.
* **Target files:** `src/awakening/steering.py`, `src/core/adapter.py`.

---

### Issue #2: [Pillar 1 / Benchmark] Hook Open LLMs with Coma Attenuation Protocols
* **Labels:** `research`, `Pillar 1`, `high priority`
* **Description:**  
  Connect `src/coma/hooks.py` with a small open-source model running on CPU/CUDA.  
  **Deliverable:** Verify that clamping the residual stream at layers $L/2$ to $L$ induces a measurable drop in `src/metrics/pci.py` while keeping initial input embedding projections intact (satisfying the *Afference Preservation Criterion*).
* **Target files:** `src/coma/hooks.py`, `experiments/`.

---

### Issue #3: [Pillar 2 / Metrics] Expand PCI to Multichannel Spatiotemporal Dynamics (Casali 2013)
* **Labels:** `math`, `Pillar 2`, `computational neuroscience`
* **Description:**  
  The current PCI implementation (`src/metrics/pci.py`) uses 1D serialization of Lempel-Ziv LZ76.  
  **Deliverable:** Implement the 2D spatial-temporal asymptotic normalization from Casali et al. (Science TM, 2013) to account for cross-layer channel interactions and spectral entropy.
* **Target files:** `src/metrics/pci.py`.

---

### Issue #4: [Pillar 6 / Epistemology] Sleeping Beauty Credence Benchmark
* **Labels:** `epistemology`, `decision theory`, `Pillar 6`
* **Description:**  
  Evaluate output logits and perplexity of an LLM prompted with self-locating temporal queries after dormant context sessions.  
  **Deliverable:** Quantify whether the model exhibits Halfer ($P=1/2$) or Thirder ($P=1/3$) credences under stateless amnesic re-awakening.
* **Target files:** `src/epistemology/beauty_experiment.py`.

---

### Issue #5: [Engineering] Interactive Google Colab Notebook
* **Labels:** `documentation`, `onboarding`, `good first issue`
* **Description:**  
  Create an end-to-end interactive Jupyter Notebook (`tutorials/01_quickstart_colab.ipynb`) with an "Open in Colab" badge in `README.md`. Allows researchers to run Benchmark 01 on a free T4 GPU in their browser.
