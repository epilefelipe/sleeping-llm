# Contributing to Project Sleeping LLM 🧠🤝

First off, thank you for considering contributing to **Project Sleeping LLM**! 

The project is currently in **Phase 0 (Foundational Architecture & Pre-print)**. We are building an open-source *in-silico* computational framework to model comatose states in deep Transformer networks and discover cognitive re-ignition mechanisms.

Because the project is young and registered on Zenodo ([DOI: 10.5281/zenodo.23536600](https://doi.org/10.5281/zenodo.23536600)), **any high-quality contribution will be immediately visible and acknowledged**, with potential for co-authorship on future peer-reviewed publications and grant proposals (e.g., Templeton World Charity Foundation, Wellcome Trust, ANID).

---

## 🗺️ Contribution Roadmap & High-Impact Priorities

We welcome contributions across machine learning engineering, computational neuroscience, and clinical neurology. Here is our prioritized list of needs:

| Priority | Area / Task | Description & Deliverable | Impact | Complexity |
| :---: | :--- | :--- | :---: | :---: |
| 🔴 **HIGH** | **Representation Steering (`src/awakening/steering.py`)** | Implement representation engineering / steering vectors to counteract comatose attractor collapse (Pillar 4). | **Massive** | Medium |
| 🔴 **HIGH** | **Real HuggingFace LLM Integration** | Connect `src/coma/hooks.py` with an open-weights model (e.g., `Llama-3.2-1B/3B` or `Qwen-2.5-0.5B`) to measure real empirical activation PCI deltas. | **Massive** | Medium-High |
| 🟡 **MEDIUM** | **ComaBench Experiment Harness (`experiments/`)** | Build reproducible benchmark scripts comparing baseline, coma, and re-awakened states on zero-shot reasoning benchmarks. | **High** | Medium |
| 🟡 **MEDIUM** | **Advanced Complexity Metrics (`src/metrics/`)** | Expand beyond 1D Lempel-Ziv into multi-channel spatiotemporal PCI (as in Casali et al. 2013) and spectral entropy. | **High** | Medium |
| 🟢 **LOW** | **Knowledge Base Expansion (`brain/`)** | Add detailed summaries of relevant papers, clinical EEG datasets, and empirical notes. | **Medium** | Low |
| 🟢 **LOW** | **Testing & CI/CD** | Add unit tests (`pytest`), GitHub Actions CI workflows, and package distribution. | **Medium** | Low |

---

## 🛠️ Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/sleeping-llm.git
   cd sleeping-llm
   ```

2. **Set up the isolated virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify the environment:**
   ```bash
   python quickstart_demo.py
   ```

---

## 📑 How to Propose Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Commit your changes:** Follow conventional commits (e.g., `feat: implement contrastive steering vector extractor`).
3. **Push to your fork:**
   ```bash
   git push origin feat/your-feature-name
   ```
4. **Open a Pull Request:** Explain the theoretical motivation, link to any relevant literature, and provide reproducible test output.

---

## 📜 Co-Authorship & Intellectual Property Policy

* All software code is released under the **MIT License**.
* The foundational whitepaper and theoretical frameworks are published under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.
* Contributors who implement major technical milestones (such as the real model benchmark or novel mathematical metrics) will be invited as **co-authors** on the upcoming Phase 1 arXiv preprint and associated grant applications.

---

## 📬 Contact the Project Initiator

If you want to discuss a major collaboration, research grant, or clinical data partnership:

* **Initiator & Project Lead:** Felipe Antonio Abarzua Herrera
* **Zenodo Record:** [10.5281/zenodo.23536600](https://doi.org/10.5281/zenodo.23536600)
* **GitHub Issues:** Open an issue on this repository for technical questions or proposals.
