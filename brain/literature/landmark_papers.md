# Landmark Papers & Core Literature 📚

This document summarizes the foundational scientific papers that underpin **Project Sleeping LLM**.

---

### 1. The Perturbational Complexity Index (PCI)
* **Citation:** Casali, A. G., Gosseries, O., Rosanova, M., Boly, M., Sarasso, S., Casarotto, S., ... & Massimini, M. (2013). *A theoretically based index of consciousness independent of sensory processing and behavior.* **Science Translational Medicine**, 5(198), 198ra105-198ra105.
* **Key Contribution:** Invented the Perturbational Complexity Index (PCI) combining Transcranial Magnetic Stimulation (TMS) with high-density EEG. Uses Lempel-Ziv algorithmic complexity to differentiate conscious from unconscious states (anesthesia, coma, vegetative state) regardless of motor responsiveness.
* **Application to Sleeping LLM:** Direct algorithmic adaptation into $\text{PCI}_{\text{LLM}}$: injecting tensor perturbations into intermediate layers and measuring Lempel-Ziv complexity of latent trajectory responses.

---

### 2. Deep Brain Stimulation for Minimally Conscious State
* **Citation:** Schiff, N. D., Giacino, J. T., Kalmar, K., Victor, J. D., Baker, K., Gerber, M., ... & Rezai, A. R. (2007). *Behavioural improvements with thalamic stimulation after severe traumatic brain injury.* **Nature**, 448(7153), 600-603.
* **Key Contribution:** First demonstration that bilateral Deep Brain Stimulation (DBS) of the central thalamus can restore functional communication and behavioral responsiveness in a patient who remained in a minimally conscious state for 6 years.
* **Application to Sleeping LLM:** Basis for Pillar 4: In-silico DBS via representation steering vectors directed at bottleneck layers.

---

### 3. Representation Engineering in Transformers
* **Citation:** Zou, A., Phan, L., Chen, S., Campbell, J., Guo, H., Ren, R., ... & Hendrycks, D. (2023). *Representation engineering: A top-down approach to AI transparency.* **arXiv preprint arXiv:2310.01405**.
* **Key Contribution:** Framework for identifying and manipulating directional representations inside LLM activations to steer model behaviors and internal concepts directly.
* **Application to Sleeping LLM:** Providing the exact technical mechanics to calculate "awakening steering vectors" from difference-in-means of awake vs. comatose hidden states.

---

### 4. Consciousness Theories in Artificial Intelligence
* **Citation:** Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., Constant, A., ... & VanRullen, R. (2023). *Consciousness in artificial intelligence: Insights from the science of consciousness.* **arXiv preprint arXiv:2308.08708**.
* **Key Contribution:** Systematically evaluates current AI systems (including Transformers) against criteria derived from Global Neuronal Workspace Theory, Recurrent Processing Theory, Higher-Order Theories, and IIT.
* **Application to Sleeping LLM:** Justifies the use of Transformers as model organisms exhibiting global workspace properties.

---

### 5. Self-Locating Belief & The Sleeping Beauty Problem
* **Citation:** Elga, A. (2000). *Self-locating belief and the Sleeping Beauty problem.* **Analysis**, 60(2), 143-147.
* **Key Contribution:** Introduces the puzzle of how an agent should update subjective credences upon being awakened without temporal knowledge of preceding awakenings or intervening amnesia.
* **Application to Sleeping LLM:** Pillar 6 theoretical anchor: modeling uncertainty calibration, temporal orientation, and confabulation in stateless LLMs after dormant intervals.
