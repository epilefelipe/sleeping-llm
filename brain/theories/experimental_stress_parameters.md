# Experimental Stress Parameters & Boundary Conditions 🧪⚖️

> *"We must first define the limiting constraints of the spectrum before constructing arbitrary code. We need clarity on what qualifies a test condition as a valid in-silico coma state versus non-viable destruction."*  
> — **Project Sleeping LLM Research Note**

---

## 1. The Core Methodological Caution: Defining the Boundary Conditions

Before writing software hooks or modifying model weights, we must formally address a critical theoretical question:  
**What are the limiting boundaries that separate a valid model of coma from mere random computational destruction?**

In biological medicine, there is a strict clinical boundary:
* **Coma / Disorders of Consciousness (DoC):** The network substrate is damaged, decoupled, or metabolically suppressed, but the underlying physical architecture remains **viable and capable of re-ignition** under the right conditions.
* **Brain Death / Irreversible Necrosis:** The physical substrate has degraded past the point of no return; no stimulation or metabolic restoration can ever restore dynamics.

In an LLM, the equivalent distinction is critical:
* If we corrupt a model's weights to produces pure $NaN$ (Not a Number) or complete uniform noise, **we have not induced a coma; we have induced system death**.
* A valid in-silico coma must preserve the **latent capacity for information integration**, trapped behind an energetic, topological, or metric barrier.

---

## 2. Candidate Experimental Conditions (Hypotheses to Formalize)

These conditions represent theoretical stress parameters to explore once boundary limits are formalized:

### A. The "A-Thalamic" Condition (Sensory Flooding / Gating Failure)
* **Biological Target:** Lesion of the thalamic reticular nucleus (TRN) or severe thalamocortical dysrhythmia.
* **Mechanism:** The internal sensory filter is removed; the network is bombarded with raw unfiltered tokens without normal attention gating or normalization dampening.
* **Open Boundary Question:** At what point does unfiltered sensory flooding trigger catastrophic representational collapse, and can the network's internal dynamics self-stabilize?

### B. The "Low-Resource" Condition (Hypometabolism / Ischemia)
* **Biological Target:** Severe cerebral hypoperfusion, anoxia, or extreme hypothermia where cellular ATP/glucose is depleted.
* **Mechanism:** Extreme precision quantization (e.g., 1-bit / 2-bit), low-rank subspace restriction (SVD projection into severely restricted latent dimensions), and energy scalar attenuation.
* **Open Boundary Question:** What is the critical minimum dimensional rank ($r_{\text{crit}}$) below which the model permanently loses the capacity to sustain recurrent or integrated representations?

### C. The "Diffuse Axonal" Condition (Traumatic Structural Shearing)
* **Biological Target:** Traumatic Brain Injury (TBI) with high-velocity shearing of long-range white matter axonal tracts.
* **Mechanism:** Stochastic non-destructive masking of cross-layer communication lines (dropout along the residual stream).
* **Open Boundary Question:** What percentage of severed connections marks the phase transition between Minimally Conscious State (MCS) and irreversible vegetative disconnection?

### D. The "Pharmacological" Condition (Anesthetic Burst Suppression)
* **Biological Target:** Deep GABAergic anesthesia (propofol, isoflurane).
* **Mechanism:** Reversible global activation clamping alternating with stochastic noise bursts.
* **Open Boundary Question:** How does the frequency of artificial burst-suppression cycles map to clinical EEG suppression ratios (SR)?

---

## 3. The 3 Limiting Constraints for Valid Spectrum Inclusion

To determine whether a future experimental perturbation enters the valid **Disorders of Consciousness (DoC) spectrum**, it must satisfy three criteria:

1. **Reversibility Criterion (Viability Floor):**  
   The intervention must be mathematically reversible without reloading fresh baseline weights from disk. If a network cannot theoretically be steered or perturbed back to coherence, it lies outside the coma spectrum (it is dead).
2. **Afference Preservation Criterion (The Input Gate):**  
   The network must continue to accept input stimuli into its initial embedding layers. A system that rejects input entirely is disconnected from observation.
3. **Perturbational Measurability Criterion (Non-Zero Metric Space):**  
   The internal latent space must retain non-zero geometric structure such that an injected perturbation $\delta$ produces a measurable (even if highly stereotyped or collapsed) latent trajectory.
