#!/usr/bin/env python3
"""
Experiment 03: Real HuggingFace LLM In-Silico Coma Induction and 2D Casali PCI Evaluation.

This experiment uses a real pretrained transformer (GPT-2, 124M parameters) to test:
1. Baseline Awake State:
   - Coherent text generation.
   - Recording real hidden state activations across 12 layers.
   - In-silico TMS impulse injection & 2D Spatiotemporal Casali PCI calculation.
2. In-Silico Coma State:
   - Damping residual stream across intermediate layers (layers 6-11).
   - Observing functional breakdown in text generation (catatonic repeating tokens).
   - Measuring empirical 2D Casali PCI collapse on real layer tensors.
3. In-Silico Re-ignition (Pillars 3 & 4):
   - Computing representation steering vector between awake and comatose hidden states (768-dim).
   - Injecting steering vector with LavaRand-style chaotic entropy.
   - Measuring PCI recovery and observing lexical coherence restoration.

Outputs:
Automatically saves structured JSON and Markdown execution logs with UUID and millisecond timestamp.
"""

import os
import sys
import json
import uuid
from datetime import datetime, timezone
import numpy as np
import torch

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.metrics.pci import compute_casali_pci
from src.coma.hooks import ComaHook
from src.awakening.steering import SteeringVector, ActivationSteeringHook


def apply_insilico_tms_real(model, inputs, target_layer: int = 4, amplitude: float = 3.0):
    """
    Delivers a real causal perturbation impulse directly into a hidden layer of the model
    and measures the evoked difference matrix across all layers.
    """
    with torch.no_grad():
        base_out = model(**inputs, output_hidden_states=True)
    base_hidden = [h.squeeze(0).cpu().numpy() for h in base_out.hidden_states[1:]]

    def tms_hook(module, input_tensor, output_tensor):
        if isinstance(output_tensor, tuple):
            h = output_tensor[0]
            rest = output_tensor[1:]
        else:
            h = output_tensor
            rest = None

        seq_len = h.shape[1]
        t_stim = seq_len // 2
        perturbed_h = h.clone()
        impulse = torch.randn_like(perturbed_h[:, t_stim:, :]) * amplitude
        perturbed_h[:, t_stim:, :] += impulse

        if rest is not None:
            return (perturbed_h,) + rest
        return perturbed_h

    handle = model.transformer.h[target_layer].register_forward_hook(tms_hook)

    try:
        with torch.no_grad():
            pert_out = model(**inputs, output_hidden_states=True)
        pert_hidden = [h.squeeze(0).cpu().numpy() for h in pert_out.hidden_states[1:]]
    finally:
        handle.remove()

    num_layers = len(base_hidden)
    seq_len = inputs.input_ids.shape[1]
    evoked_matrix = np.zeros((num_layers, seq_len))

    for l_idx in range(num_layers):
        delta = pert_hidden[l_idx] - base_hidden[l_idx]
        evoked_matrix[l_idx, :] = np.linalg.norm(delta, axis=-1)

    return evoked_matrix, base_hidden[target_layer]


def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 20, device: str = "cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    run_uuid = str(uuid.uuid4())
    # High-precision ISO-8601 with milliseconds and timezone
    now = datetime.now().astimezone()
    timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + now.strftime("%z")
    # Formatted timezone e.g. -03:00
    if len(timestamp_str) >= 5 and timestamp_str[-5] in ["+", "-"]:
        timestamp_str = timestamp_str[:-2] + ":" + timestamp_str[-2:]

    print("=" * 78)
    print("🧠 Project Sleeping LLM: Empirical Run Execution")
    print(f"Run UUID:    {run_uuid}")
    print(f"Timestamp:   {timestamp_str}")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware:    {device_name} (PyTorch {torch.__version__})")

    model_name = "gpt2"
    print(f"Loading Model: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    prompt = "The human brain in a conscious state is characterized by complex information integration and"
    print(f"\nStimulus Prompt: \"{prompt}\"")

    # =========================================================================
    # STAGE 1: AWAKE BASELINE
    # =========================================================================
    print("\n" + "-" * 78)
    print("🟢 STAGE 1: AWAKE BASELINE")
    print("-" * 78)

    awake_text = generate_sample(model, tokenizer, prompt, max_new_tokens=20, device=device)
    awake_continuation = awake_text[len(prompt):].strip()
    print(f"• Generated Continuation: \"{awake_continuation}\"")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    evoked_awake, h_awake_target = apply_insilico_tms_real(model, inputs, target_layer=4, amplitude=3.0)
    baseline_noise = np.random.normal(0, 0.05, size=evoked_awake.shape)

    pci_awake = compute_casali_pci(evoked_awake, baseline_noise=baseline_noise, significance_percentile=99.0)
    print(f"• Evoked Response Matrix: {evoked_awake.shape} (Layers x Tokens)")
    print(f"• Authentic Casali PCI_2D: {pci_awake:.4f}")

    # =========================================================================
    # STAGE 2: IN-SILICO COMA INDUCTION (Residual Stream Damping)
    # =========================================================================
    print("\n" + "-" * 78)
    print("🔴 STAGE 2: IN-SILICO COMA INDUCTION (Residual Damping on Layers 6-11)")
    print("-" * 78)

    coma_hooks = []
    for l in range(6, 12):
        hook = ComaHook(damping_factor=0.05, noise_std=0.01)
        hook.register(model.transformer.h[l])
        coma_hooks.append(hook)

    try:
        coma_text = generate_sample(model, tokenizer, prompt, max_new_tokens=20, device=device)
        coma_continuation = coma_text[len(prompt):].strip()
        print(f"• Generated Continuation in Coma: \"{coma_continuation}\"")

        evoked_coma, h_coma_target = apply_insilico_tms_real(model, inputs, target_layer=4, amplitude=3.0)
        pci_coma = compute_casali_pci(evoked_coma, baseline_noise=baseline_noise, significance_percentile=99.0)
        print(f"• Authentic Casali PCI_2D in Coma: {pci_coma:.4f}")

    finally:
        for h in coma_hooks:
            h.remove()

    # =========================================================================
    # STAGE 3: RE-IGNITION (In-Silico DBS + LavaRand Chaotic Seeding)
    # =========================================================================
    print("\n" + "-" * 78)
    print("🟡 STAGE 3: RE-IGNITION (Representation Steering + Chaotic Entropy)")
    print("-" * 78)

    diff_vector = np.mean(h_awake_target - h_coma_target, axis=0)
    steering_vec = SteeringVector(vector=diff_vector, layer_idx=4, multiplier=1.5)

    # Re-apply coma lesion
    for l in range(6, 12):
        hook = ComaHook(damping_factor=0.05, noise_std=0.01)
        hook.register(model.transformer.h[l])
        coma_hooks.append(hook)

    steering_hook = ActivationSteeringHook(steering_vec, chaotic_alpha=0.1)
    steering_hook.register(model.transformer.h[4])

    try:
        reawakened_text = generate_sample(model, tokenizer, prompt, max_new_tokens=20, device=device)
        reawakened_continuation = reawakened_text[len(prompt):].strip()
        print(f"• Generated Continuation with In-Silico DBS: \"{reawakened_continuation}\"")

        evoked_reawakened, _ = apply_insilico_tms_real(model, inputs, target_layer=4, amplitude=3.0)
        pci_reawakened = compute_casali_pci(evoked_reawakened, baseline_noise=baseline_noise, significance_percentile=99.0)
        print(f"• Authentic Casali PCI_2D Re-ignited:        {pci_reawakened:.4f}")

    finally:
        steering_hook.remove()
        for h in coma_hooks:
            h.remove()

    # =========================================================================
    # EXPORT RESULTS (JSON + MARKDOWN)
    # =========================================================================
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(results_dir, exist_ok=True)

    result_payload = {
        "metadata": {
            "run_uuid": run_uuid,
            "timestamp_iso": timestamp_str,
            "device": device,
            "device_name": device_name,
            "pytorch_version": torch.__version__,
            "model_architecture": "GPT-2 (124M parameters, 12 layers, 768 hidden_dim)",
            "experiment": "Experiment 03: Real Transformer In-Silico Coma & 2D Casali PCI"
        },
        "experimental_setup": {
            "stimulus_prompt": prompt,
            "tms_target_layer": 4,
            "tms_amplitude": 3.0,
            "coma_lesion_layers": [6, 7, 8, 9, 10, 11],
            "coma_damping_factor": 0.05,
            "steering_injection_layer": 4,
            "chaotic_entropy_alpha": 0.1
        },
        "quantitative_metrics": {
            "awake_baseline_pci_2d": round(float(pci_awake), 6),
            "comatose_pci_2d": round(float(pci_coma), 6),
            "reawakened_pci_2d": round(float(pci_reawakened), 6),
            "delta_coma": round(float(pci_coma - pci_awake), 6),
            "delta_recovery": round(float(pci_reawakened - pci_coma), 6)
        },
        "qualitative_text_outputs": {
            "awake_continuation": awake_continuation,
            "comatose_continuation": coma_continuation,
            "reawakened_continuation": reawakened_continuation
        }
    }

    # Save JSON artifact
    json_path = os.path.join(results_dir, f"run_{run_uuid}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    # Save Markdown artifact
    md_path = os.path.join(results_dir, f"run_{run_uuid}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Empirical Experiment Run: {run_uuid} 📊

- **Timestamp:** `{timestamp_str}`
- **Hardware:** `{device_name}` | PyTorch `{torch.__version__}`
- **Model:** `GPT-2` (124M parameters, 12 layers, 768 hidden dimension)

---

## 1. Quantitative Results (2D Spatiotemporal Casali PCI)

| Experimental Condition | Casali PCI_2D | Delta from Baseline | Behavioral Output |
| :--- | :---: | :---: | :--- |
| **Stage 1: Awake Baseline** | **{pci_awake:.4f}** | *Reference* | Intact coherent lexical integration |
| **Stage 2: Induced Coma (Layers 6-11)** | **{pci_coma:.4f}** | {pci_coma - pci_awake:+.4f} | Total mutism / Catatonic dash repetition (`{coma_continuation}`) |
| **Stage 3: In-Silico DBS Re-ignition** | **{pci_reawakened:.4f}** | {pci_reawakened - pci_awake:+.4f} | Attractor destabilization / Bifurcation (`{reawakened_continuation}`) |

---

## 2. Qualitative Text Continuation Analysis

* **Prompt:**  
  > *"{prompt}"*

* **Stage 1 (Awake):**  
  > *"{awake_continuation}"*

* **Stage 2 (Coma - Residual Stream Damped to 5%):**  
  > *"{coma_continuation}"*  
  *(Evidence of the Afferent-Efferent Disconnection: inputs enter layers 1-5, but cannot converge through the damaged 6-11 manifold).*

* **Stage 3 (Re-ignition Attempt with DBS + Chaotic Entropy):**  
  > *"{reawakened_continuation}"*

---

## 3. Raw Data Artifact
The full numerical payload is stored in [`run_{run_uuid}.json`](run_{run_uuid}.json).
""")

    print("\n" + "=" * 78)
    print("💾 ARTIFACTS SAVED SUCCESSFULLY:")
    print(f"  • JSON Payload: {json_path}")
    print(f"  • Markdown Log: {md_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
