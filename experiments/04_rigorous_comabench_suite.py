#!/usr/bin/env python3
"""
Experiment 04: Rigorous Empirical Benchmark Suite (ComaBench v0.3)
Addressing Academic Reviewer & Clinical Neurobiology Requirements:

1. Fine-Grained Dose-Response Alpha-Sweep:
   - 10 levels: alpha in [1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05, 0.0]
   - Quantitative evaluation of PCI(alpha), Perplexity(alpha), and Next-Token Accuracy(alpha).
2. Multi-Seed Replication:
   - Seeds [42, 123, 999] reporting mean +/- std for statistical significance.
3. Cross-Architecture Generalization:
   - Model 1: GPT-2 (124M, OpenAI architecture, absolute positional embeddings)
   - Model 2: Pythia-160M (160M, EleutherAI GPT-NeoX architecture, rotary embeddings / RoPE)
4. Rigorous Controls:
   - Early Damping (Layers 0-5) vs Late Damping (Layers 6-11)
   - Random Sham Steering vs Difference-in-Means Directional Steering
5. Clear Scientific Demarcation:
   - (a) Reproducible In-Silico Lesioning
   - (b) Semantic Recovery Failure (Limits of DBS)
   - (c) Dynamical Criticality & Edge of Chaos Hypothesis
"""

import os
import sys
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.metrics.pci import compute_casali_pci
from src.metrics.evoked import paired_trajectories
from src.metrics.criticality import (
    compute_layer_branching_ratios,
    compute_criticality_index,
    classify_dynamical_regime
)
from src.coma.hooks import ComaHook
from src.awakening.steering import SteeringVector, ActivationSteeringHook
from src.awakening.criticality import CriticalityLayer


PROMPT_BATTERY = [
    {
        "id": "domain_a_consciousness",
        "domain": "Cognitive Neuroscience & Consciousness",
        "prompt": "The human brain in a conscious state is characterized by complex information integration and",
        "continuation": " differentiated neural dynamics that enable subjective awareness, unified sensory perception, and cognitive flexibility across cortical networks."
    },
    {
        "id": "domain_b_neurobiology",
        "domain": "Mammalian Neurobiology & Thalamus",
        "prompt": "In the mammalian central nervous system, thalamocortical reverberation is crucial because",
        "continuation": " it synchronizes distant cortical regions, regulates sensory gating, and maintains the balanced excitation needed for waking consciousness."
    },
    {
        "id": "domain_c_logic",
        "domain": "Formal Logic & Deduction",
        "prompt": "To solve the complex deduction puzzle, the investigator examined the premise and determined that",
        "continuation": " since the initial conditions were mutually exclusive, the only logically consistent conclusion was that the second suspect had an alibi."
    }
]

ALPHA_SWEEP = [1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05, 0.0]
SEEDS = [42, 123, 999]


def get_model_layers(model):
    """Abstract accessor for transformer layer modules across architectures."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Unknown architecture: {type(model)}")


def extract_empirical_baseline_noise(model, inputs, seed: int = 999) -> Tuple[np.ndarray, float]:
    """
    Derives the empirical layer-by-layer baseline noise matrix from physical micro-jitter
    in the input embeddings, strictly satisfying the pre-stimulus noise paradigm of Casali et al. (2013).
    Normalized to 1% of mean embedding norm to remain invariant across architectures.
    """
    with torch.no_grad():
        base_out = model(**inputs, output_hidden_states=True)
    base_h = np.array([h.squeeze(0).float().cpu().numpy() for h in base_out.hidden_states[1:]], dtype=np.float32)

    emb_layer = model.get_input_embeddings()
    inputs_embeds = emb_layer(inputs.input_ids)
    mean_embed_norm = inputs_embeds.norm(dim=-1).mean().item()
    jitter_std = max(1e-4, 0.01 * mean_embed_norm)

    torch.manual_seed(seed)
    jitter = torch.randn_like(inputs_embeds) * jitter_std
    with torch.no_grad():
        noisy_out = model(inputs_embeds=inputs_embeds + jitter, output_hidden_states=True)
    noisy_h = np.array([h.squeeze(0).float().cpu().numpy() for h in noisy_out.hidden_states[1:]], dtype=np.float32)

    delta_baseline = np.linalg.norm(noisy_h - base_h, axis=-1)
    p99_threshold = float(np.percentile(delta_baseline, 99.0))
    return delta_baseline, p99_threshold


def measure_evoked_response(
    model,
    inputs,
    target_layer: int = 3,
    t_stim: int = 4,
    amplitude: float = 100.0,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Injects a focal in-silico TMS impulse into target_layer at token t_stim and measures
    the evoked differential tensor across all layers.
    """
    seq_len = inputs.input_ids.shape[1]
    base_h, pert_h = paired_trajectories(
        model, inputs, get_model_layers(model)[target_layer], t_stim, amplitude, seed
    )

    num_layers = len(base_h)
    evoked = np.zeros((num_layers, seq_len), dtype=np.float32)
    layer_norms = []
    for l in range(num_layers):
        delta = pert_h[l] - base_h[l]
        evoked[l, :] = np.linalg.norm(delta, axis=-1)
        layer_norms.append(float(np.linalg.norm(delta)))

    downstream_norms = layer_norms[target_layer+1:]
    ratios = compute_layer_branching_ratios(downstream_norms)
    kappa, mean_sigma, stability = compute_criticality_index(ratios)
    regime = classify_dynamical_regime(mean_sigma)

    crit_metrics = {
        "mean_branching_ratio": round(mean_sigma, 4),
        "criticality_index_kappa": round(kappa, 4),
        "critical_stability": round(stability, 4),
        "dynamical_regime": regime
    }
    return evoked, crit_metrics


def compute_language_metrics(model, full_inputs, prompt_len: int) -> Dict[str, float]:
    """Computes Cross-Entropy Loss, Perplexity, and Next-Token Accuracy on continuation tokens."""
    with torch.no_grad():
        out = model(**full_inputs)
        logits = out.logits.float()

    input_ids = full_inputs.input_ids
    shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
    shift_targets = input_ids[:, prompt_len:].contiguous()

    ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)).item()
    ppl = float(np.exp(ce_loss)) if ce_loss < 25.0 else float("inf")

    preds = torch.argmax(shift_logits, dim=-1)
    acc = (preds == shift_targets).float().mean().item()

    return {
        "ce_loss": round(ce_loss, 4),
        "perplexity": round(min(ppl, 1e8), 2),
        "accuracy": round(acc, 4)
    }


def generate_sample(model, tokenizer, inputs, prompt_text: str, max_tokens: int = 25, seed: int = 42) -> str:
    """Generates autoregressive sample continuation."""
    torch.manual_seed(seed)
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    return text[len(prompt_text):].strip()


def run_benchmark():
    run_uuid = str(uuid.uuid4())
    start_time = time.time()
    now = datetime.now().astimezone()
    timestamp_str = now.isoformat()

    print("=" * 84)
    print("🧠 Project Sleeping LLM: ComaBench v0.3 (Rigorous Empirical Suite)")
    print(f"Run UUID:    {run_uuid}")
    print(f"Timestamp:   {timestamp_str}")
    print("=" * 84)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware:    {device_name} (PyTorch {torch.__version__})")

    models_to_test = [
        {"name": "gpt2", "label": "GPT-2 (124M)", "type": "openai_gpt2", "num_layers": 12, "hidden_dim": 768},
        {"name": "EleutherAI/pythia-160m", "label": "Pythia-160M", "type": "neox_pythia", "num_layers": 12, "hidden_dim": 768}
    ]

    all_models_results = {}

    for model_cfg in models_to_test:
        m_name = model_cfg["name"]
        m_label = model_cfg["label"]
        print(f"\n{'#' * 84}")
        print(f"🔬 BENCHMARKING MODEL: {m_label} ('{m_name}')")
        print(f"{'#' * 84}")

        tokenizer = AutoTokenizer.from_pretrained(m_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(m_name).to(device)
        model.eval()
        layers = get_model_layers(model)
        total_layers = len(layers)

        # ---------------------------------------------------------------------
        # 1. FINE-GRAINED ALPHA-SWEEP ACROSS PROMPTS & SEEDS
        # ---------------------------------------------------------------------
        print(f"\n--- Part 1: Fine-Grained Alpha Dose-Response Sweep ({len(ALPHA_SWEEP)} levels x {len(SEEDS)} seeds) ---")
        
        sweep_data = {alpha: {"pci": [], "ce_loss": [], "ppl": [], "acc": [], "sigma": [], "kappa": []} for alpha in ALPHA_SWEEP}
        sample_generations = {alpha: "" for alpha in ALPHA_SWEEP}

        for p_idx, prompt_info in enumerate(PROMPT_BATTERY):
            p_text = prompt_info["prompt"]
            full_text = p_text + prompt_info["continuation"]
            
            inputs = tokenizer(p_text, return_tensors="pt").to(device)
            full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]
            seq_len = inputs.input_ids.shape[1]
            t_stim = max(3, seq_len // 3)

            # Extract data-driven non-parametric baseline noise floor
            delta_baseline, p99_th = extract_empirical_baseline_noise(model, inputs)

            # Unperturbed awake representations
            with torch.no_grad():
                base_out = model(**inputs, output_hidden_states=True)
            base_h = np.array([h.squeeze(0).float().cpu().numpy() for h in base_out.hidden_states[1:]], dtype=np.float32)

            # Calibrate TMS amplitude to 2.5x mean activation norm at target layer 3
            l3_norm = float(np.linalg.norm(base_h[3], axis=-1).mean())
            calibrated_amp = 2.5 * l3_norm

            for alpha in ALPHA_SWEEP:
                # Coma hooks on late half (layers 6-11)
                half = total_layers // 2
                hooks = [ComaHook(damping_factor=alpha) for _ in range(total_layers - half)]
                for i, h in enumerate(hooks):
                    h.register(layers[half + i])

                try:
                    for seed in SEEDS:
                        evoked, crit = measure_evoked_response(
                            model, inputs, target_layer=3, t_stim=t_stim, amplitude=calibrated_amp, seed=seed
                        )
                        pci_val = compute_casali_pci(evoked, baseline_noise=delta_baseline, significance_percentile=99.0)
                        lang = compute_language_metrics(model, full_inputs, prompt_len)

                        sweep_data[alpha]["pci"].append(pci_val)
                        sweep_data[alpha]["ce_loss"].append(lang["ce_loss"])
                        sweep_data[alpha]["ppl"].append(lang["perplexity"])
                        sweep_data[alpha]["acc"].append(lang["accuracy"])
                        sweep_data[alpha]["sigma"].append(crit["mean_branching_ratio"])
                        sweep_data[alpha]["kappa"].append(crit["criticality_index_kappa"])

                    # Save sample generation for seed 42 on first prompt
                    if p_idx == 0:
                        sample_generations[alpha] = generate_sample(model, tokenizer, inputs, p_text, max_tokens=20, seed=42)

                finally:
                    for h in hooks:
                        h.remove()

        # Compute summary statistics (mean +/- std) per alpha
        alpha_summary = []
        for alpha in ALPHA_SWEEP:
            pci_arr = np.array(sweep_data[alpha]["pci"])
            ppl_arr = np.array(sweep_data[alpha]["ppl"])
            acc_arr = np.array(sweep_data[alpha]["acc"])
            loss_arr = np.array(sweep_data[alpha]["ce_loss"])
            sig_arr = np.array(sweep_data[alpha]["sigma"])

            alpha_summary.append({
                "alpha": alpha,
                "pci_mean": round(float(np.mean(pci_arr)), 4),
                "pci_std": round(float(np.std(pci_arr)), 4),
                "loss_mean": round(float(np.mean(loss_arr)), 4),
                "ppl_mean": round(float(np.mean(ppl_arr)), 2),
                "acc_mean": round(float(np.mean(acc_arr)), 4),
                "acc_std": round(float(np.std(acc_arr)), 4),
                "branching_mean": round(float(np.mean(sig_arr)), 4),
                "sample_generation": sample_generations[alpha]
            })

        print(f"| Alpha | PCI_2D (mean+/-std) | Perplexity | Accuracy | Branching σ | Sample Text |")
        print(f"| :---: | :---: | :---: | :---: | :---: | :--- |")
        for s in alpha_summary:
            print(f"| {s['alpha']:<5.2f} | {s['pci_mean']:.4f} +/- {s['pci_std']:.4f} | {s['ppl_mean']:<10.1f} | {s['acc_mean']:.3f} | {s['branching_mean']:.3f} | \"{s['sample_generation'][:30]}\" |")

        # ---------------------------------------------------------------------
        # 2. CONTROLS: EARLY VS LATE DAMPING
        # ---------------------------------------------------------------------
        print(f"\n--- Part 2: Anatomical Control: Early Damping (L0-5) vs Late Damping (L6-11) ---")
        prompt_test = PROMPT_BATTERY[0]
        inputs = tokenizer(prompt_test["prompt"], return_tensors="pt").to(device)
        full_inputs = tokenizer(prompt_test["prompt"] + prompt_test["continuation"], return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        seq_len = inputs.input_ids.shape[1]
        t_stim = max(3, seq_len // 3)

        delta_baseline, _ = extract_empirical_baseline_noise(model, inputs)
        with torch.no_grad():
            base_out = model(**inputs, output_hidden_states=True)
        base_h = np.array([h.squeeze(0).float().cpu().numpy() for h in base_out.hidden_states[1:]], dtype=np.float32)
        l3_norm = float(np.linalg.norm(base_h[3], axis=-1).mean())
        calibrated_amp = 2.5 * l3_norm

        # A. Awake Baseline
        ev_awake, crit_awake = measure_evoked_response(model, inputs, target_layer=3, t_stim=t_stim, amplitude=calibrated_amp)
        pci_ref_awake = compute_casali_pci(ev_awake, baseline_noise=delta_baseline, significance_percentile=99.0)
        lang_ref_awake = compute_language_metrics(model, full_inputs, prompt_len)
        text_ref_awake = generate_sample(model, tokenizer, inputs, prompt_test["prompt"], seed=42)

        # B. Early Damping (Layers 0-5) with alpha=0.05
        early_hooks = [ComaHook(damping_factor=0.05) for _ in range(6)]
        for i, h in enumerate(early_hooks):
            h.register(layers[i])
        try:
            ev_early, crit_early = measure_evoked_response(model, inputs, target_layer=3, t_stim=t_stim, amplitude=calibrated_amp)
            pci_early = compute_casali_pci(ev_early, baseline_noise=delta_baseline, significance_percentile=99.0)
            lang_early = compute_language_metrics(model, full_inputs, prompt_len)
            text_early = generate_sample(model, tokenizer, inputs, prompt_test["prompt"], seed=42)
        finally:
            for h in early_hooks:
                h.remove()

        # C. Late Damping (Layers 6-11) with alpha=0.05
        late_hooks = [ComaHook(damping_factor=0.05) for _ in range(6)]
        for i, h in enumerate(late_hooks):
            h.register(layers[6 + i])
        try:
            ev_late, crit_late = measure_evoked_response(model, inputs, target_layer=3, t_stim=t_stim, amplitude=calibrated_amp)
            pci_late = compute_casali_pci(ev_late, baseline_noise=delta_baseline, significance_percentile=99.0)
            lang_late = compute_language_metrics(model, full_inputs, prompt_len)
            text_late = generate_sample(model, tokenizer, inputs, prompt_test["prompt"], seed=42)
        finally:
            for h in late_hooks:
                h.remove()

        early_late_control = {
            "awake": {"pci": round(pci_ref_awake, 4), "ppl": lang_ref_awake["perplexity"], "acc": lang_ref_awake["accuracy"], "text": text_ref_awake},
            "early_damping_L0_5": {"pci": round(pci_early, 4), "ppl": lang_early["perplexity"], "acc": lang_early["accuracy"], "text": text_early},
            "late_damping_L6_11": {"pci": round(pci_late, 4), "ppl": lang_late["perplexity"], "acc": lang_late["accuracy"], "text": text_late}
        }
        print(f"  • Awake Ref:     PCI={pci_ref_awake:.4f} | PPL={lang_ref_awake['perplexity']} | Acc={lang_ref_awake['accuracy']} | \"{text_ref_awake[:30]}...\"")
        print(f"  • Early (L0-5):  PCI={pci_early:.4f} | PPL={lang_early['perplexity']} | Acc={lang_early['accuracy']} | \"{text_early[:30]}...\"")
        print(f"  • Late (L6-11):  PCI={pci_late:.4f} | PPL={lang_late['perplexity']} | Acc={lang_late['accuracy']} | \"{text_late[:30]}...\"")

        # ---------------------------------------------------------------------
        # 3. CONTROLS: DIFFERENCE-IN-MEANS VS RANDOM SHAM STEERING
        # ---------------------------------------------------------------------
        print(f"\n--- Part 3: Neuromodulation Control: Difference-in-Means vs Random Sham Vector ---")
        late_hooks = [ComaHook(damping_factor=0.05) for _ in range(6)]
        for i, h in enumerate(late_hooks):
            h.register(layers[6 + i])

        try:
            # Baseline Coma
            ev_coma, crit_coma = measure_evoked_response(model, inputs, target_layer=3, t_stim=t_stim, amplitude=calibrated_amp)
            pci_coma = compute_casali_pci(ev_coma, baseline_noise=delta_baseline, significance_percentile=99.0)
            text_coma = generate_sample(model, tokenizer, inputs, prompt_test["prompt"], seed=42)

            # Steering Vector (Difference-in-means: mean awake activation at Layer 4)
            steer_v = base_h[4].mean(axis=0)
            norm_v = float(np.linalg.norm(steer_v))

            # Directional Steering
            svec_dir = SteeringVector(vector=steer_v, layer_idx=4, multiplier=1.5)
            shk_dir = ActivationSteeringHook(svec_dir, chaotic_alpha=0.0)
            shk_dir.register(layers[4])
            try:
                ev_dir, crit_dir = measure_evoked_response(model, inputs, target_layer=3, t_stim=t_stim, amplitude=calibrated_amp)
                pci_dir = compute_casali_pci(ev_dir, baseline_noise=delta_baseline, significance_percentile=99.0)
                text_dir = generate_sample(model, tokenizer, inputs, prompt_test["prompt"], seed=42)
            finally:
                shk_dir.remove()

            # Random Sham Vector (norm-matched to steer_v * multiplier)
            torch.manual_seed(777)
            rand_v = torch.randn(steer_v.shape[0])
            rand_v = (rand_v / torch.norm(rand_v)).numpy() * (norm_v * 1.5)
            svec_sham = SteeringVector(vector=rand_v, layer_idx=4, multiplier=1.0)
            shk_sham = ActivationSteeringHook(svec_sham, chaotic_alpha=0.0)
            shk_sham.register(layers[4])
            try:
                ev_sham, crit_sham = measure_evoked_response(model, inputs, target_layer=3, t_stim=t_stim, amplitude=calibrated_amp)
                pci_sham = compute_casali_pci(ev_sham, baseline_noise=delta_baseline, significance_percentile=99.0)
                text_sham = generate_sample(model, tokenizer, inputs, prompt_test["prompt"], seed=42)
            finally:
                shk_sham.remove()

            # Criticality Layer (L5 gateway)
            crit_layer = CriticalityLayer(hidden_dim=model_cfg["hidden_dim"], homeostatic_gain=1.25, stochastic_resonance_std=0.01)
            crit_layer.register(layers[5])
            try:
                ev_crit, crit_mod = measure_evoked_response(model, inputs, target_layer=3, t_stim=t_stim, amplitude=calibrated_amp)
                pci_crit = compute_casali_pci(ev_crit, baseline_noise=delta_baseline, significance_percentile=99.0)
                text_crit = generate_sample(model, tokenizer, inputs, prompt_test["prompt"], seed=42)
            finally:
                crit_layer.remove()

        finally:
            for h in late_hooks:
                h.remove()

        neuromod_controls = {
            "unassisted_coma": {"pci": round(pci_coma, 4), "text": text_coma, "sigma": crit_coma["mean_branching_ratio"]},
            "random_sham_vector": {"pci": round(pci_sham, 4), "text": text_sham, "sigma": crit_sham["mean_branching_ratio"]},
            "directional_diff_in_means": {"pci": round(pci_dir, 4), "text": text_dir, "sigma": crit_dir["mean_branching_ratio"]},
            "criticality_layer": {"pci": round(pci_crit, 4), "text": text_crit, "sigma": crit_mod["mean_branching_ratio"]}
        }
        print(f"  • Unassisted Coma: PCI={pci_coma:.4f} | σ={crit_coma['mean_branching_ratio']:.3f} | \"{text_coma[:30]}...\"")
        print(f"  • Random Sham:     PCI={pci_sham:.4f} | σ={crit_sham['mean_branching_ratio']:.3f} | \"{text_sham[:30]}...\"")
        print(f"  • Directional Vec: PCI={pci_dir:.4f} | σ={crit_dir['mean_branching_ratio']:.3f} | \"{text_dir[:30]}...\"")
        print(f"  • Criticality L5:  PCI={pci_crit:.4f} | σ={crit_mod['mean_branching_ratio']:.3f} | \"{text_crit[:30]}...\"")

        all_models_results[m_name] = {
            "config": model_cfg,
            "alpha_summary": alpha_summary,
            "early_late_control": early_late_control,
            "neuromod_controls": neuromod_controls
        }

    # -------------------------------------------------------------------------
    # EXPORT BENCHMARK ARTIFACTS
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(results_dir, exist_ok=True)

    payload = {
        "metadata": {
            "run_uuid": run_uuid,
            "timestamp_iso": timestamp_str,
            "duration_seconds": round(duration, 2),
            "benchmark_version": "ComaBench v0.3 (Rigorous Empirical Suite)",
            "hardware": {
                "device": device,
                "device_name": device_name,
                "pytorch_version": torch.__version__
            },
            "parameters": {
                "alpha_grid": ALPHA_SWEEP,
                "seeds": SEEDS,
                "tms_amplitude_ratio": 2.5,
                "noise_jitter_ratio": 0.01,
                "significance_percentile": 99.0
            }
        },
        "models": all_models_results
    }

    json_path = os.path.join(results_dir, f"run_comabench_v03_{run_uuid}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 84)
    print("💾 BENCHMARK COMPLETE & JSON SAVED:")
    print(f"  • JSON File: {json_path}")
    print(f"  • Total Duration: {duration:.1f}s")
    print("=" * 84)

    return payload, json_path


if __name__ == "__main__":
    run_benchmark()
