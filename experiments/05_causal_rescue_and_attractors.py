#!/usr/bin/env python3
"""
Experiment 05: Mechanistic Analysis of the Hyphen Attractor,
Intervention Types (State vs. Update Damping), and Causal Rescue via Activation Patching.

Directly addresses peer-review questions:
1. Intervention Type A vs B:
   - Type A: Full state damping: h_{l+1} = alpha * Block(h_l)
   - Type B: Update branch damping: h_{l+1} = h_l + alpha * F_l(LN(h_l)) (attn & mlp outputs)
2. Mechanistic Diagnostics of the Repetitive Attractor ('-'):
   - Step-by-step Shannon entropy of logits H(t)
   - Margin between top-1 and top-2 logits (z_{(1)} - z_{(2)})
   - Rank and probability of the hyphen token '-'
   - Generation dynamics: Greedy (T=0.0) vs. Temperature Sampling (T=0.7, T=1.2)
3. Causal Rescue via Dynamic Activation Patching:
   - Dynamic caching of clean awake residual activations
   - Patching at severed layers to test restoration of NLL and Top-1 accuracy
   - Unseen prompt generalization
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.coma.hooks import ComaHook
from src.awakening.steering import SteeringVector, ActivationSteeringHook


PROMPT_TRAIN = {
    "prompt": "The human brain in a conscious state is characterized by complex information integration and",
    "continuation": " differentiated neural dynamics that enable subjective awareness and unified perception."
}

PROMPT_UNSEEN = {
    "prompt": "In theoretical physics, dynamical systems poised near the edge of chaos maximize computational",
    "continuation": " storage and long-range information transmission across distributed interacting networks."
}


def get_sublayer_modules(model):
    """Returns lists of attention and mlp submodules across architectures."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        attn_mods = [b.attn for b in blocks]
        mlp_mods = [b.mlp for b in blocks]
        return blocks, attn_mods, mlp_mods
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        blocks = model.gpt_neox.layers
        attn_mods = [b.attention for b in blocks]
        mlp_mods = [b.mlp for b in blocks]
        return blocks, attn_mods, mlp_mods
    raise ValueError(f"Unsupported model architecture: {type(model)}")


def compute_nll_metrics(model, full_inputs, prompt_len: int) -> Dict[str, float]:
    """Computes bounded teacher-forced NLL, Delta NLL, and Top-1 / Top-5 Accuracy."""
    with torch.no_grad():
        out = model(**full_inputs)
        logits = out.logits.float()

    input_ids = full_inputs.input_ids
    shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
    shift_targets = input_ids[:, prompt_len:].contiguous()

    # NLL / Cross Entropy
    nll = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)).item()
    ppl = float(np.exp(nll)) if nll < 25.0 else float("inf")

    # Top-1 Accuracy
    preds = torch.argmax(shift_logits, dim=-1)
    acc_top1 = (preds == shift_targets).float().mean().item()

    # Top-5 Accuracy
    top5_preds = torch.topk(shift_logits, k=5, dim=-1).indices
    expanded_targets = shift_targets.unsqueeze(-1).expand_as(top5_preds)
    acc_top5 = (top5_preds == expanded_targets).any(dim=-1).float().mean().item()

    # Logit Entropy
    probs = F.softmax(shift_logits, dim=-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()

    return {
        "nll": round(nll, 4),
        "ppl": round(min(ppl, 1e8), 2),
        "top1_acc": round(acc_top1, 4),
        "top5_acc": round(acc_top5, 4),
        "mean_logit_entropy": round(entropy, 4)
    }


def track_generation_dynamics(
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int = 20,
    decoding_mode: str = "greedy",
    temperature: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Step-by-step tracking of logit entropy, top-1 margin, and hyphen rank during autoregressive generation.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    hyphen_id = tokenizer.encode("-")[-1]

    torch.manual_seed(seed)
    current_ids = input_ids.clone()
    step_records = []
    generated_tokens = []

    with torch.no_grad():
        for step in range(max_tokens):
            out = model(current_ids, use_cache=False)
            next_logits = out.logits[:, -1, :].float()

            # 1. Shannon Entropy
            probs = F.softmax(next_logits, dim=-1)
            log_probs = F.log_softmax(next_logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).item()

            # 2. Margin between top-1 and top-2
            top2_vals, top2_indices = torch.topk(next_logits, k=2, dim=-1)
            margin = (top2_vals[0, 0] - top2_vals[0, 1]).item()

            # 3. Hyphen token stats
            hyphen_prob = probs[0, hyphen_id].item()
            sorted_indices = torch.argsort(next_logits[0], descending=True)
            hyphen_rank = int((sorted_indices == hyphen_id).nonzero(as_tuple=True)[0].item()) + 1

            # Select next token
            if decoding_mode == "greedy":
                next_token = top2_indices[0, 0].view(1, 1)
            else:
                scaled_logits = next_logits / max(temperature, 1e-5)
                sample_probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(sample_probs, num_samples=1)

            tok_str = tokenizer.decode(next_token[0])
            generated_tokens.append(tok_str)

            step_records.append({
                "step": step + 1,
                "token": tok_str,
                "entropy_bits": round(entropy / np.log(2), 3),
                "margin": round(margin, 3),
                "hyphen_prob": round(hyphen_prob, 5),
                "hyphen_rank": hyphen_rank
            })

            current_ids = torch.cat([current_ids, next_token], dim=-1)

    return {
        "text": "".join(generated_tokens),
        "steps": step_records
    }


def run_experiment_05():
    run_uuid = str(uuid.uuid4())
    start_time = time.time()
    now = datetime.now().astimezone()
    timestamp_str = now.isoformat()

    print("=" * 84)
    print("🔬 Experiment 05: Mechanistic Attractor Analysis & Causal Rescue")
    print(f"Run UUID:    {run_uuid}")
    print(f"Timestamp:   {timestamp_str}")
    print("=" * 84)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware:    {device_name} (PyTorch {torch.__version__})")

    model_name = "gpt2"
    print(f"Loading Model: '{model_name}' (124M parameters)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    blocks, attn_mods, mlp_mods = get_sublayer_modules(model)
    total_layers = len(blocks)
    half = total_layers // 2  # Layers 6..11

    # Prepare inputs for Train and Unseen
    t_prompt = PROMPT_TRAIN["prompt"]
    t_full = PROMPT_TRAIN["prompt"] + PROMPT_TRAIN["continuation"]
    train_full_inp = tokenizer(t_full, return_tensors="pt").to(device)
    train_prompt_len = tokenizer(t_prompt, return_tensors="pt").input_ids.shape[1]

    # Baseline Awake Metrics
    awake_metrics = compute_nll_metrics(model, train_full_inp, train_prompt_len)
    base_nll = awake_metrics["nll"]
    print(f"\n[Awake Intact Model] NLL = {base_nll:.4f} | PPL = {awake_metrics['ppl']} | Top1 = {awake_metrics['top1_acc']*100:.1f}%")

    # -------------------------------------------------------------------------
    # PART 1: INTERVENTION A (STATE) VS INTERVENTION B (UPDATE BRANCH)
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 84}")
    print("🧪 PART 1: Intervention A (State Damping) vs Intervention B (Update Branch Damping)")
    print(f"{'#' * 84}")

    alpha_levels = [1.0, 0.70, 0.50, 0.20, 0.05, 0.0]
    comparison_records = []

    for alpha in alpha_levels:
        # A. Type A: Full state damping on blocks 6..11
        hooks_a = [ComaHook(damping_factor=alpha) for _ in range(total_layers - half)]
        for i, h in enumerate(hooks_a):
            h.register(blocks[half + i])
        try:
            metrics_a = compute_nll_metrics(model, train_full_inp, train_prompt_len)
            sample_a = track_generation_dynamics(model, tokenizer, t_prompt, max_tokens=15, decoding_mode="greedy")["text"]
        finally:
            for h in hooks_a: h.remove()

        # B. Type B: Update branch damping (attn & mlp sublayers of blocks 6..11)
        hooks_b = []
        for i in range(half, total_layers):
            h_attn = ComaHook(damping_factor=alpha)
            h_attn.register(attn_mods[i])
            h_mlp = ComaHook(damping_factor=alpha)
            h_mlp.register(mlp_mods[i])
            hooks_b.extend([h_attn, h_mlp])
        try:
            metrics_b = compute_nll_metrics(model, train_full_inp, train_prompt_len)
            sample_b = track_generation_dynamics(model, tokenizer, t_prompt, max_tokens=15, decoding_mode="greedy")["text"]
        finally:
            for h in hooks_b: h.remove()

        delta_nll_a = round(metrics_a["nll"] - base_nll, 4)
        delta_nll_b = round(metrics_b["nll"] - base_nll, 4)

        comparison_records.append({
            "alpha": alpha,
            "type_a_state": {"nll": metrics_a["nll"], "delta_nll": delta_nll_a, "top1": metrics_a["top1_acc"], "text": sample_a},
            "type_b_update": {"nll": metrics_b["nll"], "delta_nll": delta_nll_b, "top1": metrics_b["top1_acc"], "text": sample_b}
        })

        print(f"  α = {alpha:<4.2f} | Type A: NLL={metrics_a['nll']:.2f} (Δ={delta_nll_a:+.2f}) Top1={metrics_a['top1_acc']*100:<4.1f}% | \"{sample_a[:22]}\"")
        print(f"         | Type B: NLL={metrics_b['nll']:.2f} (Δ={delta_nll_b:+.2f}) Top1={metrics_b['top1_acc']*100:<4.1f}% | \"{sample_b[:22]}\"")

    # -------------------------------------------------------------------------
    # PART 2: MECHANISTIC ATTRACTOR DIAGNOSTICS (GREEDY VS SAMPLING)
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 84}")
    print("📊 PART 2: Hyphen Attractor Dynamics (Greedy vs Temperature Sampling at α=0.05)")
    print(f"{'#' * 84}")

    # Induce Coma (Type A at alpha=0.05)
    hooks_coma = [ComaHook(damping_factor=0.05) for _ in range(total_layers - half)]
    for i, h in enumerate(hooks_coma): h.register(blocks[half + i])

    attractor_results = {}
    try:
        modes = [
            {"mode": "greedy", "temp": 0.0, "label": "Greedy (T=0.0)"},
            {"mode": "sampling", "temp": 0.7, "label": "Sampling (T=0.7)"},
            {"mode": "sampling", "temp": 1.2, "label": "High Temp (T=1.2)"}
        ]

        for m_info in modes:
            gen_data = track_generation_dynamics(
                model, tokenizer, t_prompt, max_tokens=15,
                decoding_mode=m_info["mode"], temperature=m_info["temp"], seed=42
            )
            attractor_results[m_info["label"]] = gen_data
            steps_preview = gen_data["steps"][:5]
            mean_ent = np.mean([s["entropy_bits"] for s in gen_data["steps"]])
            mean_margin = np.mean([s["margin"] for s in gen_data["steps"]])
            hyphen_pct = sum(1 for s in gen_data["steps"] if "-" in s["token"]) / len(gen_data["steps"]) * 100

            print(f"\n  [{m_info['label']}] Text: \"{gen_data['text']}\"")
            print(f"    • Mean Logit Entropy: {mean_ent:.2f} bits | Top-1 Margin: {mean_margin:.2f} | Hyphen Frequency: {hyphen_pct:.1f}%")
            trace_strs = [s['token'] + " (r=" + str(s['hyphen_rank']) + ")" for s in steps_preview]
            print(f"    • Initial Step Trace: {trace_strs}")

    finally:
        for h in hooks_coma: h.remove()

    # -------------------------------------------------------------------------
    # PART 3: CAUSAL RESCUE VIA DYNAMIC ACTIVATION PATCHING
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 84}")
    print("🩺 PART 3: Causal Rescue (Dynamic Activation Patching vs Static Steering)")
    print(f"{'#' * 84}")

    def evaluate_rescue_protocol(prompt_dict, test_label: str):
        p_txt = prompt_dict["prompt"]
        f_txt = prompt_dict["prompt"] + prompt_dict["continuation"]
        full_inp = tokenizer(f_txt, return_tensors="pt").to(device)
        p_len = tokenizer(p_txt, return_tensors="pt").input_ids.shape[1]

        # 1. Awake clean pass to extract ground-truth hidden states
        with torch.no_grad():
            clean_out = model(**full_inp, output_hidden_states=True)
        # hidden_states[1..12] are post-block outputs
        clean_states = [clean_out.hidden_states[l + 1].clone() for l in range(total_layers)]
        awake_m = compute_nll_metrics(model, full_inp, p_len)

        # 2. Coma unassisted (alpha=0.05 on layers 6..11)
        hooks_c = [ComaHook(damping_factor=0.05) for _ in range(total_layers - half)]
        for i, h in enumerate(hooks_c): h.register(blocks[half + i])
        try:
            coma_m = compute_nll_metrics(model, full_inp, p_len)
            text_coma = track_generation_dynamics(model, tokenizer, p_txt, max_tokens=15, decoding_mode="greedy")["text"]

            # 3. Static Steering Vector (Layer 4 difference-in-means)
            mean_awake_l4 = clean_states[4].squeeze(0).mean(dim=0).cpu().numpy()
            svec = SteeringVector(vector=mean_awake_l4, layer_idx=4, multiplier=1.5)
            shk = ActivationSteeringHook(svec, chaotic_alpha=0.0)
            shk.register(blocks[4])
            try:
                steer_m = compute_nll_metrics(model, full_inp, p_len)
                text_steer = track_generation_dynamics(model, tokenizer, p_txt, max_tokens=15, decoding_mode="greedy")["text"]
            finally:
                shk.remove()

            # 4. Single-Layer Causal Patch (Layer 6 restored with clean dynamic activations)
            def patch_l6_hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                seq_dim = h.shape[1]
                if seq_dim <= clean_states[6].shape[1]:
                    patched_h = clean_states[6][:, :seq_dim, :].to(h.device)
                else:
                    patched_h = h
                if isinstance(out, tuple):
                    return (patched_h,) + out[1:]
                return patched_h

            h_patch6 = blocks[6].register_forward_hook(patch_l6_hook)
            try:
                patch6_m = compute_nll_metrics(model, full_inp, p_len)
                text_patch6 = track_generation_dynamics(model, tokenizer, p_txt, max_tokens=15, decoding_mode="greedy")["text"]
            finally:
                h_patch6.remove()

            # 5. Full Causal Patching (Restoring clean updates across all severed layers 6..11)
            patch_handles = []
            for lyr in range(half, total_layers):
                def make_patch(l_idx):
                    def p_hook(mod, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        seq_dim = h.shape[1]
                        if seq_dim <= clean_states[l_idx].shape[1]:
                            ph = clean_states[l_idx][:, :seq_dim, :].to(h.device)
                        else:
                            ph = h
                        return (ph,) + out[1:] if isinstance(out, tuple) else ph
                    return p_hook
                hnd = blocks[lyr].register_forward_hook(make_patch(lyr))
                patch_handles.append(hnd)
            try:
                fullpatch_m = compute_nll_metrics(model, full_inp, p_len)
                text_fullpatch = track_generation_dynamics(model, tokenizer, p_txt, max_tokens=15, decoding_mode="greedy")["text"]
            finally:
                for ph in patch_handles: ph.remove()

        finally:
            for h in hooks_c: h.remove()

        return {
            "test_label": test_label,
            "prompt": p_txt,
            "awake": {**awake_m, "delta_nll": 0.0},
            "coma": {**coma_m, "delta_nll": round(coma_m["nll"] - awake_m["nll"], 4), "text": text_coma},
            "static_steer": {**steer_m, "delta_nll": round(steer_m["nll"] - awake_m["nll"], 4), "text": text_steer},
            "causal_patch_l6": {**patch6_m, "delta_nll": round(patch6_m["nll"] - awake_m["nll"], 4), "text": text_patch6},
            "causal_patch_all_severed": {**fullpatch_m, "delta_nll": round(fullpatch_m["nll"] - awake_m["nll"], 4), "text": text_fullpatch}
        }

    train_rescue = evaluate_rescue_protocol(PROMPT_TRAIN, "Train Domain Prompt")
    unseen_rescue = evaluate_rescue_protocol(PROMPT_UNSEEN, "Unseen Physics Domain Prompt")

    print(f"\n--- Rescue Results on: [{train_rescue['test_label']}] ---")
    for cond_k in ["awake", "coma", "static_steer", "causal_patch_l6", "causal_patch_all_severed"]:
        res = train_rescue[cond_k]
        txt = res.get("text", "N/A (Reference)")
        print(f"  • {cond_k:<25} | NLL={res['nll']:<5.2f} (Δ={res['delta_nll']:+5.2f}) | Top1={res['top1_acc']*100:<4.1f}% | \"{txt[:25]}\"")

    print(f"\n--- Rescue Results on: [{unseen_rescue['test_label']}] ---")
    for cond_k in ["awake", "coma", "static_steer", "causal_patch_l6", "causal_patch_all_severed"]:
        res = unseen_rescue[cond_k]
        txt = res.get("text", "N/A (Reference)")
        print(f"  • {cond_k:<25} | NLL={res['nll']:<5.2f} (Δ={res['delta_nll']:+5.2f}) | Top1={res['top1_acc']*100:<4.1f}% | \"{txt[:25]}\"")

    # -------------------------------------------------------------------------
    # EXPORT ARTIFACTS
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    payload = {
        "metadata": {
            "run_uuid": run_uuid,
            "timestamp_iso": timestamp_str,
            "duration_seconds": round(duration, 2),
            "experiment": "Experiment 05 (Mechanistic Attractor Analysis & Causal Rescue)",
            "model": "gpt2 (124M)",
            "hardware": device_name
        },
        "part1_intervention_comparison": comparison_records,
        "part2_attractor_dynamics": attractor_results,
        "part3_causal_rescue": {
            "train_domain": train_rescue,
            "unseen_domain": unseen_rescue
        }
    }

    json_path = os.path.join(results_dir, f"run_exp05_causal_rescue_{run_uuid}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 84)
    print("💾 EXPERIMENT 05 COMPLETE & JSON SAVED:")
    print(f"  • File: {json_path}")
    print(f"  • Total Execution Duration: {duration:.2f}s")
    print("=" * 84)


if __name__ == "__main__":
    run_experiment_05()
