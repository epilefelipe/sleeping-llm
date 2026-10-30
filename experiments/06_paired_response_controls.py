"""Check intervention-matched response subtraction on a real language model."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coma.hooks import ComaHook
from src.metrics.evoked import paired_trajectories
from src.metrics.pci import compute_casali_pci


def hidden_states(model, inputs):
    with torch.inference_mode():
        result = model(**inputs, output_hidden_states=True, use_cache=False)
    return np.stack([state[0].float().cpu().numpy() for state in result.hidden_states[1:]])


def continuation_metrics(model, inputs, prompt_length):
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[:, prompt_length - 1:-1].float()
    target = inputs['input_ids'][:, prompt_length:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    accuracy = (logits.argmax(-1) == target).float().mean()
    return {'nll': loss.item(), 'accuracy': accuracy.item(), 'tokens': target.numel()}


def summarize(matrix, noise):
    return {
        'norm': float(np.linalg.norm(matrix)),
        'max_norm': float(matrix.max()),
        'complexity': compute_casali_pci(matrix, baseline_noise=noise),
    }


def run(args):
    config = json.loads(args.config.read_text())
    prompts = config['prompts']
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=args.local_only, dtype=torch.bfloat16,
        attn_implementation='sdpa'
    ).to(args.device).eval()
    layers = model.model.layers
    started = time.perf_counter()
    metadata = {
        'model': config['model'], 'configuration': config,
        'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'device': torch.cuda.get_device_name(), 'dtype': 'bfloat16',
        'seed': args.seed, 'started_at': datetime.now(timezone.utc).isoformat(),
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for prompt in prompts:
        inputs = tokenizer(prompt['prompt'], return_tensors='pt').to(args.device)
        full = tokenizer(prompt['prompt'] + prompt['continuation'], return_tensors='pt').to(args.device)
        length = inputs['input_ids'].shape[1]
        intact = hidden_states(model, inputs)
        amplitude = config['amplitude_multiplier'] * np.linalg.norm(
            intact[config['target_layer']], axis=-1
        ).mean()
        embeddings = model.get_input_embeddings()(inputs['input_ids'])
        jitter_std = config['jitter_fraction'] * embeddings.norm(dim=-1).mean().item()
        generator = torch.Generator(device=args.device).manual_seed(config['noise_seed'])
        jitter = torch.randn(embeddings.shape, generator=generator, device=args.device, dtype=embeddings.dtype)
        noisy = hidden_states(model, {'inputs_embeds': embeddings + jitter * jitter_std})
        noise = np.linalg.norm(noisy - intact, axis=-1)
        token_index = max(3, length // 3)
        for alpha in config['alpha']:
            stem = f"{prompt['id']}_alpha-{alpha:g}_seed-{args.seed}"
            record = args.output / f'{stem}.json'
            if record.exists():
                continue
            hooks = [ComaHook(damping_factor=alpha) for _ in layers[len(layers)//2:]]
            for hook, layer in zip(hooks, layers[len(layers)//2:]):
                hook.register(layer)
            try:
                baseline_zero, zero = paired_trajectories(
                    model, inputs, layers[config['target_layer']], token_index, 0.0, args.seed
                )
                baseline, pulsed = paired_trajectories(
                    model, inputs, layers[config['target_layer']], token_index, float(amplitude), args.seed
                )
                language = continuation_metrics(model, full, length)
            finally:
                for hook in hooks:
                    hook.remove()
            matrices = {
                'matched_zero': np.linalg.norm(zero - baseline_zero, axis=-1),
                'unmatched_zero': np.linalg.norm(zero - intact, axis=-1),
                'matched_pulse': np.linalg.norm(pulsed - baseline, axis=-1),
                'unmatched_pulse': np.linalg.norm(pulsed - intact, axis=-1),
                'baseline_noise': noise,
            }
            assert np.count_nonzero(matrices['matched_zero']) == 0
            if alpha == 1.0:
                np.testing.assert_array_equal(matrices['matched_pulse'], matrices['unmatched_pulse'])
            restored = hidden_states(model, inputs)
            np.testing.assert_array_equal(restored, intact)
            np.savez_compressed(args.output / f'{stem}.npz', **matrices)
            result = {
                'prompt': prompt['id'], 'alpha': alpha, 'seed': args.seed,
                'target_layer': config['target_layer'], 'target_token': token_index,
                'amplitude': float(amplitude), 'noise_threshold': float(np.percentile(noise, 99)),
                'metrics': {key: summarize(value, noise) for key, value in matrices.items() if key != 'baseline_noise'},
                'language': language, 'zero_response_passed': True, 'restoration_passed': True,
            }
            record.write_text(json.dumps(result, indent=2) + '\n')
            print(stem, result['metrics'], flush=True)
    metadata['seconds'] = time.perf_counter() - started
    metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, choices=[42, 123, 999], required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'data/paired_response_config.json')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
