"""Paired hidden-state measurements under the same active intervention."""

import numpy as np
import torch


def paired_trajectories(model, inputs, target_layer, token_index, amplitude, seed):
    """Return unpulsed and pulsed states with all existing hooks held fixed.

    Inputs have batch size one. The model must be in evaluation mode on one
    device. Stochastic intervention hooks use common random numbers in the two
    passes; the impulse has its own generator and does not consume that stream.
    Returned arrays follow Hugging Face's hidden_states[1:] convention, including
    any final normalization applied to the last state.
    """
    device = inputs['input_ids'].device
    devices = [device.index] if device.type == 'cuda' else []
    generator = torch.Generator(device=device).manual_seed(seed + 1)

    def pulse(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        direction = torch.randn(hidden.shape[-1], generator=generator, device=device)
        direction = direction / direction.norm() * amplitude
        changed[:, token_index, :] += direction.to(hidden.dtype)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    trajectories = []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        for pulsed in [False, True]:
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            handle = target_layer.register_forward_hook(pulse) if pulsed else None
            try:
                result = model(**inputs, output_hidden_states=True, use_cache=False)
                trajectories.append(np.stack([
                    state[0].float().cpu().numpy() for state in result.hidden_states[1:]
                ]))
            finally:
                if handle is not None:
                    handle.remove()
    return tuple(trajectories)
