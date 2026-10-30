# Perturbation responses need a baseline under the same intervention

The response measurement in Experiments 03 and 04 previously subtracted an intact-model trajectory from a trajectory obtained with both an intervention and an impulse. This difference includes the effect of the intervention itself. A zero-amplitude impulse can therefore produce a nonzero reported response and complexity score.

Let $H_{a,d}$ denote the hidden states with intervention setting $a$ and impulse $d$. The previous difference was

$$H_{a,d}-H_{1,0} = (H_{a,d}-H_{a,0}) + (H_{a,0}-H_{1,0}).$$

The first term is the response to the impulse under the intervention. The second is the change caused by the intervention without an impulse. Subtracting $H_{a,0}$ isolates the first term. At $d=0$, that difference must vanish for a deterministic model. This is a property of the contrast, independent of a particular complexity measure or a biological interpretation.

`src/metrics/evoked.py` implements paired measurements while keeping the active damping, steering or stochastic hooks in place. The two passes use the same random stream for stochastic interventions. The impulse has a separate generator. The helper restores random-generator states and removes its impulse hook. Experiments 03 and 04 now use this helper instead of an externally supplied intact trajectory.

## Real-model control experiment

Experiment 06 applies this comparison to Qwen/Qwen3-4B. Its full configuration is in `experiments/data/paired_response_config.json`. It uses all three existing Experiment 04 prompt and continuation pairs, all ten existing damping settings and all three existing seeds, for 90 records. No prompt, damping setting or seed was selected after observing the results. These short, authored examples are a measurement-control battery, not a representative language benchmark.

Damping is applied to the latter half of the decoder layers. An impulse is added after layer 3 at token `max(3, sequence_length // 3)`. Its norm is 2.5 times the mean intact-state norm at that layer. The zero-amplitude control uses the same procedure. The saved response matrices are norms over the hidden dimension at each layer and token. They follow the model's `hidden_states[1:]` convention, including final normalization in the last state.

To isolate subtraction as the changed factor, both comparisons retain the existing calibration convention: Gaussian input-embedding jitter with per-coordinate standard deviation equal to 0.01 times the mean embedding norm, and a pooled 99th-percentile threshold from the intact model. This is a computational convention, not measured physiological noise. In particular, the total jitter norm also depends on hidden dimension. No threshold was adjusted to improve the outcomes.

| Damping | Unmatched zero-impulse score | Matched zero-impulse score | Unmatched pulse score | Matched pulse score | Continuation NLL |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3.147098 |
| 0.85 | 0.420056 | 0.000000 | 0.417818 | 0.000000 | 4.376776 |
| 0.70 | 0.248239 | 0.000000 | 0.242719 | 0.042479 | 13.538025 |
| 0.55 | 0.238931 | 0.000000 | 0.227102 | 0.000000 | 15.643573 |
| 0.40 | 0.205490 | 0.000000 | 0.206306 | 0.000000 | 14.415226 |
| 0.30 | 0.217447 | 0.000000 | 0.211149 | 0.000000 | 14.360782 |
| 0.20 | 0.209130 | 0.000000 | 0.209039 | 0.000000 | 12.454863 |
| 0.10 | 0.221641 | 0.000000 | 0.221331 | 0.000000 | 14.826093 |
| 0.05 | 0.210757 | 0.000000 | 0.210757 | 0.000000 | 13.068954 |
| 0.00 | 0.285527 | 0.000000 | 0.285527 | 0.000000 | 11.931214 |

Scores are the repository's current `compute_casali_pci` output, averaged over three prompts and three impulse directions. NLL is averaged over prompts; repeating a deterministic language calculation for different impulse seeds does not add independent observations. The table is descriptive. It does not provide population confidence intervals.

All 81 records with nontrivial damping had a nonzero response and nonzero score under unmatched subtraction even when the impulse amplitude was zero. Matched subtraction gave an exactly zero response in all 90 zero-impulse controls. Without damping, matched and unmatched pulse responses were identical. Removal of the damping hooks restored the intact outputs in every record.

All nine intact-model pulse responses had nonzero activation norms but zero complexity scores under the retained calibration. Thus this experiment establishes the subtraction problem and its correction. It does not validate the threshold, normalization or sensitivity of the current score. The bfloat16 calculation, short prompt battery and single model also limit generalization. Historical GPT-2 and Pythia result files have not been overwritten or claimed to be reproduced by this Qwen experiment.

## Reproduction and checks

The run used Python 3.12, PyTorch 2.6.0+cu124, Transformers 5.15.1, NumPy 2.2.6, bfloat16 weights and SDPA attention. Each seed ran independently on one NVIDIA H100. The measured analysis time after model loading was 6.7–6.9 seconds per seed. The code, configuration and complete numerical records are below 1 MB; model weights are a separate dependency.

From the repository root:

```bash
python experiments/06_paired_response_controls.py --model Qwen/Qwen3-4B --seed 42 --output experiments/results/paired-response/seed-42
python experiments/06_paired_response_controls.py --model Qwen/Qwen3-4B --seed 123 --output experiments/results/paired-response/seed-123
python experiments/06_paired_response_controls.py --model Qwen/Qwen3-4B --seed 999 --output experiments/results/paired-response/seed-999
python experiments/summarize_paired_response.py experiments/results/paired-response
```

Use an empty output directory for a new run. Existing completed records are reused to resume interrupted runs. The original configuration must remain fixed when resuming. GPU and library differences can change numerical values.

Three regression tests use an actual small, randomly initialized GPT-2 model to check the zero-impulse response under damping, causal pulse propagation and hook removal, and common random numbers with random-state restoration. The complete suite passed 15 tests with Python and NumPy seeds fixed at 20261030. An initial unseeded full-suite run also failed the pre-existing probabilistic Sleeping Beauty test; that test was not changed. `quickstart_demo.py` completed. These software checks do not validate the demo's biological interpretations.

The original clinical PCI work concerns brain responses to physical stimulation, with its own acquisition, preprocessing and validation procedures: [Casali et al., 2013](https://doi.org/10.1126/scitranslmed.3006294). No clinical cutoff, consciousness inference or patient-treatment implication is transferred to the transformer measurements here.

Haoyang Su, Fudan University and Shanghai Innovation Institute. OpenAI Codex assisted with identifying the contrast, implementation, execution and reporting. Human scientific review remains pending.
