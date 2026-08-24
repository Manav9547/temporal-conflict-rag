# Design Decisions

This document records every place where the implementation deviates from a literal
reading of `Proposal.pdf`, and why. Each decision was made to keep the project
achievable inside a 2-week, $0-budget, free-GPU (Colab/Kaggle T4) constraint while
staying faithful to the proposal's scientific intent.

## 1. Generator model: Qwen2.5-1.5B-Instruct (not Llama-3.2-1B-Instruct)

The proposal's evaluation table claims the conflict probe performs best at "layers
18–24" (Section 5). That range only makes sense for a model with at least ~28
layers:

- `Qwen/Qwen2.5-1.5B-Instruct` has **28 transformer layers** — the 18–24 target
  band sits at ~64–86% depth, a plausible location for a late-but-not-final
  abstraction to live.
- `meta-llama/Llama-3.2-1B-Instruct` has only **16 layers** — layers 18–24 don't
  exist on this model at all.

**Decision:** use Qwen2.5-1.5B-Instruct as the primary (only, for v1) generator.
Secondary reason: Qwen2.5 is ungated on Hugging Face, so there's no license
approval wait; Llama-3.2 requires accepting Meta's license and an approved HF
token, which is friction we can't afford under a 2-week deadline.

If time permits at the end (Day 13 stretch), Llama-3.2-1B can be added as an
ablation with a *proportionally rescaled* probe layer range: 18–24 out of 28
layers is ~64–86% depth, which on a 16-layer model corresponds to roughly
layers 10–14.

## 2. Temperature τ is fixed at 0.05, not learned

The proposal's loss section calls τ "a learned temperature scaling factor" but
also states its value as a constant (`τ = 0.05`) alongside the fixed margin
`γ = 0.2`. Making τ learnable adds a training-stability concern (it can collapse
toward 0, causing loss spikes) for marginal benefit at our data scale (~1000
synthetic triples).

**Decision:** treat τ as a fixed hyperparameter, τ = 0.05, matching the stated
value. Documented as a simplification, not a contradiction of the proposal's
intent.

## 3. Conflict probe is checked once per query, not per generation token

Component 2 of the proposal writes the probe output as `ŷ_conflict^(t)`,
suggesting a per-token (per decoding step) re-evaluation. However, the
Operational Step-by-Step Execution Lifecycle table (Section 4) checks the probe
exactly once — after the prompt forward pass, before generation starts — and
then commits to one branch (steered or greedy) for the entire response.

Per-token re-checking would require re-running the probe (and potentially
re-deciding the branch) at every decoding step, tripling the number of forward
passes needed during generation and adding meaningful engineering complexity
(the branch could flip mid-generation, which the proposal's architecture
diagram doesn't account for).

**Decision:** implement the single-check-at-prompt-end version, matching the
lifecycle table. This is the v1 (and likely final) behavior. Per-step
re-evaluation is listed as an optional Day-13 stretch goal only if time remains.

## 4. Generator runs in fp16, not 4-bit NF4

The proposal's hardware section lists 4-bit NF4 as an option for "LLM inference
& hidden state probing." Quantization introduces dequantization noise into
activations — this is a bad property for a probe that is trying to detect a
subtle linear signal (conflict) in the residual stream. Noisy activations would
make Stage 3's ROC-AUC numbers harder to trust and harder to reproduce.

Qwen2.5-1.5B-Instruct in fp16 is only ~3GB of weights. A T4 has 16GB of VRAM,
which comfortably fits fp16 weights plus two independent KV caches for Stage 4's
dual forward pass (context-conditioned + parametric-only streams), with room to
spare.

**Decision:** run the generator in fp16 throughout. Fall back to 4-bit only if
VRAM pressure is empirically observed (not expected at this model scale).

---

*Last updated: Day 1 (2026-08-24).*
