"""
Learning exercise (not used by the pipeline itself): confirms, hands-on, that
`register_forward_hook` and `output_hidden_states=True` are reading the exact
same residual-stream values, just via two different mechanisms.

- `output_hidden_states=True` is declarative and read-only: you get every
  layer's output handed back to you after the full forward pass completes.
- A forward hook is imperative and runs *during* the forward pass, at the
  moment a specific module finishes executing. It's the only option if you
  ever need to *modify* an activation mid-pass (e.g. to intervene on the
  residual stream directly) rather than just inspect it afterwards -- Stage
  3/4 here only ever read, so extract_activations.py uses the simpler
  declarative path, but Stage 4's steering concept (blending two logit
  streams) is conceptually closer to "intervention," which is why this
  mechanism is worth having built once by hand.

For a decoder-only model built the way Qwen2.5 is, `model.model.layers` is
the list of transformer blocks; each block's forward returns a tuple whose
first element is the residual-stream hidden state after that block runs.
That value is identical to `hidden_states[i + 1]` from
`output_hidden_states=True` (index 0 there is the pre-layer embedding
output) for every layer EXCEPT the last one -- and that exception is the
actual point of running this script.

VERIFIED DISCREPANCY (worth knowing before Stage 3): HF's internals apply
the model's final normalization (`model.model.norm`, an RMSNorm) to the last
decoder layer's output before storing it as `hidden_states[-1]`. Every other
entry in `hidden_states` is the RAW, pre-norm residual stream, but the very
last entry is NOT -- it's post-norm. Running this file confirms it directly:
layers 0 and 14 match their hook capture exactly (diff = 0.0), but layer 27
(the last of 28) differs by ~244 in max absolute value, entirely explained
by the norm. Since our Stage-3 probe target (layers 18-24) sits well short
of the final layer, this doesn't corrupt our results -- but it would be an
easy, silent bug if the probe ever swept all the way to the last layer
assuming every index means "raw residual stream after layer i."

Usage: python -m src.generator.hooks
"""

from __future__ import annotations

import torch

from src.generator.extract_activations import GEN_MODEL_ID, load_generator


def demo_forward_hooks(model_name: str = GEN_MODEL_ID) -> None:
    model, tokenizer, device = load_generator(model_name)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            # output is a tuple for a Qwen2 decoder layer; output[0] is the
            # residual-stream hidden state after this layer's attention+FFN.
            captured[layer_idx] = output[0].detach()
        return hook

    hook_layers = [0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]
    handles = [model.model.layers[i].register_forward_hook(make_hook(i)) for i in hook_layers]

    prompt = "Context: The battery life is rated at 38 hours.\nQuery: What is the battery life?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    for h in handles:
        h.remove()

    last_layer_idx = model.config.num_hidden_layers - 1
    print(f"Checked layers: {hook_layers}\n")
    for layer_idx in hook_layers:
        hook_state = captured[layer_idx]
        ref_state = out.hidden_states[layer_idx + 1]  # +1: index 0 is pre-layer embeddings
        match = torch.allclose(hook_state, ref_state, atol=1e-5)
        max_diff = (hook_state - ref_state).abs().max().item()
        note = "  <- expected mismatch: hidden_states[-1] has the final norm applied, this hook capture doesn't" \
            if (layer_idx == last_layer_idx and not match) else ""
        print(f"layer {layer_idx:2d}: hook output == hidden_states[{layer_idx + 1}]? "
              f"{match}  (max abs diff: {max_diff:.2e}){note}")


if __name__ == "__main__":
    demo_forward_hooks()
