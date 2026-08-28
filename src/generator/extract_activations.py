"""
Residual-stream activation extraction from the frozen generator
(Qwen2.5-1.5B-Instruct). Uses `output_hidden_states=True` on a plain forward
pass rather than manual forward hooks -- see src/generator/hooks.py for a
side-by-side confirmation that the two approaches agree; this module is the
"production" path since Stage 3/4 only ever need to *read* activations, not
intervene on them mid-pass.

`extract_last_token_all_layers` returns exactly the feature vector Stage 3's
per-layer probes are trained on: the residual-stream state at the final
prompt token, at every layer (index 0 = input embeddings, index i = after
transformer layer i).

Running this file directly performs the Day-5 milestone: retrieve with the
Day-4 trained BiEncoder, build a Stage-2 prompt, extract activations, and
print shapes/values on a handful of real held-out queries.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load_generator(model_name: str = GEN_MODEL_ID, device: torch.device | None = None):
    """fp16 only makes sense on CUDA -- see docs/DECISIONS.md #4. On CPU we
    fall back to fp32 since CPU fp16 matmul is either unsupported or far
    slower than fp32 on most hardware."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def extract_all_layer_hidden_states(model, tokenizer, prompt: str, device: torch.device, max_length: int = 512):
    """Returns the raw `hidden_states` tuple: length num_hidden_layers + 1,
    each tensor [1, seq_len, hidden_size]."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model(**inputs, output_hidden_states=True)
    return out.hidden_states


@torch.no_grad()
def extract_last_token_all_layers(model, tokenizer, prompt: str, device: torch.device, max_length: int = 512) -> torch.Tensor:
    """Returns [num_layers + 1, hidden_size] -- the residual-stream state at
    the final prompt token, for every layer. This is the Stage-3 probe's
    input feature vector for one example."""
    hidden_states = extract_all_layer_hidden_states(model, tokenizer, prompt, device, max_length)
    last_token_vectors = torch.stack([h[0, -1, :] for h in hidden_states], dim=0)
    return last_token_vectors.float().cpu()


def _demo() -> None:
    """Day-5 milestone: Stage 1 (retrieve) -> Stage 2 (prompt + extract) on a
    handful of real held-out queries."""
    import json
    from pathlib import Path

    from src.generator.prompt_utils import build_context_prompt
    from src.retriever.model import BiEncoder

    repo_root = Path(__file__).resolve().parents[2]
    retriever_device = torch.device("cpu")

    retriever = BiEncoder().to(retriever_device)
    ckpt_path = repo_root / "checkpoints" / "retriever" / "best.pt"
    retriever.load_state_dict(torch.load(ckpt_path, map_location=retriever_device))
    retriever.eval()

    with open(repo_root / "data" / "synthetic" / "ood_test.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f][:5]

    print("Loading generator (Qwen2.5-1.5B-Instruct)... this may take a minute on CPU.")
    model, tokenizer, gen_device = load_generator()
    print(f"Generator loaded on {gen_device}, {model.config.num_hidden_layers} layers, "
          f"hidden_size={model.config.hidden_size}")

    for rec in records:
        # Stage 1: retrieve top-1 among this record's own {fresh, stale} pair
        # (mirrors eval_retriever.py's per-query candidate set at small scale).
        q_emb = retriever.encode([rec["query"]], retriever_device)
        cand_texts = [rec["fresh_doc"], rec["stale_doc"]]
        cand_emb = retriever.encode(cand_texts, retriever_device)
        top1_idx = (q_emb @ cand_emb.t()).argmax(dim=1).item()
        retrieved_doc = cand_texts[top1_idx]
        retrieved_is_fresh = (top1_idx == 0)

        # Stage 2: prompt construction + activation extraction
        prompt = build_context_prompt(retrieved_doc, rec["query"])
        vectors = extract_last_token_all_layers(model, tokenizer, prompt, gen_device)

        print("\n---")
        print(f"query: {rec['query']}")
        print(f"retrieved: {'FRESH' if retrieved_is_fresh else 'STALE'} -> {retrieved_doc[:80]}...")
        print(f"hidden-state stack shape: {tuple(vectors.shape)}  "
              f"(expect [{model.config.num_hidden_layers + 1}, {model.config.hidden_size}])")
        print(f"layer 20 vector, first 5 dims: {vectors[20, :5]}")


if __name__ == "__main__":
    _demo()
