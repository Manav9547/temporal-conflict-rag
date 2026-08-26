"""
Full corpus-level evaluation of the Stage-1 retriever: MRR@1, alignment,
and uniformity, comparing the trained checkpoint against the frozen
(untrained) MiniLM baseline -- mirroring the proposal's evaluation table.

MRR@1 note: with k=1, mean reciprocal rank collapses to plain Hit@1 /
Precision@1 (reciprocal rank is either 1, if the correct doc is ranked
first, or 0 otherwise -- there's no partial credit at k=1). We still call it
MRR@1 to match the proposal's terminology, but it is numerically identical
to top-1 accuracy here.

Corpus construction: for a given split, the retrieval corpus is every
fresh_doc AND every stale_doc across all records in that split (not just the
query's own pair) -- this makes retrieval a real ranking problem against
many distractors, most of which share the query's general domain/template
but not its specific entity.

Usage:
    python -m src.retriever.eval_retriever --split ood_test
    python -m src.retriever.eval_retriever --split ood_test --checkpoint checkpoints/retriever/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.retriever.losses import alignment, uniformity
from src.retriever.model import BiEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def evaluate_split(model: BiEncoder, records: list[dict], device: torch.device) -> dict:
    model.eval()
    n = len(records)

    queries = [r["query"] for r in records]
    fresh_docs = [r["fresh_doc"] for r in records]
    stale_docs = [r["stale_doc"] for r in records]

    q_emb = model.encode(queries, device)                    # [n, d]
    fresh_emb = model.encode(fresh_docs, device)              # [n, d]
    stale_emb = model.encode(stale_docs, device)               # [n, d]

    corpus_emb = torch.cat([fresh_emb, stale_emb], dim=0)      # [2n, d] -- fresh_docs[i] is corpus index i
    sims = q_emb @ corpus_emb.t()                               # [n, 2n]

    top1 = sims.argmax(dim=1)
    correct_idx = torch.arange(n, device=device)                # query i's relevant doc is corpus index i
    mrr_at_1 = (top1 == correct_idx).float().mean().item()

    return {
        "n": n,
        "mrr_at_1": mrr_at_1,
        "alignment": alignment(q_emb, fresh_emb).item(),
        "uniformity_of_docs": uniformity(corpus_emb).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="ood_test", choices=["train", "in_domain_val", "ood_val", "ood_test"])
    parser.add_argument("--checkpoint", default="checkpoints/retriever/best.pt")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(REPO_ROOT / args.data_dir / f"{args.split}.jsonl")
    print(f"Evaluating split='{args.split}' (n={len(records)}) on device={device}")

    baseline_model = BiEncoder(args.model_name).to(device)
    baseline_metrics = evaluate_split(baseline_model, records, device)

    checkpoint_path = REPO_ROOT / args.checkpoint
    trained_metrics = None
    if checkpoint_path.exists():
        trained_model = BiEncoder(args.model_name).to(device)
        trained_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        trained_metrics = evaluate_split(trained_model, records, device)
    else:
        print(f"WARNING: checkpoint {checkpoint_path} not found -- reporting baseline only. "
              f"Run train_retriever.py first.")

    print("\n=== Retriever Evaluation ===")
    print(f"{'metric':22s} {'baseline (untrained)':22s} {'trained':>10s}  {'target (proposal)':>18s}")
    targets = {"mrr_at_1": "> 0.85", "alignment": "n/a", "uniformity_of_docs": "< -2.20"}
    for key in ["mrr_at_1", "alignment", "uniformity_of_docs"]:
        base_val = f"{baseline_metrics[key]:.4f}"
        trained_val = f"{trained_metrics[key]:.4f}" if trained_metrics else "n/a"
        print(f"{key:22s} {base_val:22s} {trained_val:>10s}  {targets[key]:>18s}")

    results_dir = REPO_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "split": args.split,
        "baseline": baseline_metrics,
        "trained": trained_metrics,
    }
    out_path = results_dir / f"retriever_eval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
