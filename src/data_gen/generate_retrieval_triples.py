"""
Builds the Stage-1 retrieval triples dataset: for every (query, fresh_doc,
stale_doc) record produced by fact_mutation_templates, compute a lexical
overlap score between fresh_doc and stale_doc, then split by *domain* into:

  - train           : TRAIN_DOMAINS, 90% of records
  - in_domain_val    : TRAIN_DOMAINS, remaining 10% (same domains/templates as
                       train, different entities — monitors ordinary fit)
  - ood_val          : OOD_VAL_DOMAIN, entirely unseen during training (used
                       for model selection / early stopping on unseen domain)
  - ood_test         : OOD_TEST_DOMAIN, entirely unseen until final evaluation

This domain-level split (not a per-triple random split) is what actually
tests whether the retriever learned genuine temporal/factual reasoning
instead of memorizing template surface patterns.

"Random in-batch negatives" (e_dj- in the proposal's loss) are NOT stored
here — they are just other examples' fresh_doc within a training minibatch,
constructed on the fly in train_retriever.py. Only the "mined hard negative"
(e_dk^r == stale_doc) is precomputed per-triple, since it requires the
domain-specific template logic above.

Usage:
    python -m src.data_gen.generate_retrieval_triples
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.data_gen.fact_mutation_templates import (
    ALL_DOMAINS,
    OOD_TEST_DOMAIN,
    OOD_VAL_DOMAIN,
    TRAIN_DOMAINS,
    generate_domain_triples,
)

SEED = 42
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
TRAIN_FRACTION = 0.9  # within TRAIN_DOMAINS, held-out fraction for in_domain_val


def jaccard_overlap(a: str, b: str) -> float:
    """Token-level Jaccard similarity — quantifies how lexically close the
    fresh and stale docs are. Near 1.0 means a keyword-overlap retriever
    cannot distinguish them; this is the property the templates are
    designed to produce."""
    tok_a = set(a.lower().replace(".", "").replace(",", "").split())
    tok_b = set(b.lower().replace(".", "").replace(",", "").split())
    if not tok_a or not tok_b:
        return 0.0
    return len(tok_a & tok_b) / len(tok_a | tok_b)


def build_all_records(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    records = []
    rec_id = 0
    for domain in ALL_DOMAINS:
        domain_records = generate_domain_triples(domain, rng)
        for rec in domain_records:
            rec["id"] = rec_id
            rec["lexical_overlap"] = round(jaccard_overlap(rec["fresh_doc"], rec["stale_doc"]), 4)
            records.append(rec)
            rec_id += 1
    return records


def assign_splits(records: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    rng = random.Random(seed + 1)
    splits: dict[str, list[dict]] = {
        "train": [], "in_domain_val": [], "ood_val": [], "ood_test": [],
    }
    for rec in records:
        domain = rec["domain"]
        if domain == OOD_VAL_DOMAIN:
            splits["ood_val"].append(rec)
        elif domain == OOD_TEST_DOMAIN:
            splits["ood_test"].append(rec)
        elif domain in TRAIN_DOMAINS:
            if rng.random() < TRAIN_FRACTION:
                splits["train"].append(rec)
            else:
                splits["in_domain_val"].append(rec)
        else:
            raise ValueError(f"Domain {domain} not in ALL_DOMAINS routing")
    return splits


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    records = build_all_records()
    splits = assign_splits(records)

    write_jsonl(records, OUT_DIR / "all_triples.jsonl")
    for split_name, split_records in splits.items():
        write_jsonl(split_records, OUT_DIR / f"{split_name}.jsonl")

    overlaps = [r["lexical_overlap"] for r in records]
    print(f"Total records: {len(records)}")
    for split_name, split_records in splits.items():
        domains_in_split = sorted({r["domain"] for r in split_records})
        print(f"  {split_name:15s}: {len(split_records):5d} records  domains={domains_in_split}")
    print(f"Lexical overlap (fresh_doc vs stale_doc): "
          f"mean={sum(overlaps)/len(overlaps):.3f}, "
          f"min={min(overlaps):.3f}, max={max(overlaps):.3f}")
    print(f"Written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
