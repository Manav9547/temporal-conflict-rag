# Temporal Knowledge Conflict Resolution

Resolving parametric knowledge conflicts in LLMs via temporal metric learning and
residual-stream probing. Implementation of the architecture described in
`Proposal.pdf`, with deviations documented in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Pipeline

1. **Time-aware retriever** (`src/retriever/`) — a MiniLM bi-encoder fine-tuned
   with a margin-penalized InfoNCE loss to separate temporally fresh documents
   from stale ones, even under high lexical overlap.
2. **Prompt construction + forward pass** (`src/generator/`) — builds the
   `Context / Query / Answer` prompt and extracts residual-stream hidden states
   from Qwen2.5-1.5B-Instruct.
3. **Conflict probing** (`src/probe/`) — per-layer logistic regression probes
   that detect when the model's internal state signals a contradiction between
   its parametric memory and the retrieved context.
4. **Contrastive steered decoding** (`src/decoding/`) — when conflict is
   detected, a custom generation loop blends context-conditioned and
   context-free logits to favor the retrieved (fresh) context.

## Status

Day-by-day build log follows the plan in
`C:\Users\jinda\.claude\plans\read-the-proposal-pdf-and-memoized-hollerith.md`.

- [x] Day 1 — repo scaffold, environment setup, design decisions
- [x] Day 2 — synthetic fact-mutation data generation
- [x] Day 3 — retriever model + loss implementation
- [ ] Day 4 — retriever training + evaluation
- [ ] Day 5 — generator prompt construction + activation extraction
- [ ] Day 6 — conflict-probe dataset generation
- [ ] Day 7 — probe training (midpoint milestone)
- [ ] Day 8 — thin end-to-end pipeline wiring
- [ ] Day 9 — contrastive decoding implementation
- [ ] Day 10 — full pipeline integration
- [ ] Day 11 — evaluation harness
- [ ] Day 12 — full evaluation run
- [ ] Day 13 — error analysis, ablations, polish
- [ ] Day 14 — buffer / packaging

## Setup

```bash
pip install -r requirements.txt
```

Run [`notebooks/00_setup_and_env_check.ipynb`](notebooks/00_setup_and_env_check.ipynb)
on a Colab or Kaggle T4 GPU runtime to verify the environment before proceeding.

## Data

Synthetic retrieval triples are generated (not stored raw in git — regenerate
with the command below) from parameterized templates across 5 domains:
corporate policy, software versions, regulatory guidelines, pricing tiers,
and product specs. Each triple pairs a query with a temporally *fresh*
document and a lexically near-identical *stale* document that differs only
in the fact value and date — this is what forces the retriever to learn
temporal signal rather than keyword overlap.

Splits are assigned by **domain**, not per-triple, to test generalization:
`train`/`in_domain_val` (3 domains), `ood_val` (pricing_tier, unseen during
training), `ood_test` (product_spec, held out until final evaluation).

```bash
python -m src.data_gen.generate_retrieval_triples
```

Current dataset: 980 records (641 train / 59 in-domain val / 120 ood val /
160 ood test), mean lexical (Jaccard) overlap between fresh/stale docs: 0.78.

**Untrained MiniLM baseline** (sanity check the hard negatives are genuinely
hard): given a query and its own fresh vs. stale doc, untrained MiniLM
prefers the fresh doc only 60% of the time (mean cosine sim 0.854 vs. 0.853
— nearly indistinguishable). This is the gap Stage 1 training (Day 4) should
close. Run `python -m pytest tests/` to verify the loss implementation
against a hand-computed example first.

## Models

- Retriever: `sentence-transformers/all-MiniLM-L6-v2` (22M params)
- Generator: `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, fp16) — see
  `docs/DECISIONS.md` §1 for why this was chosen over Llama-3.2-1B-Instruct.
