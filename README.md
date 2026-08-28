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
- [x] Day 4 — retriever training + evaluation
- [x] Day 5 — generator prompt construction + activation extraction
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

## Stage 1 results (Day 4)

Trained locally on CPU (~3 min, 7 epochs, early-stopped) -- MiniLM at this
data scale doesn't need a GPU. MRR@1 (== Hit@1 at k=1) baseline vs. trained,
against the proposal's targets:

| split          | domain          | MRR@1 baseline | MRR@1 trained | target | uniformity trained | target |
|----------------|------------------|:--:|:--:|:--:|:--:|:--:|
| in_domain_val  | train domains    | 0.63 | **1.00** | >0.85 | -2.29 | <-2.20 (missed) |
| ood_val        | pricing_tier     | 0.55 | **0.81** | >0.85 (missed) | -1.86 | <-2.20 (missed) |
| ood_test       | product_spec     | 0.64 | **0.91** | >0.85 | -2.34 | <-2.20 (met) |

Run: `python -m src.retriever.train_retriever` then
`python -m src.retriever.eval_retriever --split <name>`.

**Honest read of these numbers:** MRR@1 improves substantially everywhere,
but training loss collapses to near-zero after epoch 1 -- almost certainly
because every domain's fresh/stale split maps onto the same two year pools
(`FRESH_YEARS=[2025,2026]` vs `STALE_YEARS=[2018-2022]`, see
`fact_mutation_templates.py`), so the model likely learns a shallow "does
this doc contain a recent-looking year token" shortcut rather than deep
per-domain temporal reasoning. That shortcut is domain-invariant, which is
exactly why it transfers to held-out domains at all -- but it also means the
ood_val/ood_test split doesn't cleanly separate "learned temporal semantics"
from "learned to detect one specific token pattern." Uniformity getting
*worse* on two splits after training is consistent with this: the model is
optimizing for a fairly narrow discriminative signal, not genuinely spreading
representations across the embedding space. A stronger generalization test
(e.g. holding out specific year values from training, not just domains)
is a reasonable Day-13 stretch item if time allows -- documented here rather
than silently treated as solved.

## Stage 2 notes (Day 5)

`python -m src.generator.extract_activations` runs the Stage 1 -> Stage 2
smoke test: retrieves top-1 with the trained retriever, builds the
`Context/Query/Answer` prompt, and extracts the residual-stream state at
every layer for the final prompt token (shape `[29, 1536]` for
Qwen2.5-1.5B: 28 layers + the embedding layer). Ran on 5 real held-out
queries -- retriever picked the fresh document in all 5.

`python -m src.generator.hooks` cross-checks that against manual
`register_forward_hook` capture, and surfaces a real subtlety worth knowing
before Stage 3: `hidden_states[-1]` (the last entry) is **not** the raw
residual stream -- HF applies the model's final RMSNorm to the last layer's
output before storing it there, so it doesn't match a raw hook capture at
that index (diff ~244), while every other layer matches exactly (diff 0.0).
Irrelevant to our probe range (layers 18-24) but would be a silent bug for
any sweep that assumed every `hidden_states` index means the same thing.

## Models

- Retriever: `sentence-transformers/all-MiniLM-L6-v2` (22M params)
- Generator: `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, fp16) — see
  `docs/DECISIONS.md` §1 for why this was chosen over Llama-3.2-1B-Instruct.
