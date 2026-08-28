"""
Prompt construction for Stage 2. The exact template from the proposal's
Operational Step-by-Step Execution Lifecycle table (Section 4):

    "Context: {d+}\nQuery: {q}\nAnswer:"

Also provides the "bare" (context-free) variant of the same query, which
Stage 3 (Day 6) needs to elicit the generator's *parametric-only* answer --
i.e. what it would say from pretrained memory alone, with nothing retrieved
-- and which Stage 4's contrastive decoding forward pass reuses directly.
Both live here so the exact wording only needs to be defined once.
"""

from __future__ import annotations


def build_context_prompt(context: str, query: str) -> str:
    """Stage 2's main prompt: context + query, as specified by the proposal."""
    return f"Context: {context}\nQuery: {query}\nAnswer:"


def build_bare_prompt(query: str) -> str:
    """Query-only prompt, no retrieved context -- probes the generator's
    parametric (pretrained) belief about the answer."""
    return f"Query: {query}\nAnswer:"
