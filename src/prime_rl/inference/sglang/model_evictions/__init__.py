"""Model-specific KV-Streams eviction plans, chosen by the model's ``model_type``.

A chat template may render a model's history differently from how each turn was
sampled; an append-only token stream keeps every turn as sampled, so the model sees
a history it never saw in training. A model plan evicts the parts of it the model is
worse for attending to - not necessarily everything the template drops. It runs
whenever KV-Streams eviction is on and the served model's HF config ``model_type`` is
in ``MODEL_TYPES``.

Each module here defines ``MARKERS`` (name -> text; the launcher resolves each to
token ids, as the SGLang server runs without a tokenizer) and ``Plan(**ids)``, an
``eviction.EvictionPlan`` taking ``<name>_ids`` for every marker. A model family adds
a module, a ``PLANS`` entry and its ``MODEL_TYPES``.
"""

from __future__ import annotations

from typing import Any

from . import qwen35

PLANS = {"qwen3.5": qwen35}

MODEL_TYPES = {
    "qwen3_5": "qwen3.5",
    "qwen3_5_text": "qwen3.5",
    "qwen3_5_moe": "qwen3.5",
    "qwen3_5_moe_text": "qwen3.5",
}
"""HF config ``model_type`` -> its plan."""


def build(name: str, ids: dict[str, list[int]]) -> Any:
    """The named plan, from its markers' token ids (``{"<marker>_ids": [...]}``)."""
    return PLANS[name].Plan(**ids)
