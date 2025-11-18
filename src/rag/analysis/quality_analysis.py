"""Quality degradation analysis utilities.

This module provides simple semantic similarity-based comparison between
baseline and fine-tuned responses using sentence-transformers embeddings.
"""
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer, util

_EMBED_MODEL = None


def _get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def compare_responses(
    baseline_responses: List[str], finetuned_responses: List[str]
) -> Dict[str, Any]:
    """Compute semantic similarity between baseline and fine-tuned responses.

    Returns a dict containing per-query similarities and an average similarity
    score. Similarity is cosine similarity between sentence-transformers
    embeddings (0..1).
    """
    embedder = _get_embedder()

    # Ensure same length
    n = min(len(baseline_responses), len(finetuned_responses))
    baseline_responses = baseline_responses[:n]
    finetuned_responses = finetuned_responses[:n]

    if n == 0:
        return {"avg_similarity": 0.0, "per_query": []}

    # Compute embeddings in batch
    emb_a = embedder.encode(baseline_responses, convert_to_tensor=True)
    emb_b = embedder.encode(finetuned_responses, convert_to_tensor=True)

    sims = util.cos_sim(emb_a, emb_b).diagonal().cpu().numpy()

    per_query = [float(s) for s in sims.tolist()]
    avg_similarity = float(float(np.mean(sims)))

    return {"avg_similarity": avg_similarity, "per_query": per_query}
