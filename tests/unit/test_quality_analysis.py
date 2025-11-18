import torch

from src.rag.analysis import quality_analysis


class StubEmbedder:
    def __init__(self):
        pass

    def encode(self, texts, convert_to_tensor=True):
        # Return identity embeddings for deterministic similarity
        n = len(texts)
        return torch.eye(n)


class StubEmbedderDiff:
    def encode(self, texts, convert_to_tensor=True):
        # If text contains 'A' returns vector of ones, else zeros
        import torch

        vecs = []
        for t in texts:
            if "A" in t:
                vecs.append(torch.ones(4))
            else:
                vecs.append(torch.zeros(4))
        return torch.stack(vecs)


def test_compare_responses_identical(monkeypatch):
    monkeypatch.setattr(
        quality_analysis, "_get_embedder", lambda model_name=None: StubEmbedder()
    )

    res = quality_analysis.compare_responses(["a", "b"], ["a", "b"])
    assert "avg_similarity" in res
    assert len(res["per_query"]) == 2
    assert res["avg_similarity"] > 0.99


def test_compare_responses_different(monkeypatch):
    monkeypatch.setattr(
        quality_analysis, "_get_embedder", lambda model_name=None: StubEmbedderDiff()
    )

    baseline = ["A1", "A2"]
    finetuned = ["B1", "B2"]
    res = quality_analysis.compare_responses(baseline, finetuned)
    assert res["avg_similarity"] < 0.5
    assert len(res["per_query"]) == 2
