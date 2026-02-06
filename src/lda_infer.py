from __future__ import annotations

from sklearn.decomposition import LatentDirichletAllocation


def infer_doc_topics(
    model: LatentDirichletAllocation, doc_term_matrix
):
    return model.transform(doc_term_matrix)
