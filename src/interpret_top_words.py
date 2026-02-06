from __future__ import annotations

from typing import Dict, List

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def get_top_words(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_n: int = 10,
) -> List[Dict[str, object]]:
    feature_names = vectorizer.get_feature_names_out()
    topics: List[Dict[str, object]] = []
    for topic_id, weights in enumerate(model.components_):
        top_indices = weights.argsort()[:-top_n - 1 : -1]
        words = [feature_names[i] for i in top_indices]
        topics.append({"topic_id": topic_id, "top_words": words})
    return topics
