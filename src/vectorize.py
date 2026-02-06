from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer

from .config import VectorizeConfig


def fit_vectorizer(
    documents: List[str], config: VectorizeConfig
) -> Tuple[CountVectorizer, "scipy.sparse.csr_matrix"]:
    vectorizer = CountVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=config.ngram_range,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def transform_documents(
    vectorizer: CountVectorizer, documents: List[str]
) -> "scipy.sparse.csr_matrix":
    return vectorizer.transform(documents)
