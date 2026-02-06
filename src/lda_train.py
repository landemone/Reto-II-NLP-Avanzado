from __future__ import annotations

from sklearn.decomposition import LatentDirichletAllocation

from .config import LDAConfig


def train_lda(
    doc_term_matrix, config: LDAConfig
) -> LatentDirichletAllocation:
    model = LatentDirichletAllocation(
        n_components=config.n_topics,
        max_iter=config.max_iter,
        learning_method=config.learning_method,
        random_state=config.random_state,
        doc_topic_prior=config.doc_topic_prior,
        topic_word_prior=config.topic_word_prior,
    )
    model.fit(doc_term_matrix)
    return model
