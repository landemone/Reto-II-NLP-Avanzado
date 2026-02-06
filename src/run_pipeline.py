from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .cache import CacheManager, hash_dict
from .config import load_config, update_config
from .interpret_top_words import get_top_words
from .interpret_with_llm import label_topics_with_llm
from .lda_infer import infer_doc_topics
from .lda_train import train_lda
from .openai_client import build_openai_client
from .preprocess import (
    compute_corpus_fingerprint,
    discover_documents,
    load_documents,
    preprocess_documents,
)
from .vectorize import fit_vectorizer


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_pickle(path: Path, obj) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path):
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def _build_report(
    report_path: Path,
    num_docs: int,
    n_topics: int,
    topics: List[Dict[str, object]],
    labels: Optional[Dict[int, str]],
    config_dict: Dict,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: List[str] = []
    lines.append("# LDA Report")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Documents: {num_docs}")
    lines.append(f"- Topics: {n_topics}")
    lines.append("")
    lines.append("## Topics")
    for topic in topics:
        topic_id = topic["topic_id"]
        label = labels.get(topic_id) if labels else None
        if label:
            lines.append(f"### Topic {topic_id} - {label}")
        else:
            lines.append(f"### Topic {topic_id}")
        words = ", ".join(topic["top_words"])
        lines.append(f"Top words: {words}")
        lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(config_dict, indent=2, ensure_ascii=True))
    lines.append("```")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    config_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_topics: Optional[int] = None,
    no_cache: bool = False,
    no_llm: bool = False,
) -> None:
    config = load_config(config_path)
    update_config(
        config,
        input_dir=input_dir,
        output_dir=output_dir,
        n_topics=n_topics,
        no_cache=no_cache,
        no_llm=no_llm,
    )

    input_dir_path = Path(config.input_dir)
    output_dir_path = Path(config.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cache = CacheManager(config.cache.dir, enabled=config.cache.enabled)

    paths = discover_documents(input_dir_path)
    if not paths:
        raise ValueError(f"No documents found in {input_dir_path}")

    fingerprint = compute_corpus_fingerprint(paths)

    preprocess_key = f"preprocess_{fingerprint}_{hash_dict(asdict(config.preprocess))}"
    if cache.exists(preprocess_key):
        preprocess_data = cache.load(preprocess_key)
        documents = preprocess_data["documents"]
        token_lists = preprocess_data["token_lists"]
        joined_docs = preprocess_data["joined_docs"]
    else:
        documents = load_documents(paths, input_dir_path)
        token_lists, joined_docs = preprocess_documents(documents, config.preprocess)
        cache.save(
            preprocess_key,
            {
                "documents": documents,
                "token_lists": token_lists,
                "joined_docs": joined_docs,
            },
        )

    vectorize_key = f"vectorize_{fingerprint}_{hash_dict(asdict(config.vectorize))}"
    if cache.exists(vectorize_key):
        vectorize_data = cache.load(vectorize_key)
        vectorizer = vectorize_data["vectorizer"]
        doc_term_matrix = vectorize_data["doc_term_matrix"]
    else:
        vectorizer, doc_term_matrix = fit_vectorizer(joined_docs, config.vectorize)
        cache.save(
            vectorize_key,
            {"vectorizer": vectorizer, "doc_term_matrix": doc_term_matrix},
        )

    lda_key = f"lda_{fingerprint}_{hash_dict(asdict(config.lda))}"
    if cache.exists(lda_key):
        model = cache.load(lda_key)
    else:
        model = train_lda(doc_term_matrix, config.lda)
        cache.save(lda_key, model)

    doc_topics_key = f"doc_topics_{fingerprint}_{hash_dict(asdict(config.lda))}"
    if cache.exists(doc_topics_key):
        doc_topics = cache.load(doc_topics_key)
    else:
        doc_topics = infer_doc_topics(model, doc_term_matrix)
        cache.save(doc_topics_key, doc_topics)

    topics = get_top_words(model, vectorizer, top_n=config.report.top_words)

    labels: Optional[List[Dict[str, object]]] = None
    label_map: Optional[Dict[int, str]] = None
    llm_error: Optional[str] = None
    if config.llm.enabled and config.llm.api_key:
        client = build_openai_client(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            organization=config.llm.organization,
            project=config.llm.project,
        )
        labels, llm_error = label_topics_with_llm(topics, config.llm, client)
        if labels:
            label_map = {item["topic_id"]: item["label"] for item in labels}
    else:
        if not config.llm.enabled:
            llm_error = "LLM disabled"
        elif not config.llm.api_key:
            llm_error = "OPENAI_API_KEY not set"

    config_dict = asdict(config)
    if "llm" in config_dict and config_dict["llm"].get("api_key"):
        config_dict["llm"]["api_key"] = None

    model_dir = output_dir_path / "lda_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    _save_pickle(model_dir / "model.pkl", model)
    _save_pickle(model_dir / "vectorizer.pkl", vectorizer)
    _write_json(model_dir / "config.json", config_dict)

    topic_rows = []
    for topic in topics:
        topic_id = topic["topic_id"]
        label = label_map.get(topic_id) if label_map else None
        topic_rows.append(
            {
                "topic_id": topic_id,
                "label": label or "",
                "top_words": ", ".join(topic["top_words"]),
            }
        )
    pd.DataFrame(topic_rows).to_csv(output_dir_path / "topics_top_words.csv", index=False)

    if labels:
        pd.DataFrame(labels).to_csv(
            output_dir_path / "topics_labels.csv", index=False
        )
    if llm_error:
        _write_text(output_dir_path / "llm_error.txt", llm_error)

    topic_cols = [f"topic_{i}" for i in range(config.lda.n_topics)]
    doc_df = pd.DataFrame(doc_topics, columns=topic_cols)
    doc_df.insert(0, "path", [doc.path for doc in documents])
    doc_df.insert(0, "doc_id", [doc.doc_id for doc in documents])
    doc_df["top_topic"] = doc_df[topic_cols].idxmax(axis=1)
    doc_df["top_score"] = doc_df[topic_cols].max(axis=1)
    doc_df.to_csv(output_dir_path / "doc_topics.csv", index=False)

    _build_report(
        report_path=Path(config.report.report_path),
        num_docs=len(documents),
        n_topics=config.lda.n_topics,
        topics=topics,
        labels=label_map,
        config_dict=config_dict,
    )

    print("Pipeline completed.")
    print(f"Outputs saved to: {output_dir_path}")
    if llm_error:
        print(f"LLM warning: {llm_error}")
        print(f"Details saved to: {output_dir_path / 'llm_error.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LDA pipeline.")
    parser.add_argument("--config", help="Path to JSON config file", default=None)
    parser.add_argument("--input-dir", help="Override input dir", default=None)
    parser.add_argument("--output-dir", help="Override output dir", default=None)
    parser.add_argument("--topics", help="Override number of topics", default=None)
    parser.add_argument("--no-cache", help="Disable cache", action="store_true")
    parser.add_argument("--no-llm", help="Disable LLM labeling", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_topics=args.topics,
        no_cache=args.no_cache,
        no_llm=args.no_llm,
    )


if __name__ == "__main__":
    main()
