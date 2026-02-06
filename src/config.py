from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    normalize_accents: bool = True
    remove_numbers: bool = True
    min_token_len: int = 2
    stopwords_path: Optional[str] = None
    extra_stopwords: list[str] = field(default_factory=list)


@dataclass
class VectorizeConfig:
    max_features: Optional[int] = 5000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)


@dataclass
class LDAConfig:
    n_topics: int = 10
    max_iter: int = 20
    learning_method: str = "batch"
    random_state: int = 42
    doc_topic_prior: Optional[float] = None
    topic_word_prior: Optional[float] = None


@dataclass
class CacheConfig:
    enabled: bool = True
    dir: str = "outputs/cache"


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    language: str = field(default_factory=lambda: os.getenv("OPENAI_LANG", "es"))
    temperature: float = 0.2
    max_tokens: int = 200
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )
    organization: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_ORG")
    )
    project: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_PROJECT")
    )


@dataclass
class ReportConfig:
    top_words: int = 12
    report_path: str = "outputs/report.md"


@dataclass
class PipelineConfig:
    input_dir: str = "data/raw"
    output_dir: str = "outputs"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    vectorize: VectorizeConfig = field(default_factory=VectorizeConfig)
    lda: LDAConfig = field(default_factory=LDAConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(obj: Any, overrides: Dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: Optional[str] = None) -> PipelineConfig:
    config = PipelineConfig()
    if path:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _merge_dataclass(config, data)
    return config


def update_config(
    config: PipelineConfig,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_topics: Optional[int] = None,
    no_cache: bool = False,
    no_llm: bool = False,
) -> PipelineConfig:
    if input_dir:
        config.input_dir = input_dir
    if output_dir:
        config.output_dir = output_dir
    if n_topics is not None:
        config.lda.n_topics = int(n_topics)
    if no_cache:
        config.cache.enabled = False
    if no_llm:
        config.llm.enabled = False
    return config


def config_summary(config: PipelineConfig) -> Dict[str, Any]:
    return config.to_dict()
