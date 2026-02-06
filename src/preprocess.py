from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata
from typing import Iterable, List, Sequence

from .config import PreprocessConfig


SPANISH_STOPWORDS = {
    "de",
    "la",
    "que",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "se",
    "las",
    "por",
    "un",
    "para",
    "con",
    "no",
    "una",
    "su",
    "al",
    "lo",
    "como",
    "mas",
    "pero",
    "sus",
    "le",
    "ya",
    "o",
    "este",
    "si",
    "porque",
    "esta",
    "entre",
    "cuando",
    "muy",
    "sin",
    "sobre",
    "tambien",
    "me",
    "hasta",
    "hay",
    "donde",
    "quien",
    "desde",
    "todo",
    "nos",
    "durante",
    "todos",
    "uno",
    "les",
    "ni",
    "contra",
    "otros",
    "ese",
    "eso",
    "ante",
    "ellos",
    "e",
    "esto",
    "mi",
    "antes",
    "algunos",
    "unos",
    "yo",
    "otro",
    "otras",
    "otra",
    "tanto",
    "esa",
    "estos",
    "mucho",
    "quienes",
    "nada",
    "muchos",
    "cual",
    "poco",
    "ella",
    "estar",
    "estas",
    "algunas",
    "algo",
    "nosotros",
    "mis",
    "tu",
    "te",
    "ti",
    "tus",
    "ellas",
    "nosotras",
    "vosotros",
    "vosotras",
    "os",
    "mio",
    "mia",
    "mios",
    "mias",
    "tuyo",
    "tuya",
    "tuyos",
    "tuyas",
    "suyo",
    "suya",
    "suyos",
    "suyas",
    "nuestro",
    "nuestra",
    "nuestros",
    "nuestras",
    "vuestro",
    "vuestra",
    "vuestros",
    "vuestras",
    "esos",
    "esas",
    "estoy",
    "estan",
    "soy",
    "eres",
    "es",
    "somos",
    "son",
    "sea",
    "ser",
    "fui",
    "fue",
    "fuimos",
    "fueron",
    "tener",
    "tengo",
    "tiene",
    "tenemos",
    "tienen",
    "haber",
}


@dataclass
class Document:
    doc_id: str
    path: str
    text: str


def discover_documents(input_dir: Path | str, pattern: str = "**/*.txt") -> List[Path]:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    return sorted(base.glob(pattern))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def load_documents(paths: Sequence[Path], base_dir: Path | str) -> List[Document]:
    base = Path(base_dir)
    docs: List[Document] = []
    for path in paths:
        rel = path.relative_to(base).as_posix()
        doc_id = rel.replace("/", "__")
        docs.append(Document(doc_id=doc_id, path=rel, text=_read_text(path)))
    return docs


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def load_stopwords(config: PreprocessConfig) -> set[str]:
    stopwords = set(SPANISH_STOPWORDS)
    if config.stopwords_path:
        path = Path(config.stopwords_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                term = line.strip().lower()
                if term:
                    stopwords.add(term)
    for term in config.extra_stopwords:
        stopwords.add(str(term).strip().lower())
    if config.normalize_accents:
        stopwords = {_strip_accents(term) for term in stopwords}
    return stopwords


def preprocess_text(text: str, config: PreprocessConfig, stopwords: set[str]) -> List[str]:
    if config.lowercase:
        text = text.lower()
    if config.normalize_accents:
        text = _strip_accents(text)
    if config.remove_numbers:
        text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [
        token
        for token in tokens
        if len(token) >= config.min_token_len and token not in stopwords
    ]
    return tokens


def preprocess_documents(
    documents: Sequence[Document],
    config: PreprocessConfig,
) -> tuple[List[List[str]], List[str]]:
    stopwords = load_stopwords(config)
    token_lists: List[List[str]] = []
    joined_docs: List[str] = []
    for doc in documents:
        tokens = preprocess_text(doc.text, config, stopwords)
        token_lists.append(tokens)
        joined_docs.append(" ".join(tokens))
    return token_lists, joined_docs


def compute_corpus_fingerprint(paths: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        payload = f"{path.as_posix()}|{stat.st_size}|{stat.st_mtime}".encode("utf-8")
        hasher.update(payload)
    return hasher.hexdigest()
