from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from .config import LLMConfig


def _build_prompt(topics: List[Dict[str, object]], language: str) -> str:
    lines = []
    for topic in topics:
        words = ", ".join(topic["top_words"])
        lines.append(f"Topic {topic['topic_id']}: {words}")
    joined = "\n".join(lines)
    lang = language.strip().lower()
    if lang.startswith("es"):
        instructions = (
            "Etiqueta cada topico con una frase corta y clara (3-6 palabras) "
            "en espanol. Devuelve un arreglo JSON de objetos con claves: "
            "topic_id (int), label (str). Usa solo caracteres ASCII."
        )
    else:
        instructions = (
            "Label each topic with a short, clear phrase (3-6 words). "
            "Return a JSON array of objects with keys: topic_id (int), label (str). "
            "Use ASCII characters only."
        )
    return f"{instructions}\n\n{joined}"


def _extract_json_array(text: str) -> Optional[List[Dict[str, object]]]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    labels = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "topic_id" in item and "label" in item:
            labels.append({"topic_id": int(item["topic_id"]), "label": str(item["label"])})
    return labels or None


def _parse_labels(text: str) -> Optional[List[Dict[str, object]]]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            labels = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                if "topic_id" in item and "label" in item:
                    labels.append(
                        {"topic_id": int(item["topic_id"]), "label": str(item["label"])}
                    )
            if labels:
                return labels
    except json.JSONDecodeError:
        pass

    extracted = _extract_json_array(text)
    if extracted:
        return extracted

    labels: List[Dict[str, object]] = []
    for line in text.splitlines():
        match = re.match(r"\s*Topic\s+(\d+)\s*[:\-]\s*(.+)", line)
        if match:
            labels.append(
                {"topic_id": int(match.group(1)), "label": match.group(2).strip()}
            )
            continue
        match = re.match(r"\s*(\d+)\s*[:\-]\s*(.+)", line)
        if match:
            labels.append(
                {"topic_id": int(match.group(1)), "label": match.group(2).strip()}
            )
    return labels or None


def label_topics_with_llm(
    topics: List[Dict[str, object]],
    config: LLMConfig,
    client,
) -> Tuple[Optional[List[Dict[str, object]]], Optional[str]]:
    if not config.enabled:
        return None, "LLM disabled"
    if not config.api_key:
        return None, "OPENAI_API_KEY not set"
    prompt = _build_prompt(topics, config.language)
    try:
        lang = config.language.strip().lower()
        if lang.startswith("es"):
            system_message = "Eres un asistente que etiqueta topicos para analistas. Responde en espanol."
        else:
            system_message = "You label topic models for analysts."
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        content = response.choices[0].message.content or ""
        labels = _parse_labels(content)
        if not labels:
            return None, "Could not parse LLM response"
        return labels, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
