from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def hash_dict(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CacheManager:
    def __init__(self, base_dir: Path | str, enabled: bool = True) -> None:
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.base_dir / f"{key}.pkl"

    def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        return self.path_for(key).exists()

    def load(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def save(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self.path_for(key)
        with path.open("wb") as f:
            pickle.dump(value, f)
