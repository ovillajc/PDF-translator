import sqlite3
import os
import hashlib
from typing import Optional, List, Dict, Tuple

class TranslationCache:
    """Caché persistente en SQLite para reanudar libros largos al instante y evitar retraducciones."""

    def __init__(self, cache_path: Optional[str] = None):
        if cache_path is None:
            cache_dir = os.path.expanduser("~/.cache/pdf_traductor_mlx")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "translations_cache.sqlite3")
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS translations ("
            "  key TEXT PRIMARY KEY,"
            "  source_text TEXT,"
            "  translated_text TEXT"
            ")"
        )
        self.conn.commit()

    @staticmethod
    def _make_key(model_id: str, src_lang: str, tgt_lang: str, text: str) -> str:
        raw = f"{model_id}|{src_lang}|{tgt_lang}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_many(self, model_id: str, src_lang: str, tgt_lang: str, texts: List[str]) -> Dict[str, str]:
        if not texts:
            return {}
        key_to_text = {self._make_key(model_id, src_lang, tgt_lang, t): t for t in texts}
        keys = list(key_to_text.keys())
        found: Dict[str, str] = {}
        # Consultar en bloques de 500 para respetar límites de SQLite
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            placeholders = ",".join("?" for _ in chunk)
            cur = self.conn.execute(
                f"SELECT key, translated_text FROM translations WHERE key IN ({placeholders})",
                chunk,
            )
            for k, val in cur.fetchall():
                orig_t = key_to_text.get(k)
                if orig_t is not None and val:
                    found[orig_t] = val
        return found

    def put_many(self, model_id: str, src_lang: str, tgt_lang: str, pairs: List[Tuple[str, str]]):
        if not pairs:
            return
        rows = [
            (self._make_key(model_id, src_lang, tgt_lang, src), src, tgt)
            for src, tgt in pairs
            if src and tgt
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO translations (key, source_text, translated_text) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
