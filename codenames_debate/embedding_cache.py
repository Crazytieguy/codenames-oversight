import sqlite3
import threading

import numpy as np

DB_PATH = "embedding_cache.sqlite"


class EmbeddingCache:
    def __init__(self) -> None:
        self.local = threading.local()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                word TEXT PRIMARY KEY,
                embedding BLOB
            )
            """)

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(DB_PATH)
        return self.local.conn

    def insert(self, word: str, embedding: np.ndarray) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            embedding_blob = embedding.tobytes()
            cursor.execute(
                """
            INSERT OR REPLACE INTO embedding_cache (word, embedding) VALUES (?, ?)
            """,
                (word, embedding_blob),
            )

    def get(self, word: str) -> np.ndarray | None:
        with self.get_connection():
            cursor = self.get_connection().cursor()
            cursor.execute(
                """
            SELECT embedding FROM embedding_cache WHERE word=?
            """,
                (word,),
            )
            result = cursor.fetchone()
        if result:
            # Convert the binary data back to a NumPy array
            embedding_blob = result[0]
            return np.frombuffer(embedding_blob, dtype=np.float64)
        return None
