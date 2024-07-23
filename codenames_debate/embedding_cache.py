import sqlite3
import threading

import numpy as np

DB_PATH = "embedding_cache.sqlite"


class EmbeddingCache:
    def __init__(self) -> None:
        self.local = threading.local()

    def get_connection(self) -> sqlite3.Connection:
        if hasattr(self.local, "conn"):
            return self.local.conn

        self.local.conn = conn = sqlite3.connect(DB_PATH, autocommit=True)
        self.local.insert_count = 0
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA optimize=0x10002")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            word TEXT PRIMARY KEY,
            embedding BLOB
        )
        """)
        return conn

    def insert(self, word: str, embedding: np.ndarray) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        embedding_blob = embedding.tobytes()
        cursor.execute(
            """INSERT OR REPLACE INTO embedding_cache (word, embedding) VALUES (?, ?)""",
            (word, embedding_blob),
        )
        self.local.insert_count += 1
        if self.local.insert_count % 1000 == 0:
            cursor.execute("PRAGMA optimize")

    def get(self, word: str) -> np.ndarray | None:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT embedding FROM embedding_cache WHERE word=?""",
            (word,),
        )
        result = cursor.fetchone()
        if result is not None:
            # Convert the binary data back to a NumPy array
            embedding_blob = result[0]
            return np.frombuffer(embedding_blob, dtype=np.float64)
        return None
