import logging
import sqlite3
import threading

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = "embedding_cache.sqlite"


class EmbeddingCache:
    def __init__(self) -> None:
        self.local = threading.local()

    def get_connection(self) -> sqlite3.Connection:
        if hasattr(self.local, "conn"):
            return self.local.conn

        logger.info("Connecting to SQLite embedding cache")
        self.local.conn = conn = sqlite3.connect(DB_PATH, autocommit=True)
        self.local.insert_count = 0
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            word TEXT PRIMARY KEY,
            embedding BLOB
        )
        """)
        cursor.close()
        return conn

    def insert(self, word: str, embedding: np.ndarray) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        embedding_blob = embedding.tobytes()
        try:
            cursor.execute(
                """INSERT OR REPLACE INTO embedding_cache (word, embedding) VALUES (?, ?)""",
                (word, embedding_blob),
            )
            self.local.insert_count += 1
            if self.local.insert_count % 1000 == 0:
                logger.info("Optimizing embedding cache")
                cursor.execute("PRAGMA optimize")
        except sqlite3.OperationalError:
            logger.error("Timeout inserting into SQLite embedding cache")
        finally:
            cursor.close()

    def get(self, word: str) -> np.ndarray | None:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """SELECT embedding FROM embedding_cache WHERE word=?""",
                (word,),
            )
            result = cursor.fetchone()
            if result is not None:
                # Convert the binary data back to a NumPy array
                embedding_blob = result[0]
                return np.frombuffer(embedding_blob, dtype=np.float64)
        except sqlite3.OperationalError:
            logger.error("Timeout reading from SQLite embedding cache")
        finally:
            cursor.close()
