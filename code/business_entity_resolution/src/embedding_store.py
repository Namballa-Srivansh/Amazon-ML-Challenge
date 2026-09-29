"""
embedding_store.py
===================
SQLite-backed key-value store for text -> embedding vector, replacing the
in-memory Python dict cache that caused the 22GB OOM.

Why this fixes the root cause, not just the symptom:
  - A full corpus of embeddings (3.8M S23 texts, 384-dim) is ~11GB+ in
    float16, before candidates, TF-IDF matrices, or the model itself are
    even loaded. That data volume is real -- no amount of gc.collect() or
    smaller dtypes changes the fact that it doesn't fit in 16GB RAM
    alongside everything else the pipeline needs at the same time.
  - The actual fix is to never hold the full corpus in RAM at all.
    get_many() below fetches ONLY the vectors you ask for -- RAM cost is
    proportional to the current chunk's candidate count, never the full
    corpus. SQLite does the random-access disk I/O; Python never sees
    more than one chunk's worth of vectors at a time.
  - Encoding is also now crash-safe AND resumable across separate process
    runs for free: every batch commits immediately (WAL journal mode), so
    killing the process mid-encode loses at most one batch (a few
    seconds), and the next run of ANY script that uses this store picks
    up exactly where it left off -- no separate pickle-cache bookkeeping
    needed, no "did the cache save in time" risk.

Vectors are stored as float16 (half the size of float32, negligible
precision cost for cosine similarity at this scale).
"""

import sqlite3
import numpy as np


class EmbeddingStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")     # crash-safe incremental commits
        self.conn.execute("PRAGMA synchronous=NORMAL")   # WAL makes this safe + much faster
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (text TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        self.conn.commit()

    def _missing(self, texts: list) -> list:
        """Which of these texts are NOT yet stored. Queried in batches of
        500 to stay under SQLite's default ~999 host-parameter limit."""
        uniq = list(dict.fromkeys(texts))  # dedupe, keep order
        have = set()
        for i in range(0, len(uniq), 500):
            batch = uniq[i:i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT text FROM embeddings WHERE text IN ({placeholders})", batch
            ).fetchall()
            have.update(r[0] for r in rows)
        return [t for t in uniq if t not in have]

    def encode_and_store_missing(self, texts: list, embedder, batch_size: int = 256,
                                  show_progress: bool = True):
        """Encodes only texts not already in the store. Commits every
        batch -- an interrupted run loses at most one batch of GPU work,
        and a re-run of this method (even from a different script) skips
        everything already encoded, automatically."""
        todo = self._missing(texts)
        if not todo:
            return
        n_batches = (len(todo) + batch_size - 1) // batch_size
        for bi, start in enumerate(range(0, len(todo), batch_size)):
            batch = todo[start:start + batch_size]
            vecs = embedder.encode(batch, batch_size=batch_size, convert_to_numpy=True).astype(np.float16)
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (text, vector) VALUES (?, ?)",
                [(t, v.tobytes()) for t, v in zip(batch, vecs)]
            )
            self.conn.commit()
            if show_progress and ((bi + 1) % 10 == 0 or (bi + 1) == n_batches):
                print(f"    embedding store: {bi+1}/{n_batches} batches "
                      f"({start+len(batch):,}/{len(todo):,} new texts encoded)")

    def get_many(self, texts: list) -> dict:
        """Fetches vectors for exactly these texts, and only these.
        RAM cost is O(len(texts)), never O(corpus size) -- this is the
        core fix for the 22GB dict-in-RAM problem."""
        out = {}
        uniq = list(dict.fromkeys(texts))
        for i in range(0, len(uniq), 500):
            batch = uniq[i:i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT text, vector FROM embeddings WHERE text IN ({placeholders})", batch
            ).fetchall()
            for text, blob in rows:
                out[text] = np.frombuffer(blob, dtype=np.float16)
        return out

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def close(self):
        self.conn.close()
