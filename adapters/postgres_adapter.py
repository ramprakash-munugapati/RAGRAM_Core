"""PostgreSQL + pgvector implementation of the storage interface.

Schema (table `ragram_chunks`):
    chunk_id    TEXT PRIMARY KEY   -- stable id, e.g. "<doc_id>:<index>"
    text        TEXT NOT NULL      -- raw chunk text
    embedding   VECTOR(dim)        -- pgvector column, cosine-indexed via HNSW
    meta        JSONB              -- structural metadata (source, doc id, ...)
    created_at  TIMESTAMPTZ        -- server-side insert timestamp
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert

from adapters.base import RetrievedChunk, VectorStorageInterface


class PostgresVectorAdapter(VectorStorageInterface):
    """Stores and retrieves chunk embeddings in PostgreSQL using pgvector.

    All SQL is executed through SQLAlchemy Core with psycopg3; the pgvector
    `Vector` type handles bidirectional serialization (list[float] <-> vector)
    and the `cosine_distance` comparator compiles to the pgvector `<=>`
    operator (equivalent to raw SQL: `ORDER BY embedding <=> :query LIMIT k`).
    """

    def __init__(self, db_url: str, *, dimension: int, table_name: str = "ragram_chunks") -> None:
        """`dimension` must equal the embedding model's output dimensionality;
        it pins the VECTOR(dim) column type and is validated on every write."""
        self._db_url = db_url
        self._dimension = dimension
        self._table_name = table_name
        self._engine: sa.Engine | None = None
        self._table: sa.Table | None = None

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    @property
    def engine(self) -> sa.Engine:
        if self._engine is None:
            raise RuntimeError("PostgresVectorAdapter is not connected; call connect() first.")
        return self._engine

    @property
    def table(self) -> sa.Table:
        if self._table is None:
            raise RuntimeError("PostgresVectorAdapter is not connected; call connect() first.")
        return self._table

    def connect(self) -> None:
        """Create the engine, enable pgvector, and ensure the schema exists."""
        # pool_pre_ping recovers dead connections after DB restarts/failovers.
        self._engine = sa.create_engine(self._db_url, pool_pre_ping=True)

        with self.engine.begin() as conn:
            # Idempotent; requires the pgvector extension binaries on the server.
            # Without this, the `vector` column type below is unavailable.
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

        self._table = self._build_table()
        # checkfirst=True makes re-runs safe against an existing production DB.
        self._table.create(self.engine, checkfirst=True)

        # HNSW approximate nearest-neighbour index tuned for cosine distance.
        # vector_cosine_ops is the pgvector operator class required for `<=>`.
        self._build_hnsw_index().create(self.engine, checkfirst=True)

    def disconnect(self) -> None:
        """Dispose the connection pool."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._table = None

    # ------------------------------------------------------------------ #
    # DDL helpers
    # ------------------------------------------------------------------ #

    def _build_table(self) -> sa.Table:
        """Define the chunks table. A fixed VECTOR(dim) is required so the
        HNSW index can be created (untyped VECTOR columns cannot be indexed)."""
        metadata = sa.MetaData()
        return sa.Table(
            self._table_name,
            metadata,
            sa.Column("chunk_id", sa.Text, primary_key=True),
            sa.Column("text", sa.Text, nullable=False),
            sa.Column("embedding", Vector(self._dimension), nullable=False),
            sa.Column("meta", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )

    def _build_hnsw_index(self) -> sa.Index:
        """HNSW index -> fast ANN search. Compiles to:
        CREATE INDEX ... ON ragram_chunks USING hnsw (embedding vector_cosine_ops)
        """
        return sa.Index(
            f"ix_{self._table_name}_embedding_hnsw",
            self.table.c.embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )

    def _validate_embedding(self, embedding: list[float]) -> None:
        """Guard against dimension drift between config and embedding model."""
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding has dimension {len(embedding)}, but the storage "
                f"schema expects {self._dimension}. Fix RAGRAM_EMBEDDING_DIM "
                "to match your embedding model."
            )

    # ------------------------------------------------------------------ #
    # VectorStorageInterface implementation
    # ------------------------------------------------------------------ #

    def insert_chunk(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Idempotent upsert keyed on chunk_id: re-ingesting a document
        refreshes its chunks instead of duplicating them."""
        self._validate_embedding(embedding)

        stmt = (
            pg_insert(self.table)
            .values(
                chunk_id=chunk_id,
                text=text,
                # pgvector's bind processor serializes list[float] -> vector literal.
                embedding=embedding,
                meta=metadata,
            )
            .on_conflict_do_update(
                index_elements=["chunk_id"],
                # On conflict, overwrite payload columns with the incoming values.
                set_={
                    "text": text,
                    "embedding": embedding,
                    "meta": metadata,
                },
            )
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """Cosine-distance k-NN search.

        Compiles to (valid pgvector SQL):
            SELECT chunk_id, text, meta, embedding <=> :query AS distance
            FROM ragram_chunks
            ORDER BY embedding <=> :query
            LIMIT :limit
        `<=>` is cosine distance in [0, 2]; 0 = identical direction.
        The ORDER BY uses the HNSW vector_cosine_ops index above.
        """
        self._validate_embedding(query_embedding)

        distance_expr = self.table.c.embedding.cosine_distance(query_embedding)
        stmt = (
            sa.select(
                self.table.c.chunk_id,
                self.table.c.text,
                self.table.c.meta,
                distance_expr.label("distance"),
            )
            .order_by(distance_expr)
            .limit(limit)
        )

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()

        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                text=row["text"],
                metadata=dict(row["meta"] or {}),
                distance=float(row["distance"]),
            )
            for row in rows
        ]
