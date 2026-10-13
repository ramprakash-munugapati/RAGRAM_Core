"""RAGRAM Core verification script.

Initializes the pgvector schema, ingests a sample multi-line document,
then runs an in-scope query and an out-of-scope query (to demonstrate the
anti-hallucination refusal path).

Run:
    1. cp .env.example .env  and fill in credentials
    2. pip install -r requirements.txt
    3. python main.py
"""

from __future__ import annotations

import sys

import sqlalchemy as sa

from adapters.postgres_adapter import PostgresVectorAdapter
from config import Settings
from embeddings import create_embedding_provider
from engine import RagramCoreEngine
from llm import OpenAILLMClient

# A multi-line sample document to index (domain-agnostic demo content).
SAMPLE_DOCUMENT = """\
RAGRAM Core is the foundational tier of the RAGRAM framework. It implements a
baseline Naive RAG (retrieval-augmented generation) pipeline on top of PostgreSQL.

The system stores text chunks together with their vector embeddings in a table
managed by the pgvector extension. Chunks are created by a recursive character
splitter that respects paragraph and sentence boundaries while preserving a
configurable overlap between neighbouring chunks.

Retrieval works by encoding the user query with the same embedding model used at
ingestion time, then performing a cosine-distance nearest-neighbour search. An
HNSW index using the vector_cosine_ops operator class accelerates this lookup.

The storage layer is defined by the VectorStorageInterface abstract base class.
The default implementation, PostgresVectorAdapter, can be replaced with another
backend without changing the RagramCoreEngine coordinator.

The final answer is produced by an LLM synthesis stage that is strictly grounded
in the retrieved context. If the knowledge base does not contain the answer, the
system refuses to respond rather than fabricating information.
"""


def _print_result(title: str, result) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(f"context_used: {result.context_used}")
    print(f"answer: {result.answer}")
    if result.sources:
        print("sources:")
        for chunk in result.sources:
            preview = chunk.text[:80].replace("\n", " ")
            print(f"  - {chunk.chunk_id}  distance={chunk.distance:.4f}  {preview!r}")


def main() -> int:
    # 1. Load and validate configuration from environment variables / .env.
    settings = Settings.from_env()

    # 2. Build the embedding provider (its dimension pins the VECTOR column).
    embedder = create_embedding_provider(settings)

    # 3. Instantiate the storage adapter and initialize the schema.
    storage = PostgresVectorAdapter(settings.sqlalchemy_url, dimension=embedder.dimension)
    synthesizer = OpenAILLMClient(model=settings.llm_model, api_key=settings.openai_api_key)

    try:
        print(f"Connecting to PostgreSQL at {settings.pg_host}:{settings.pg_port}...")
        storage.connect()  # enables pgvector, creates table + HNSW index
        print("Schema ready.")

        engine = RagramCoreEngine(
            storage=storage,
            embedder=embedder,
            synthesizer=synthesizer,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.top_k,
            max_distance=settings.max_distance,
        )

        # 4. Index the sample document.
        chunk_ids = engine.ingest_document(SAMPLE_DOCUMENT, {"source": "main.py:sample"})
        print(f"Indexed {len(chunk_ids)} chunk(s): {chunk_ids}")

        # 5. In-scope query: should be answered from retrieved context.
        _print_result(
            "QUERY 1 (in-scope): How does RAGRAM Core perform similarity search?",
            engine.query("How does RAGRAM Core perform similarity search?"),
        )

        # 6. Out-of-scope query: should trigger the refusal path.
        _print_result(
            "QUERY 2 (out-of-scope): What is the capital of France?",
            engine.query("What is the capital of France?"),
        )
    except sa.exc.OperationalError as exc:
        print(
            "\nDatabase connection failed.\n"
            "Check RAGRAM_PG_* settings and ensure the pgvector extension is\n"
            "available on the server (as a superuser once):\n"
            "    CREATE EXTENSION IF NOT EXISTS vector;\n"
            f"\nUnderlying error: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        storage.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
