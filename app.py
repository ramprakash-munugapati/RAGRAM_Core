"""RAGRAM Core web UI (Streamlit).

A thin front-end over `RagramCoreEngine`: ask grounded questions with streamed
answers and source inspection, and index new documents.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import atexit
import json

import streamlit as st

from adapters.base import RetrievedChunk
from adapters.postgres_adapter import PostgresVectorAdapter
from config import Settings
from embeddings import EmbeddingProvider, create_embedding_provider
from engine import QueryStream, RagramCoreEngine
from llm import OpenAILLMClient
from prompts import NO_CONTEXT_ANSWER

st.set_page_config(page_title="RAGRAM Core", layout="wide")


# Heavy components (DB pool, embedding model, LLM client) are built once per
# session and reused across reruns. Streamlit reruns the script on every
# interaction, so caching here avoids reconnecting / reloading models.
@st.cache_resource(show_spinner="Connecting to PostgreSQL and loading models...")
def load_components() -> tuple[Settings, EmbeddingProvider, PostgresVectorAdapter, OpenAILLMClient]:
    settings = Settings.from_env()
    embedder = create_embedding_provider(settings)
    storage = PostgresVectorAdapter(settings.sqlalchemy_url, dimension=embedder.dimension)
    storage.connect()  # enables pgvector, creates the table + HNSW index
    atexit.register(storage.disconnect)
    synthesizer = OpenAILLMClient(model=settings.llm_model, api_key=settings.openai_api_key)
    return settings, embedder, storage, synthesizer


def build_engine(
    settings: Settings,
    embedder: EmbeddingProvider,
    storage: PostgresVectorAdapter,
    synthesizer: OpenAILLMClient,
    *,
    top_k: int,
    max_distance: float | None,
) -> RagramCoreEngine:
    """Engine is cheap to rebuild, so it reflects the live sidebar settings."""
    return RagramCoreEngine(
        storage,
        embedder,
        synthesizer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=top_k,
        max_distance=max_distance,
    )


def render_sources(sources: list[RetrievedChunk]) -> None:
    """Show retrieved chunks with their cosine distance and metadata."""
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            st.markdown(f"**`{source.chunk_id}`** · distance `{source.distance:.4f}`")
            st.caption(json.dumps(source.metadata, default=str))
            preview = source.text if len(source.text) <= 900 else source.text[:900] + "..."
            st.text(preview)
            st.divider()


def main() -> None:
    st.title("RAGRAM Core")
    st.caption("Vector-only Naive RAG over PostgreSQL + pgvector")

    try:
        settings, embedder, storage, synthesizer = load_components()
    except Exception as exc:  # noqa: BLE001 - surface any startup failure to the user
        st.error("Could not initialize RAGRAM Core. Check your `.env` / database, then reload.")
        st.exception(exc)
        st.stop()

    # ---------------------------------------------------------------- #
    # Sidebar: connection status + live retrieval settings
    # ---------------------------------------------------------------- #
    with st.sidebar:
        st.header("Status")
        st.success(f"Connected · {settings.pg_host}:{settings.pg_port}/{settings.pg_dbname}")
        st.caption(
            f"Embeddings: `{settings.embedding_provider}` · "
            f"`{settings.embedding_model}` · dim {embedder.dimension}"
        )
        st.caption(f"LLM: `{settings.llm_model}`")
        st.caption("Table: `ragram_chunks`")

        st.divider()
        st.header("Retrieval settings")
        top_k = st.slider("top_k", min_value=1, max_value=20, value=int(settings.top_k))
        filter_distance = st.checkbox(
            "Filter by max distance", value=settings.max_distance is not None
        )
        default_distance = settings.max_distance if settings.max_distance is not None else 0.5
        max_distance = st.slider(
            "max_distance (cosine)",
            min_value=0.0,
            max_value=1.5,
            value=float(default_distance),
            step=0.01,
            disabled=not filter_distance,
        )
        st.caption("Lower = stricter relevance (0 = identical, 2 = opposite).")

    engine = build_engine(
        settings,
        embedder,
        storage,
        synthesizer,
        top_k=top_k,
        max_distance=max_distance if filter_distance else None,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    tab_query, tab_ingest = st.tabs(["Query", "Ingest"])
    with tab_query:
        _render_query_tab(engine)
    with tab_ingest:
        _render_ingest_tab(engine)


def _render_query_tab(engine: RagramCoreEngine) -> None:
    # A form (rather than st.chat_input) works inside a tab and keeps history.
    with st.form("query_form", clear_on_submit=True):
        question = st.text_area(
            "Ask a question",
            height=90,
            placeholder="e.g. How does RAGRAM Core perform similarity search?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    # Replay prior turns (including their sources).
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    if submitted and question.strip():
        _answer(engine, question.strip())


def _answer(engine: RagramCoreEngine, question: str) -> None:
    """Stream a grounded answer and persist the turn to session history."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    stream: QueryStream = engine.stream_query(question)

    with st.chat_message("assistant"):
        if not stream.context_used:
            answer = NO_CONTEXT_ANSWER
            st.markdown(answer)
        else:
            # Consume the token iterator and render progressively.
            answer = st.write_stream(stream.tokens)
        if stream.sources:
            render_sources(stream.sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": stream.sources,
            "context_used": stream.context_used,
        }
    )


def _render_ingest_tab(engine: RagramCoreEngine) -> None:
    st.subheader("Index a document")
    st.caption(
        "Paste text or upload a file (`.txt`, `.md`, `.csv`, `.json`). Chunks are "
        "embedded and upserted into `ragram_chunks` (idempotent by document id)."
    )

    with st.form("ingest_form", clear_on_submit=False):
        pasted = st.text_area("Paste text", height=200, placeholder="Paste document text here...")
        uploaded = st.file_uploader(
            "or upload a file", type=["txt", "md", "markdown", "csv", "json"]
        )
        source = st.text_input("Source label (metadata)", value="manual")
        submitted = st.form_submit_button("Ingest", type="primary")

    if not submitted:
        return

    content = ""
    title = "pasted text"
    if uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="replace")
        title = uploaded.name
    elif pasted.strip():
        content = pasted

    if not content.strip():
        st.warning("Provide pasted text or upload a file before ingesting.")
        return

    with st.spinner("Chunking, embedding, and storing..."):
        chunk_ids = engine.ingest_document(content, metadata={"source": source, "title": title})

    st.success(f"Indexed {len(chunk_ids)} chunk(s) from {title}.")
    with st.expander("Stored chunk ids"):
        st.code("\n".join(chunk_ids))


if __name__ == "__main__":
    main()
