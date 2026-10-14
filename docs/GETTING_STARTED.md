# Getting started

This guide takes you from a fresh machine to your first grounded answer, one step at a
time. No prior experience with RAG, databases, or AI is assumed.

If a term feels unfamiliar, keep [CONCEPTS.md](CONCEPTS.md) open next to you — it
explains every idea in plain English.

**What you'll have at the end:** a small program that has read your text, stored it in
a PostgreSQL database, and can answer questions about it — showing you the exact
snippets it used, and honestly saying "I cannot find sufficient information..." when
the answer isn't in your text.

---

## Before you start (checklist)

You'll need:

- A computer with permission to install software (Windows, macOS, or Linux).
- **Python 3.10 or newer.**
- **PostgreSQL** with the **pgvector** extension. (Don't worry — we'll set this up.)
- An **OpenAI API key** for the final answer-writing step.
  *(You can make the *embedding* step free and local, but writing the final answer
  uses OpenAI by default. We'll cover both.)*

Setup should take about 15–20 minutes the first time.

---

## Step 1 — Install Python

Check whether you already have a new-enough version:

```bash
python --version
```

If that prints `Python 3.10` or higher, you're set. If not, install it from
[python.org/downloads](https://www.python.org/downloads/).

> **Windows tip:** during installation, tick the box **"Add Python to PATH"**.

---

## Step 2 — Set up PostgreSQL with pgvector

You have two options. **Docker is by far the easiest**, so we lead with that.

### Option A — Docker (recommended)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) if you
don't have it, then run this one command:

```bash
docker run -d --name ragram-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
```

That starts a PostgreSQL server that **already includes pgvector**. Now create the
database RAGRAM Core expects:

```bash
docker exec -it ragram-pg psql -U postgres -c "CREATE DATABASE ragram;"
```

That's it. Skip to Step 3.

### Option B — Use your own PostgreSQL

If you already run PostgreSQL (locally or on a server), you just need to add the
pgvector extension. Connect as a superuser and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

How you *install* pending the pgvector extension itself differs by platform
(`apt install postgresql-16-pgvector`, `brew install pgvector`, or building from
source). If that feels heavy, use **Option A**.

> RAGRAM Core also tries to create this extension automatically when it connects.
> A superuser run is only needed the very first time, if the extension isn't present.

---

## Step 3 — Get your API key(s)

Create an account at [platform.openai.com](https://platform.openai.com/) and make an
API key. It looks like `sk-...`. Keep it secret — treat it like a password.

This key is used for:

- **Embeddings** (turning text into vectors), and
- **Answer generation** (writing the final reply).

*(Prefer free and local? See the box at the end of Step 5 to run embeddings on your
own machine.)*

---

## Step 4 — Get the code and install dependencies

Download or clone the project, then open a terminal **in the project folder** (the
one containing `main.py`). All commands below assume you're in that folder.

Create a virtual environment (an isolated "sandbox" for the project's libraries):

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked when your prompt shows `(.venv)`.

Now install the libraries:

```bash
pip install -r requirements.txt
```

---

## Step 5 — Create your `.env` file

The project reads its settings from a file named `.env`. Copy the example and open it
in any text editor:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Fill it in. If you used the Docker command above, the values are exactly these:

```dotenv
RAGRAM_PG_HOST=localhost
RAGRAM_PG_PORT=5432
RAGRAM_PG_USER=postgres
RAGRAM_PG_PASSWORD=postgres
RAGRAM_PG_DBNAME=ragram

RAGRAM_EMBEDDING_PROVIDER=openai
RAGRAM_EMBEDDING_MODEL=text-embedding-3-small
RAGRAM_EMBEDDING_DIM=1536

RAGRAM_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-paste-your-key-here
```

Save the file. That's all the configuration you need to start.

<details>
<summary><strong>Optional: run embeddings for free, on your own computer</strong></summary>

Set the embedding provider to HuggingFace and switch to a small local model:

```dotenv
RAGRAM_EMBEDDING_PROVIDER=huggingface
RAGRAM_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAGRAM_EMBEDDING_DIM=384
```

Then install the local model support:

```bash
pip install sentence-transformers
```

The first run downloads the model (a few dozen MB). You still need `OPENAI_API_KEY`
for the final answer-writing step.
</details>

---

## Step 6 — Run your first query 🎉

The verification script initializes the database, indexes a sample document, and runs
two queries:

```bash
python main.py
```

You should see something like this:

```
Connecting to PostgreSQL at localhost:5432...
Schema ready.
Indexed 3 chunk(s): ['a1b2c3:0', 'a1b2c3:1', 'a1b2c3:2']

======================================================================
QUERY 1 (in-scope): How does RAGRAM Core perform similarity search?
======================================================================
context_used: True
answer: Retrieval works by encoding the user query with the same embedding model...
sources:
  - a1b2c3:1  distance=0.1832  'Retrieval works by encoding the user query ...'

======================================================================
QUERY 2 (out-of-scope): What is the capital of France?
======================================================================
context_used: False
answer: I cannot find sufficient information in the current database context.
```

**What you're looking at:**

- **`Schema ready.`** — the `ragram_chunks` table and its index were created.
- **`Indexed 3 chunk(s)`** — the sample text was cut into 3 chunks, embedded, and
  stored.
- **QUERY 1** — an answer that *is* supported by the text. Notice `context_used: True`
  and the **source** with its distance (smaller = more relevant).
- **QUERY 2** — a question the text says nothing about. RAGRAM Core refuses instead of
  making something up: `context_used: False` and the standard "I cannot find..."
  message. **This refusal is a feature, not a failure.**

If you saw both queries behave like this, congratulations — you have a working RAG
pipeline.

---

## Step 7 — Try the web UI (optional but fun)

RAGRAM Core ships a simple Streamlit interface. From the same project folder:

```bash
streamlit run ui/app.py
```

Your browser opens at `http://localhost:8501`. You can:

- **Ask questions** in the **Query** tab — answers stream in word by word, and each
  answer shows the exact chunks it used.
- **Add your own text** in the **Ingest** tab — paste text or upload a `.txt`, `.md`,
  `.csv`, or `.json` file.
- Adjust **`top_k`** and **`max_distance`** in the sidebar to see how retrieval
  changes.

> If `streamlit` isn't found, use `python -m streamlit run ui/app.py`.

---

## Step 8 — Index your own documents

Ready to use your own content? Use the **Ingest** tab in the UI, or a few lines of
Python:

```python
from config import Settings
from embeddings import create_embedding_provider
from adapters.postgres_adapter import PostgresVectorAdapter
from engine import RagramCoreEngine
from llm import OpenAILLMClient

settings = Settings.from_env()
embedder = create_embedding_provider(settings)
storage = PostgresVectorAdapter(settings.sqlalchemy_url, dimension=embedder.dimension)
storage.connect()

engine = RagramCoreEngine(
    storage, embedder,
    OpenAILLMClient(settings.llm_model, settings.openai_api_key),
)

with open("my_notes.txt", encoding="utf-8") as f:
    engine.ingest_document(f.read(), metadata={"source": "my_notes.txt"})

result = engine.query("What does my document say about ...?")
print(result.answer)
for source in result.sources:
    print(source.chunk_id, source.distance)

storage.disconnect()
```

---

## What just happened?

In one sentence: **your text was turned into vectors, stored in PostgreSQL, and a
question was turned into a vector to find the closest pieces — which were then handed
to the AI to answer from.**

If you want the full explanation of each step, read
[CONCEPTS.md](CONCEPTS.md). If you want to understand the code layout, see the
[README](../README.md).

---

## Troubleshooting

| Message you might see | What it means | How to fix it |
| --- | --- | --- |
| `type "vector" does not exist` | pgvector isn't installed in that database. | Use the Docker image from Step 2, or run `CREATE EXTENSION vector;` as a superuser. |
| `connection refused` | Nothing is listening on that host/port. | Is the database running? Check `RAGRAM_PG_HOST` / `RAGRAM_PG_PORT` in `.env`. |
| `password authentication failed` | Wrong username or password. | Fix `RAGRAM_PG_USER` / `RAGRAM_PG_PASSWORD` in `.env`. |
| `database "ragram" does not exist` | You skipped creating it. | Run the `CREATE DATABASE ragram;` command from Step 2. |
| `ModuleNotFoundError: psycopg` | The Postgres driver isn't installed. | Run `pip install "psycopg[binary]>=3.1"`. |
| `ModuleNotFoundError: streamlit` | The UI library isn't installed. | Run `pip install streamlit`, then `python -m streamlit run ui/app.py`. |
| Authentication error mentioning `api_key` | OpenAI key missing or wrong. | Set `OPENAI_API_KEY` in `.env`. |
| **Every** query returns "I cannot find sufficient information..." | Either nothing was ingested, or `max_distance` is too strict. | Make sure you ingested a document; try setting `RAGRAM_MAX_DISTANCE=none` to see raw distances. |

---

## Next steps

- [ ] Index one of your own files and ask it a question.
- [ ] Read [CONCEPTS.md](CONCEPTS.md) to understand *why* it works.
- [ ] Skim the [README](../README.md) for architecture and how to swap the database,
      embeddings, or LLM.
- [ ] Try changing `RAGRAM_CHUNK_SIZE` and `RAGRAM_TOP_K` to feel how retrieval
      quality changes.
- [ ] Explore the code visually with
      [gitingest](https://gitingest.com/ramprakash-munugapati/ragram_core) (whole repo
      as one text file) or
      [gitdiagram](https://gitdiagram.com/ramprakash-munugapati/ragram_core) (how the
      code fits together).

Happy building!
