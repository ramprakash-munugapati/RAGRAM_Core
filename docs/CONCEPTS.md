# Concepts

You do **not** need any machine-learning background to read this page. Every idea is
explained from scratch, with everyday comparisons. By the end you'll understand what
RAGRAM Core does "under the hood" and why each piece exists.

If you'd rather jump straight to running the code, read
[GETTING_STARTED.md](GETTING_STARTED.md) first and come back here later.

---

## 1. What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**.

Let's unpack that name, word by word, because it *is* the whole idea:

- **Retrieval** — looking things up. Like searching a book for the pages that match
  your question.
- **Augmented** — added to, or enhanced with. We take those pages and *add them to*
  the question we're about to ask.
- **Generation** — writing an answer. A large language model (an "LLM", like GPT)
  writes the final reply.

So RAG means: **look up relevant text first, hand it to the AI, and ask the AI to
answer using only that text.**

The AI is not answering from memory. It's answering from *your* documents, which
you just handed it. That small change is what makes answers trustworthy and
traceable.

---

## 2. Why not just ask the AI directly?

Modern AI models "know" a lot, but they have three real problems for anything
serious:

1. **They don't know your private data.** Your company handbook, your database, your
   notes — none of that was in their training.
2. **They can make things up.** When a model doesn't know something, it sometimes
   invents a confident-sounding answer. This is called **hallucination**.
3. **Their knowledge is frozen in time.** They don't know about a document you wrote
   yesterday.

RAG fixes all three:

- You **give** the model your data (so it knows your private content).
- You tell it to answer **only** from what you gave it (so it can't invent facts, and
  can honestly say "I don't know").
- Your data lives in your database and is looked up fresh **on every question** (so it
  can be as new as you want).

Think of the difference this way:

- **Plain AI:** "Tell me about the company holiday policy." → The model guesses or
  makes something up.
- **RAG:** "Here are the three policy documents that match. Answer using only these."
  → The model answers from the documents, and we can show you exactly which ones.

---

## 3. The big idea in one picture

```
        You ask a question
                │
                ▼
   ┌──────────────────────────┐
   │  Find the most relevant  │   <-- "Retrieval"
   │  pieces of your text     │
   └──────────────────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │  Add them to the prompt  │   <-- "Augmented"
   └──────────────────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │  AI writes the answer,   │   <-- "Generation"
   │  using only those pieces │
   └──────────────────────────┘
```

The hard part is the first box: *how does a computer know which pieces of text are
"relevant" to a question?* Humans read and decide. Computers need numbers. That's
where embeddings come in.

---

## 4. Embeddings: turning meaning into numbers

A computer can't directly compare "How do I reset my password?" with a paragraph
about login problems. It needs both to become numbers.

An **embedding** is a way of turning a piece of text into a list of numbers that
captures its **meaning**. Same idea → similar numbers. Different idea → different
numbers.

A helpful mental picture: imagine every possible meaning has a location on a giant
map. "How do I reset my password?" and "Steps to change your login credentials" land
**near each other** on that map, because they mean similar things — even though they
share almost no words. Meanwhile "the melting point of iron" lands somewhere far
away.

The tool that turns text into a location on that "meaning map" is called an
**embedding model**. RAGRAM Core supports two:

- **OpenAI embeddings** — call a paid API (very accurate, costs a tiny amount).
- **HuggingFace `sentence-transformers`** — run a small model on your own computer
  (free, private, slightly less accurate).

You can swap between them with one setting.

---

## 5. Vectors: the "coordinates of meaning"

The list of numbers an embedding model produces is called a **vector**.

If an embedding is a *location* on the meaning map, the vector is that location's
**coordinates**.

A real example — one small model turns a sentence into 384 numbers that look like
this (shortened here):

```
"How do I reset my password?"  →  [0.021, -0.34, 0.88, ..., 0.07]
"Steps to change login passwords" → [0.019, -0.31, 0.85, ..., 0.09]
"The melting point of iron"    →  [-0.77, 0.12, -0.45, ..., 0.62]
```

Notice the first two vectors are *similar* to each other, and the third is *very
different*. That's the whole trick: **similar meaning → similar numbers.**

> The number of values in the vector is its **dimension**. Common sizes are 384,
> 768, and 1536. A bigger dimension just means a more detailed map. Whatever you
> choose, you must use the **same** model (and therefore the same dimension) for both
> storing and searching — like using the same ruler to measure everything.

---

## 6. Similarity: measuring "close in meaning"

Once text is numbers, "which pieces are relevant?" becomes a maths question: **which
vectors are closest to the question's vector?**

The measure RAGRAM Core uses is called **cosine distance**:

- Imagine each vector as an **arrow** pointing from the centre of the meaning map.
- Cosine distance looks at the **angle** between two arrows.
- **Same direction** (small angle) → distance near **0** → very similar meaning.
- **Opposite direction** → distance near **2** → opposite meaning.

So "distance" here is small when things are similar. That's why RAGRAM Core can say
"only keep chunks whose distance is below `0.5`" — it's really saying *"only keep
pieces that are genuinely close in meaning to the question."*

(You'll see this written as the operator `<=>` in the SQL. `<=>` *is* pgvector's
cosine-distance operator.)

---

## 7. Chunking: breaking big documents into pieces

You could turn an entire 50-page manual into one embedding — but that's a bad idea.
A single giant vector blurs everything together, and the model can only "see" so much
text at once.

Instead, we **chunk**: cut a document into smaller, overlapping pieces (say, a few
hundred characters each). Think of cutting a book into **index cards**. Each card is
small enough to be precise, and we embed each card separately.

Two details matter:

- **Smart cuts.** We try to split at natural boundaries — between paragraphs, then
  sentences, then words — rather than slicing mid-sentence.
- **Overlap.** Neighbouring chunks share a little text at the edges. This stops an
  idea that straddles a boundary from being cut in half and lost. Like giving each
  index card a sentence of context from the previous card.

RAGRAM Core does this with a "recursive character splitter" in `chunking.py`, and you
control the size and overlap with two settings.

---

## 8. The vector database, and pgvector

A **vector database** is simply a database that stores vectors and can answer the
question *"which stored vectors are closest to this one?"* very quickly.

Lots of specialised vector databases exist. RAGRAM Core takes a different, friendlier
approach: it uses **PostgreSQL**, a database you may already have, plus an extension
called **pgvector**.

- **PostgreSQL** ("Postgres") is one of the world's most popular, reliable databases.
- **pgvector** teaches Postgres a new trick: a column type called `vector`, and
  distance operators like `<=>` that let it compare vectors.

Why is this nice?

- You don't run yet another service — your vectors live right next to your normal
  data.
- You get Postgres's everyday strengths: backups, transactions, permissions, tooling.
- You can still do normal SQL things (filters, joins) alongside the vector search.

In RAGRAM Core, each chunk becomes one row in a table called `ragram_chunks`:

| column | what it holds |
| --- | --- |
| `chunk_id` | a stable id for the chunk |
| `text` | the actual text of the chunk |
| `embedding` | the vector (the meaning coordinates) |
| `meta` | extra info as JSON (source file, page, document id, ...) |
| `created_at` | when the row was written |

---

## 9. The index (HNSW): the library's shortcut

If you have a million chunks and you compare the question against *every single one*,
that's slow. Real databases use an **index** to find close matches without checking
everything.

RAGRAM Core creates an **HNSW** index. You don't need to understand the algorithm to
use it. The intuition:

- Comparing against every vector = reading every book in the library to find one fact.
- An HNSW index = the library's **card catalogue** that points you to the right shelf
  in a few steps.

It's an "approximate" method — once in a blue moon it might miss the single absolute
best match — but in exchange it's dramatically faster, which is what you want at
scale. (You'll see it in the SQL as `USING hnsw (embedding vector_cosine_ops)`.)

---

## 10. Putting it together: one question, step by step

Here is the full journey when you ask RAGRAM Core a question. This is the heart of
the whole project.

**Beforehand (ingestion — happens once per document):**

1. Your document is **chunked** into small, overlapping pieces.
2. Each chunk is turned into a **vector** by the embedding model.
3. Each chunk + its vector + its metadata is **stored** as a row in Postgres.

**At question time (query — happens on every question):**

1. Your question is turned into a **vector** using the *same* embedding model.
2. Postgres finds the **nearest chunk vectors** to your question vector (cosine
   distance, accelerated by the HNSW index).
3. Chunks that are **too far away** (above your `max_distance`) are thrown out — they
   aren't relevant enough to trust.
4. **If nothing is left**, RAGRAM Core stops here and replies *"I cannot find
   sufficient information in the current database context."* It does **not** ask the
   AI to guess. This is the single most important safety rule in the project.
5. **If relevant chunks remain**, they're pasted into a prompt along with a strict
   instruction: *"Answer only from this context, cite your sources, and if the answer
   isn't here, say so."*
6. The AI writes the answer, and RAGRAM Core returns it **together with the chunks it
   used** — so you can see exactly where the answer came from.

---

## 11. Grounding, hallucinations, and the "I don't know" rule

**Grounding** means tying the AI's answer to real, provided text instead of its
memory. A grounded answer is one you can verify.

**Hallucination** is when an AI confidently states something false. It's the single
biggest reason people don't trust chatbots for factual work.

RAGRAM Core fights hallucinations in **two independent ways**, so that even if one
fails, the other holds:

1. **The engine's guard (a rule in code).** Before the AI is ever called, irrelevant
   chunks are filtered out. If nothing relevant remains, the AI isn't called at all —
   the program just returns the "I cannot find sufficient information..." message
   itself. Deterministic, guaranteed.
2. **The system prompt (an instruction to the AI).** When relevant context *is*
   found, the AI is told, in strict terms, to use **only** that context, never to
   rely on outside knowledge, to cite chunk ids, and to reply with the exact same
   "I cannot find sufficient information..." sentence if the context doesn't cover
   the question.

Because both layers share the same refusal message, the user experience is consistent
no matter which one triggers.

---

## 12. Glossary

Keep this handy while reading the code.

| Term | In plain English |
| --- | --- |
| **RAG** | Retrieval-Augmented Generation — look up relevant text, then ask the AI to answer from it. |
| **LLM** | Large Language Model — the "AI" that writes the final answer (e.g. GPT). |
| **Embedding** | Turning text into numbers that represent its meaning. |
| **Embedding model** | The tool that creates embeddings (e.g. OpenAI, HuggingFace). |
| **Vector** | The list of numbers (coordinates) that an embedding produces. |
| **Dimension** | How many numbers are in a vector (e.g. 384 or 1536). |
| **Cosine distance** | A number from 0 to 2 measuring how different two vectors' meanings are; smaller = more similar. |
| **Chunk** | A small, overlapping piece of a document. |
| **Chunking** | Cutting documents into chunks. |
| **Vector database** | A database that stores vectors and finds the closest ones fast. |
| **pgvector** | A PostgreSQL extension that adds vectors and distance operators. |
| **Index (HNSW)** | A smart shortcut that makes nearest-vector search fast. |
| **Retrieval** | Finding the chunks most relevant to a question. |
| **Retrieve** | (A method on the engine) return the relevant chunks for a query. |
| **Context** | The retrieved chunks we hand to the AI. |
| **Grounding** | Forcing the answer to come from the provided context. |
| **Hallucination** | When an AI invents something that isn't true. |
| **Ingestion** | The process of loading your documents in (chunk → embed → store). |

---

## 13. Where to go next

- **Run it:** [GETTING_STARTED.md](GETTING_STARTED.md) walks you through setup and
  your first query.
- **See the whole project:** the [README](../README.md) covers architecture, usage,
  integrations, and the API reference.
- **Read the code:** start with `engine.py` (the coordinator), then
  `adapters/postgres_adapter.py` (the database part) and `prompts.py` (the grounding
  rules).

Welcome to RAG. It's a small idea with a big payoff.
