# Research Publications Agent

A locally deployed AI assistant that answers questions about a catalogue of ~2,400 research
publications and about documents you upload. It uses **native tool calling** (the model asks for
typed, Pydantic-validated functions) inside an **agent loop** (the model decides each next step
from the previous results) and cites every claim back to the paper or page it came from.

- **Publications catalogue:** counts, trends, authors, topics, journals and full records, from SQLite and Qdrant.
- **Your documents:** PDF (with a text layer), DOCX, TXT and Markdown, searched passage by passage with page numbers.
- **General questions:** answered directly, without tools.
- **Human approval:** the one write action (changing a paper's cluster label) waits for you to click Approve.

Stack: FastAPI · Pydantic · SQLAlchemy/SQLite · Qdrant · OpenAI SDK through the Portkey gateway ·
React 19 · Vite · Tailwind · Docker Compose.

---

## Quick start

Requirements: Docker Desktop. Everything else runs in containers.

```bash
cp .env.example .env                 # then set PORTKEY_API_KEY in .env
mkdir -p data/seed && cp /path/to/papers_with_cluster_labels.csv data/seed/
make up                              # build and start; waits until healthy
make seed                            # load the CSV and build the vector index (one-off, a few minutes)
open http://localhost:8080
```

`make help` lists every command. Data lives in Docker volumes, so `make down` / `make up` keeps it.
`make clean` deletes it, and asks first.

### Portkey configuration

The app talks to Portkey exactly like the Open WebUI Portkey connection does:

| Setting | Purpose |
|---|---|
| `PORTKEY_BASE_URL` | Gateway URL, default `https://ai-gateway.apps.cloud.rt.nyu.edu/v1` |
| `PORTKEY_API_KEY` | Sent as `x-portkey-api-key`. No `Authorization: Bearer` header is sent. |
| `PORTKEY_VIRTUAL_KEY` | Optional `x-portkey-virtual-key`. Leave blank when models are named `@provider/model`. |
| `EMBEDDING_PORTKEY_API_KEY` / `EMBEDDING_PORTKEY_VIRTUAL_KEY` | Optional separate credentials for embeddings. |
| `CHAT_MODEL` / `EMBEDDING_MODEL` | Routed names of the form `@provider/model`, default `@gpt-4o-mini/gpt-4o-mini` / `@openai-embedding/text-embedding-3-small`. `GET <PORTKEY_BASE_URL>/models` lists what your key can use. |

Every agent run is tagged with `x-portkey-trace-id: <run id>`, so it can be found in the Portkey dashboard.
Check the gateway with `make test-live`: streaming, tool calls and embeddings, using your key.

---

## How it works

```
Browser ─► web (nginx :8080, 127.0.0.1 only) ─┬─ /        React app
                                              └─ /api/*  ─► api (FastAPI)
                                                              ├─ SQLite  (publications, documents, conversations, runs)
                                                              ├─ Qdrant  (publication + document-passage vectors)
                                                              └─ Portkey gateway → LLM
```

### Tool calling (`backend/app/tools/`)

Each tool is an async Python function with a Pydantic **Args** model and a Pydantic **Result** model.
The registry turns the Args models into the OpenAI `tools` JSON schema. When the model calls a tool:

1. The raw JSON arguments are validated (types, ranges, unknown fields, malformed JSON).
2. Invalid arguments go back to the model as a structured error, so it can correct itself.
3. The function runs with a timeout.
4. The result is validated against the Result model, capped at 12,000 characters, and returned as a `tool` message.

| Tool | Data | Use |
|---|---|---|
| `search_publications` | Qdrant | Topic / concept search with year, author, cluster and journal filters |
| `filter_publications` | SQLite | Exact lists, sorting, paging, totals (parameterised SQL) |
| `publication_stats` | SQLite | Counts by year, cluster, journal, author, keyword or type |
| `get_publication` | SQLite | One full record |
| `resolve_author` | SQLite | "Rajesh Ranganath" → stored form "Ranganath R." |
| `run_readonly_sql` | SQLite | Fallback: one SELECT on the publication tables (read-only, authorizer-guarded, 3 s limit) |
| `list_documents` / `search_documents` | SQLite + Qdrant | Uploaded documents, with page-level passages |
| `update_cluster_label` | SQLite + Qdrant | **Requires approval**; audited in `cluster_label_changes` |

### The agent loop (`backend/app/agent/loop.py`)

The model decides every step:
- answer directly, or call tools (several at once run in parallel)
- read the results, then retry, chain, or stop

The code around the model enforces the guardrails:
- at most `AGENT_MAX_STEPS` steps, then one final call with tools disabled
- more than three invalid tool calls end the run
- write tools pause the run for approval; the run resumes after your decision
- a new message cancels a pending change

Citations are verified: an answer may cite `[pub:<id>]` or `[doc:<id>:<chunk>]` only if a tool returned that id
in the conversation. Anything else is removed before you see it. Every message, run and tool call is stored,
and the browser receives the run live over Server-Sent Events. Closing the tab does not stop a run.

### Repository layout

```
backend/app/
  agent/      loop, SSE events, prompt, citation checks, run streaming
  tools/      tool registry + publication, SQL and document tools
  llm/        Portkey chat client (openai SDK) and embeddings client
  rag/        parsing, chunking, ingestion, Qdrant access
  db/         SQLAlchemy models, persistence helpers (Alembic migrations in backend/alembic)
  api/        HTTP routes, schemas, errors
  seed/       CSV → SQLite + vectors
backend/tests/  unit, agent-loop, API and live tests
frontend/src/   pages, components, hooks, API client, SSE parser
scripts/        smoke.py (end-to-end eval), node-docker.sh
```

---

## Testing

| Command | What it runs |
|---|---|
| `make test` | All offline tests: ~280 backend tests (coverage ≥ 85%) and ~60 frontend tests. No network; a scripted fake LLM and in-memory Qdrant. |
| `make lint` | Ruff and the TypeScript type check |
| `make test-live` | Real gateway: streaming, streamed tool calls, `tool_choice=none`, usage, embeddings |
| `make smoke` | Against the running stack: health and readiness, document upload, then 12 golden questions graded on tools used, facts and citations (target ≥ 10/12). Expected facts are computed from the CSV. |

The agent-loop tests (`backend/tests/agent/test_loop.py`) cover:
- plain answers; single, parallel and multi-step tool calls
- invalid and non-JSON arguments, unknown tools, timeouts, the step limit
- approve, reject and supersede
- LLM failures, empty completions, unverified citations, history handling

---

## Development without Docker for the API

```bash
make dev-qdrant          # terminal 1: Qdrant on localhost:6333
make dev-api             # terminal 2: API on :8000 with reload; uncomment the dev settings in .env
make dev-web             # terminal 3: Vite on :5173 (runs in the Node 22 container)
```

Vite 8, Vitest 5 and jsdom need **Node ≥ 22.22**. The `make` targets run the frontend toolchain in the official
`node:22` image, so your local Node version doesn't matter. To run `npm` directly, upgrade Node first.

---

## Operations

- **Health:** `GET /api/health` (liveness). `GET /api/ready` names each failing dependency: database,
  migrations, Qdrant, embedding model, seeded data.
- **Logs:** `make logs`. JSON lines with `request_id` and `run_id`.
- **Changing the embedding model:** set `EMBEDDING_MODEL`, then run `make restart reindex`.
  Readiness reports the mismatch until you do.
- **Backups:** the `app-data` volume (SQLite + uploads) is the source of truth; Qdrant can be rebuilt with `make reindex`.
  ```bash
  docker run --rm -v publications-agent_app-data:/data -v "$PWD":/backup alpine tar czf /backup/app-data.tgz -C /data .
  ```
- **Security:**
  - The UI is published on 127.0.0.1 only, and there is no login (single local user).
  - Containers run as non-root with read-only root filesystems.
  - nginx sends a strict Content-Security-Policy.
  - Secrets live only in `.env`, which is gitignored.

## Troubleshooting

| Symptom | Fix |
|---|---|
| API exits with `Configuration error: PORTKEY_API_KEY` | Set the key in `.env`, then `make up`. |
| Answers fail with "rejected the credentials" | Wrong key, or the gateway expects `PORTKEY_VIRTUAL_KEY`. |
| `x-portkey-provider header is required` | Use a routed model name (`@provider/model`) for `CHAT_MODEL` / `EMBEDDING_MODEL`. |
| `llm_bad_request` mentioning `stream_options` | Set `LLM_STREAM_USAGE=false`. |
| `llm_bad_request` mentioning `temperature` (reasoning models) | Leave `LLM_TEMPERATURE=` blank. |
| The model never calls tools | Use a model that supports function calling (`make test-live` checks this). |
| A PDF ends "No extractable text" | It is scanned or image-only; there is no OCR. Export it with a text layer. |
| `/api/ready` says publications are not indexed | Run `make seed`. It is resumable. |
