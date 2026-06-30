# Wiki-LLM

A production-shaped **Wiki-LLM**: a multi-format document knowledge base that
exposes the **same** in-process index through two interfaces — a **REST API**
and a **Model Context Protocol (MCP)** server available over both **stdio** and
**HTTP/SSE** transports (protocol revision **`2024-11-05`**).

* **Ingestion** of `.txt`, `.md`, `.pdf`, `.docx` (and best-effort `.doc`) via
  REST upload or directly from the filesystem.
* **Semantic search** over a pluggable vector store (**ChromaDB** or **Qdrant**)
  with pluggable embeddings (**local** `sentence-transformers` or **remote**
  OpenAI-compatible providers), including **round-robin rotation with automatic
  failover** across several providers/keys.
* **Optional API-key authentication** (disabled by default) protecting both the
  REST endpoints and the MCP HTTP endpoint.

---

## 1. Why these libraries

| Concern | Choice | Rationale |
|---|---|---|
| Web framework & SSE | **FastAPI + Uvicorn** | Async-native, Pydantic-validated, first-class multipart uploads, and — decisively — lets the MCP SSE app be **mounted in the same process** as the REST API, so one server serves both. |
| PDF parsing | **pypdf** | Pure-Python, no system dependencies, robust per-page extraction. |
| Markdown | **markdown-it-py** | Tokenises Markdown so markup is stripped to clean searchable text. |
| Text encoding | **charset-normalizer** | Detects/repairs wrong encodings so malformed `.txt` never crashes ingestion. |
| DOCX | **python-docx** | Extracts both paragraph and table text. |
| Vector store | **ChromaDB** (default) + **Qdrant** | Chroma is zero-config and embedded (agile default); Qdrant scales to a server. Both sit behind one `VectorStore` interface. |
| Embeddings | **sentence-transformers** (local) + **httpx** (remote OpenAI-compatible) | Local = offline/no-key; remote = lightweight install and access to hosted models. A rotation wrapper adds load-balancing + failover. |
| Config | **pydantic-settings + python-dotenv** | Typed, validated configuration from `.env`. |
| MCP | official **`mcp`** SDK | Native stdio + SSE transports, conformant to `2024-11-05`. |

---

## 2. Project layout

```
.
├── .env.example                 # copy to .env
├── requirements.txt
├── pyproject.toml
├── run_stdio.py                 # entry point: MCP over stdio
├── run_http.py                  # entry point: REST + MCP/SSE over HTTP
├── Dockerfile                   # multi-stage image (see DOCKER.md)
├── docker-compose.yml           # app + optional Qdrant service
├── Makefile                     # build / run / compose / publish shortcuts
├── DOCKER.md                    # Docker guide (build, run, publish)
├── scripts/publish.sh           # multi-arch build & push to Docker Hub
├── data/                        # persistent vector store (created at runtime)
└── wikillm/
    ├── config.py                # typed settings from .env
    ├── logging_setup.py         # logs to stderr/file (keeps stdout clean)
    ├── models.py                # pydantic models / API schema
    ├── auth.py                  # optional Bearer / X-API-Key auth
    ├── knowledge_base.py        # orchestrator + process-wide singleton
    ├── rest_api.py              # FastAPI app + MCP SSE mount
    ├── mcp_server.py            # MCP server, tools, stdio + SSE transports
    ├── ingestion/
    │   ├── parsers.py           # format detection + per-format extraction
    │   └── chunking.py          # overlapping, paragraph-aware chunking
    ├── embeddings/
    │   ├── base.py              # EmbeddingProvider ABC
    │   ├── local.py             # sentence-transformers
    │   ├── remote.py            # OpenAI-compatible HTTP providers
    │   ├── rotating.py          # round-robin + failover
    │   └── factory.py           # build_embedding_provider()
    └── store/
        ├── base.py              # VectorStore ABC
        ├── chroma_store.py      # ChromaDB backend (+ sqlite shim)
        ├── qdrant_store.py      # Qdrant backend
        └── factory.py           # build_vector_store()
```

---

## 3. Requirements & setup

**Python 3.10+ is required** (the `mcp` SDK does not support 3.9). On hosts whose
default `python3` is older, use an explicit interpreter such as `python3.12`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # includes sentence-transformers (heavy)

# If you only use REMOTE embeddings you can skip the heavy local model:
#   pip install -r requirements.txt --no-deps  # then install the non-torch deps
# or simply leave sentence-transformers uninstalled and set a remote provider.

cp .env.example .env                   # then edit as needed
```

> **ChromaDB & sqlite:** Chroma needs `sqlite3 >= 3.35`. On older Linux the
> bundled `pysqlite3-binary` is installed and transparently swapped in by
> `wikillm/store/chroma_store.py` — no action required.

---

## 4. Configuration cheat-sheet

All variables are documented in [`.env.example`](.env.example). The most useful:

| Variable | Default | Meaning |
|---|---|---|
| `API_KEY` | *(unset)* | When set, auth is required (Bearer or `X-API-Key`). Unset = open. |
| `VECTOR_BACKEND` | `chroma` | `chroma` or `qdrant`. |
| `EMBEDDING_PROVIDER` | `local` | `local` or a remote provider id. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name. |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` | *(unset)* | Credentials/endpoint for remote providers. |
| `EMBEDDING_ROTATION` | *(unset)* | JSON list of `{provider,model,api_key,base_url}` to enable rotation. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking window. |
| `MCP_SSE_PATH` | `/mcp` | Mount prefix for the MCP SSE transport. |

**Supported remote providers** (OpenAI-compatible `/embeddings`): `openrouter`,
`groq`, `gemini`, `cloudflare`, `cerebras`, `mistral`, `nvidia`, `puter`,
`openai_compatible`. `cloudflare`/`puter` require an explicit `EMBEDDING_BASE_URL`.

> **Changing the embedding provider changes the vector dimension.** Use a fresh
> `COLLECTION_NAME` (or wipe `data/`) when switching, and ensure all rotation
> members produce the **same** dimension (validated at startup).

---

## 5. Run the HTTP server (REST + MCP/SSE)

```bash
python3.12 run_http.py
# serving on http://0.0.0.0:8000  (interactive docs at /docs)
```

### REST examples

```bash
# Health + stats
curl -s localhost:8000/health

# Upload & ingest a file (type auto-detected)
curl -s -F "file=@./mydoc.pdf" localhost:8000/documents/upload

# Ingest a directory from the server filesystem
curl -s -X POST localhost:8000/documents/ingest-path \
     -H 'Content-Type: application/json' \
     -d '{"path": "./docs", "recursive": true}'

# Add a free-text note
curl -s -X POST localhost:8000/notes \
     -H 'Content-Type: application/json' \
     -d '{"title": "Onboarding", "content": "The wifi password is on the fridge."}'

# Search
curl -s "localhost:8000/search?q=wifi%20password&k=5"

# Manage documents
curl -s localhost:8000/documents
curl -s localhost:8000/documents/<DOCUMENT_ID>
curl -s -X DELETE localhost:8000/documents/<DOCUMENT_ID>
```

### With authentication enabled

Set `API_KEY=secret` in `.env`, then:

```bash
curl -s localhost:8000/documents                       # -> 401
curl -s -H "Authorization: Bearer secret" localhost:8000/documents   # -> 200
curl -s -H "X-API-Key: secret" localhost:8000/documents              # -> 200
```

### MCP over HTTP/SSE

With the HTTP server running, the MCP transport is available at:

* `GET  http://localhost:8000/mcp/sse` — open the SSE event stream.
* `POST http://localhost:8000/mcp/messages/` — client JSON-RPC messages.

Point any MCP HTTP/SSE client at `http://localhost:8000/mcp/sse`. When `API_KEY`
is set, send it via the `Authorization`/`X-API-Key` header.

---

## 6. Run with Docker

A multi-stage [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml),
[`Makefile`](Makefile) and a Docker Hub [`scripts/publish.sh`](scripts/publish.sh)
are included. Full details — image-size notes, stdio-in-container, Claude Desktop
config, publishing — are in **[DOCKER.md](DOCKER.md)**.

```bash
cp .env.example .env                 # pick a remote EMBEDDING_PROVIDER (see note below)
docker compose up -d --build         # or: make up
curl -s localhost:8000/health
```

> **Image size:** the default image **omits** local `sentence-transformers`
> (torch) to stay slim (~600 MB) and is meant for a **remote** embedding
> provider. To bake in local embeddings, build with
> `--build-arg WITH_LOCAL_EMBEDDINGS=true` (or `make build WITH_LOCAL_EMBEDDINGS=true`).

Run the image directly, or use the MCP **stdio** transport from the container:

```bash
make build && make run                                   # HTTP server
docker run --rm -i --env-file .env -v wikillm-data:/data \
  lordraw/llmwiki:latest python run_stdio.py             # MCP stdio
```

Publish a multi-arch image to Docker Hub (`lordraw/llmwiki`, tag from the latest
git tag):

```bash
docker login
git tag v0.1.0        # the Docker tag is derived from this (v stripped)
make publish
```

---

## 7. Run the MCP server over stdio (e.g. Claude Desktop)

```bash
python3.12 run_stdio.py     # speaks JSON-RPC on stdin/stdout; logs go to stderr
```

Add to Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "wikillm": {
      "command": "python3.12",
      "args": ["run_stdio.py"],
      "cwd": "/opt/llmwiki",
      "env": {
        "EMBEDDING_PROVIDER": "local",
        "VECTOR_BACKEND": "chroma"
      }
    }
  }
}
```

> Use absolute paths for `command`/`cwd` if Claude Desktop cannot resolve them
> (e.g. the venv interpreter at `/opt/llmwiki/.venv/bin/python`).

### Tools exposed

| Tool | Purpose |
|---|---|
| `search_wiki(query, k=5)` | Semantic search over the wiki. |
| `add_note(title, content)` | Add and index a free-text note. |
| `ingest_path(path, recursive=true)` | Ingest a file/directory from the server filesystem. |
| `list_documents()` | List indexed documents with chunk counts. |
| `get_document(document_id)` | Retrieve a document's reconstructed text + metadata. |
| `delete_document(document_id)` | Remove a document and its chunks. |

---

## 8. Embedding rotation (load-balancing + failover)

Spread embedding calls across several providers/keys. On a rate limit / timeout /
5xx, the failing member is put on a short cooldown and the call transparently
fails over to the next one.

```dotenv
EMBEDDING_ROTATION_STRATEGY=round_robin
EMBEDDING_ROTATION_COOLDOWN=30
EMBEDDING_ROTATION=[
  {"provider":"groq","model":"text-embedding-3-small","api_key":"key1"},
  {"provider":"mistral","model":"mistral-embed","api_key":"key2"},
  {"provider":"openrouter","model":"openai/text-embedding-3-small","api_key":"key3"}
]
```

A terser alternative uses parallel comma-separated lists
(`EMBEDDING_PROVIDERS`, `EMBEDDING_API_KEYS`, `EMBEDDING_MODELS`,
`EMBEDDING_BASE_URLS`) — see [`.env.example`](.env.example).

---

## 9. Logging & stdout hygiene

The MCP stdio transport uses `stdout` exclusively for JSON-RPC framing. All logs
therefore go to **stderr** (and optionally `LOG_FILE`); `stdout` is never used
for logging. This is what keeps Claude Desktop's connection from breaking.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `mcp` won't install | You're on Python < 3.10. Use `python3.12`. |
| `unsupported version of sqlite3` (Chroma) | Ensure `pysqlite3-binary` is installed (it ships in `requirements.txt`). |
| Tool calls return `{"error": ...}` about embeddings | `local` provider needs `sentence-transformers`; install it or switch to a remote provider. |
| Search returns nothing after switching providers | Vector dimension changed — use a new `COLLECTION_NAME` or wipe `data/`. |
| Claude Desktop shows no tools | Check the server runs standalone (`python3.12 run_stdio.py`) and that `cwd`/`command` are correct; inspect stderr logs. |
| Docker image too large / build slow | You built with `WITH_LOCAL_EMBEDDINGS=true`. Use a remote provider and rebuild without it. See [DOCKER.md](DOCKER.md). |
