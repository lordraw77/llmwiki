# Wiki-LLM — Docker Guide

This document explains how to build, run, and publish Wiki-LLM as a container.
Wiki-LLM is a multi-format knowledge base that exposes a **REST API** and an
**MCP server** over two transports (**stdio** and **HTTP/SSE**) from a single
process.

---

## Overview

| Artifact | Purpose |
| --- | --- |
| [`Dockerfile`](Dockerfile) | Multi-stage image (Python 3.12 slim), runs as a non-root user, ships a `/health` HEALTHCHECK. |
| [`.dockerignore`](.dockerignore) | Keeps the build context small (excludes `.env`, `data/`, venv, tests, VCS). |
| [`docker-compose.yml`](docker-compose.yml) | One-command stack: the app plus an optional Qdrant server behind a profile. |
| [`Makefile`](Makefile) | Shortcuts for build / run / compose / publish (`make help`). |
| [`scripts/publish.sh`](scripts/publish.sh) | Multi-arch build & push to Docker Hub via `docker buildx`. |

The image runs `python run_http.py` by default, serving:

- REST API on `http://<host>:8000`
- MCP HTTP/SSE at `http://<host>:8000/mcp/sse`

Persistent vector-store data lives on the `/data` volume.

---

## Embeddings: image size matters

The default embedding provider in `.env.example` is `local`
(`sentence-transformers`), which pulls in **torch** and produces a multi-GB
image. The Docker image therefore **omits** local embeddings by default and is
intended to be used with a **remote** `EMBEDDING_PROVIDER`
(`mistral`, `openrouter`, `groq`, `gemini`, `cloudflare`, `cerebras`, `nvidia`,
`puter`, or any `openai_compatible` endpoint).

Two ways to proceed:

1. **Remote embeddings (recommended for containers).** In your `.env` set, e.g.:
   ```dotenv
   EMBEDDING_PROVIDER=mistral
   EMBEDDING_MODEL=mistral-embed
   EMBEDDING_API_KEY=your-key
   ```
2. **Bake in local embeddings.** Build with the optional extra:
   ```bash
   docker build --build-arg WITH_LOCAL_EMBEDDINGS=true -t wikillm:local .
   # or: make build WITH_LOCAL_EMBEDDINGS=true
   ```

---

## Quick start

### 1. Configure

```bash
cp .env.example .env
# edit .env — at minimum pick an embedding provider (see above)
```

> Note: the compose file forces `DATA_DIR`, `CHROMA_PATH`, and `QDRANT_PATH`
> onto the `/data` volume regardless of `.env`, so storage always persists.

### 2. Run with Docker Compose (easiest)

```bash
docker compose up -d --build        # or: make up
docker compose logs -f wikillm      # or: make logs
```

The server is now on http://localhost:8000 — try http://localhost:8000/health
and the interactive docs at http://localhost:8000/docs.

To include a standalone Qdrant server (then set `VECTOR_BACKEND=qdrant` and
`QDRANT_URL=http://qdrant:6333` in `.env`):

```bash
docker compose --profile qdrant up -d --build   # or: make up-qdrant
```

### 3. Or run the image directly

```bash
make build                                       # builds wikillm/wikillm:<version> and :latest
docker run --rm -it \
  -p 8000:8000 \
  --env-file .env \
  -v wikillm-data:/data \
  wikillm/wikillm:latest                          # or: make run
```

---

## Using the MCP stdio transport from the container

The default command runs the HTTP server. For an MCP **stdio** client (e.g.
Claude Desktop) that launches the server as a subprocess, override the command:

```bash
docker run --rm -i \
  --env-file .env \
  -v wikillm-data:/data \
  wikillm/wikillm:latest python run_stdio.py
```

Example `claude_desktop_config.json` entry:

```json
{
  "mcpServers": {
    "wikillm": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "/absolute/path/to/.env",
        "-v", "wikillm-data:/data",
        "wikillm/wikillm:latest",
        "python", "run_stdio.py"
      ]
    }
  }
}
```

`stdout` is reserved for JSON-RPC framing (logs go to `stderr`), so the stdio
transport works cleanly through the container.

---

## Verify a running container

With the server up, run the integration test scripts from the host:

```bash
python3.12 test_http.py --base-url http://localhost:8000
# add --api-key <key> if API_KEY is set in .env
```

---

## Publishing to Docker Hub

`scripts/publish.sh` builds a multi-arch image (`linux/amd64,linux/arm64`) and
pushes it, tagging both the version (from `pyproject.toml`) and `latest`.

```bash
# Interactive login first (or set DOCKERHUB_TOKEN for non-interactive login):
docker login

# Publish under your Docker Hub namespace:
make publish DOCKER_USER=youruser
# equivalently:
DOCKER_USER=youruser ./scripts/publish.sh
```

Useful overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOCKER_USER` | `wikillm` | Docker Hub namespace (user or org). |
| `IMAGE_NAME` | `wikillm` | Repository name. |
| `VERSION` | from `pyproject.toml` | Image version tag. |
| `PLATFORMS` | `linux/amd64,linux/arm64` | Target architectures. |
| `WITH_LOCAL_EMBEDDINGS` | `false` | Bake in sentence-transformers. |
| `PUSH_LATEST` | `true` | Also push the `:latest` tag. |
| `DOCKERHUB_TOKEN` | — | If set, performs a non-interactive `docker login`. |

Example — single arch, pinned version, no `latest`:

```bash
DOCKER_USER=youruser VERSION=0.1.0 PLATFORMS=linux/amd64 PUSH_LATEST=false \
  ./scripts/publish.sh
```

---

## Configuration reference

All runtime configuration is via environment variables (see
[`.env.example`](.env.example) for the full list). The image sets container-
friendly defaults:

| Variable | Image default | Notes |
| --- | --- | --- |
| `APP_HOST` | `0.0.0.0` | Bind address inside the container. |
| `APP_PORT` | `8000` | Exposed port. |
| `DATA_DIR` | `/data` | Persisted via the `/data` volume. |
| `CHROMA_PATH` | `/data/chroma` | ChromaDB storage. |
| `QDRANT_PATH` | `/data/qdrant` | Qdrant embedded storage. |
| `API_KEY` | _(unset)_ | Auth disabled unless you set it. |

---

## Troubleshooting

- **Image is huge / slow to build:** you built with `WITH_LOCAL_EMBEDDINGS=true`.
  Use a remote embedding provider and rebuild without the flag.
- **`/health` is failing / container unhealthy:** check `docker compose logs -f
  wikillm`. A misconfigured embedding provider (missing API key) surfaces at
  startup because the knowledge base is initialised eagerly.
- **Switched embedding provider and search misbehaves:** changing the provider
  usually changes the vector **dimension**, which requires a fresh collection.
  Remove the volume (`docker volume rm wikillm-data`) or change `COLLECTION_NAME`.
- **401 responses:** `API_KEY` is set; send `Authorization: Bearer <key>` or
  `X-API-Key: <key>`.
