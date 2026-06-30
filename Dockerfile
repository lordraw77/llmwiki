# syntax=docker/dockerfile:1.7
# =============================================================================
# Wiki-LLM container image
# -----------------------------------------------------------------------------
# Multi-stage build:
#   * "builder" installs the Python dependencies into an isolated virtualenv.
#   * the final stage copies that venv into a slim runtime image and runs as a
#     non-root user.
#
# Python 3.12 is used because the `mcp` package requires Python 3.10+.
#
# Build-time options (ARG):
#   WITH_LOCAL_EMBEDDINGS  "true" to also install sentence-transformers (pulls
#                          in torch — large image). Default "false": the image
#                          is meant to be used with a remote EMBEDDING_PROVIDER.
#
# Examples:
#   docker build -t lordraw/llmwiki:latest .
#   docker build --build-arg WITH_LOCAL_EMBEDDINGS=true -t lordraw/llmwiki:local .
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ARG WITH_LOCAL_EMBEDDINGS=false

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv

# Build tools needed by some wheels (kept out of the final image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create the virtualenv that will be copied verbatim into the runtime stage.
RUN python -m venv "$VENV_PATH"
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /build

# Install runtime dependencies first to maximise Docker layer caching: this
# layer is only rebuilt when requirements.txt changes.
#
# sentence-transformers (and its torch dependency) is heavy. It is stripped from
# the install set unless WITH_LOCAL_EMBEDDINGS=true, keeping the default image
# slim for remote-embedding deployments. It is lazily imported at runtime, so
# the application still starts without it.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && if [ "$WITH_LOCAL_EMBEDDINGS" = "true" ]; then \
           pip install -r requirements.txt; \
       else \
           grep -ivE '^[[:space:]]*sentence-transformers' requirements.txt > requirements.runtime.txt \
           && pip install -r requirements.runtime.txt; \
       fi

# -----------------------------------------------------------------------------
# Stage 2 — runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Wiki-LLM" \
      org.opencontainers.image.description="Multi-format knowledge base with REST and MCP (stdio + HTTP/SSE) interfaces." \
      org.opencontainers.image.source="https://github.com/lordraw77/llmwiki" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    DATA_DIR=/data \
    CHROMA_PATH=/data/chroma \
    QDRANT_PATH=/data/qdrant \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

# curl is used by the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 wikillm

# Bring in the prebuilt virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application code (see .dockerignore for what is excluded).
COPY wikillm/ ./wikillm/
COPY run_http.py run_stdio.py ./
COPY README.md DOCKER.md ./

# Persistent storage lives on a volume; make it writable by the runtime user.
RUN mkdir -p /data && chown -R wikillm:wikillm /data /app

USER wikillm

VOLUME ["/data"]
EXPOSE 8000

# Liveness probe against the REST health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${APP_PORT}/health" || exit 1

# Default: run the HTTP server (REST + MCP HTTP/SSE).
# For the MCP stdio transport, override with: docker run ... python run_stdio.py
CMD ["python", "run_http.py"]
