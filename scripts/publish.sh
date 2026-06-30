#!/usr/bin/env bash
# =============================================================================
# Wiki-LLM — Docker Hub publish script
# -----------------------------------------------------------------------------
# Builds a multi-architecture image with `docker buildx` and pushes it to
# Docker Hub, tagging both the explicit version and `latest`.
#
# Environment variables (all optional, with defaults):
#   DOCKER_USER             Docker Hub namespace (user/org).   Default: lordraw
#   IMAGE_NAME              Repository name.                    Default: llmwiki
#   VERSION                 Version tag. Default: latest git tag (v0.1.0 ->
#                           0.1.0), falling back to pyproject.toml.
#   PLATFORMS               Build platforms.   Default: linux/amd64,linux/arm64
#   WITH_LOCAL_EMBEDDINGS   "true" to bake in sentence-transformers. Default: false
#   PUSH_LATEST             "true" to also tag/push :latest.    Default: true
#   DOCKERHUB_TOKEN         If set (with DOCKER_USER), a non-interactive
#                           `docker login` is performed using this token.
#
# Usage:
#   ./scripts/publish.sh
#   DOCKER_USER=lordraw VERSION=0.2.0 PLATFORMS=linux/amd64 ./scripts/publish.sh
#
# Prerequisites:
#   * Docker with the buildx plugin.
#   * You are logged in to Docker Hub (`docker login`) OR DOCKERHUB_TOKEN is set.
# =============================================================================
set -euo pipefail

# --- Resolve configuration ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DOCKER_USER="${DOCKER_USER:-lordraw}"
IMAGE_NAME="${IMAGE_NAME:-llmwiki}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
WITH_LOCAL_EMBEDDINGS="${WITH_LOCAL_EMBEDDINGS:-false}"
PUSH_LATEST="${PUSH_LATEST:-true}"

if [[ -z "${VERSION:-}" ]]; then
    # Prefer the latest git tag (v0.1.0 -> 0.1.0); fall back to pyproject.toml.
    VERSION="$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')"
fi
if [[ -z "${VERSION}" ]]; then
    VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
fi
if [[ -z "${VERSION}" ]]; then
    echo "ERROR: could not determine VERSION (set VERSION=... explicitly)." >&2
    exit 1
fi

IMAGE="${DOCKER_USER}/${IMAGE_NAME}"

echo "=============================================================="
echo " Publishing ${IMAGE}"
echo "   version           : ${VERSION}"
echo "   platforms         : ${PLATFORMS}"
echo "   local embeddings  : ${WITH_LOCAL_EMBEDDINGS}"
echo "   also tag :latest  : ${PUSH_LATEST}"
echo "=============================================================="

# --- Sanity checks -----------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not on PATH." >&2
    exit 1
fi
if ! docker buildx version >/dev/null 2>&1; then
    echo "ERROR: 'docker buildx' is required for multi-arch builds." >&2
    exit 1
fi

# --- Optional non-interactive login -----------------------------------------
if [[ -n "${DOCKERHUB_TOKEN:-}" ]]; then
    echo "Logging in to Docker Hub as '${DOCKER_USER}'..."
    echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKER_USER}" --password-stdin
fi

# --- Ensure a buildx builder exists ------------------------------------------
BUILDER_NAME="wikillm-builder"
if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
    echo "Creating buildx builder '${BUILDER_NAME}'..."
    docker buildx create --name "${BUILDER_NAME}" --use
else
    docker buildx use "${BUILDER_NAME}"
fi
docker buildx inspect --bootstrap >/dev/null

# --- Build & push ------------------------------------------------------------
TAG_ARGS=(--tag "${IMAGE}:${VERSION}")
if [[ "${PUSH_LATEST}" == "true" ]]; then
    TAG_ARGS+=(--tag "${IMAGE}:latest")
fi

docker buildx build \
    --platform "${PLATFORMS}" \
    --build-arg "WITH_LOCAL_EMBEDDINGS=${WITH_LOCAL_EMBEDDINGS}" \
    "${TAG_ARGS[@]}" \
    --push \
    .

echo
echo "Done. Pushed:"
echo "  ${IMAGE}:${VERSION}"
[[ "${PUSH_LATEST}" == "true" ]] && echo "  ${IMAGE}:latest"
echo
echo "Pull with:  docker pull ${IMAGE}:${VERSION}"
