# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

ARG PYTHON_VERSION=3.12.6
FROM python:${PYTHON_VERSION}-slim AS lightning-base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# This is necessary to keep .venv in the container after build
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Prevent uv from downloading isolated Python builds as Python is already available
ENV UV_PYTHON_DOWNLOADS=false

# While doing runs in docker, we want to avoid having automatic sync if packages are missing
ENV UV_NO_SYNC=false

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
# NB: the home directory must be writable (and is also used as uv's cache): with the usual
# `--home "/nonexistent"`, every `uv run` inside the container died with
# "Failed to initialize cache at `/nonexistent/.cache/uv`: Permission denied".
ARG UID=10001
RUN adduser \
  --disabled-password \
  --gecos "" \
  --home "/app" \
  --shell "/sbin/nologin" \
  --no-create-home \
  --uid "${UID}" \
  appuser

# the app must be able to write its outputs (logs, checkpoints, downloaded data) in here
RUN chown appuser:appuser /app

# uv needs a writable cache even with `--no-sync`
ENV UV_CACHE_DIR=/app/.cache/uv

# Which torch build to install: `cpu` (default) or `gpu`
ARG TORCH_EXTRA=cpu

# Install the dependencies first: this layer only depends on pyproject.toml/uv.lock, so it stays
# cached when the source code changes. Without `--extra`, no torch is installed at all and the
# training/eval/serve scripts cannot even start.
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked --no-install-project --extra ${TORCH_EXTRA}

# Copy the source code into the container and install the project itself (so that `import src...`
# resolves inside the image).
COPY --chown=appuser:appuser . .
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked --extra ${TORCH_EXTRA}

# Switch to the non-privileged user to run the application.
USER appuser

# Serve the model on port 8000 (requires a trained checkpoint, see `just train`)
CMD ["uv", "run", "--no-sync", "src/serve.py"]
