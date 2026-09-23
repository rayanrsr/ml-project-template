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

# torch.compile (inductor) compiles generated kernels at runtime, which needs a C++ toolchain.
# Without it the default config fails with
# "InvalidCxxCompiler: No working C++ compiler found in torch._inductor.config.cpp.cxx: (None, 'g++')".
RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc g++ \
  && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
  --disabled-password \
  --gecos "" \
  --home "/app" \
  --shell "/sbin/nologin" \
  --no-create-home \
  --uid "${UID}" \
  appuser

# Everything under /app belongs to the app user: it must be able to write its outputs (logs,
# checkpoints, downloaded data) and uv's cache, and the whole install is done *as* that user so
# that no root-owned file ends up in the image ("failed to open file
# `/app/.cache/uv/sdists-v9/.git`: Permission denied").
RUN chown appuser:appuser /app
ENV UV_CACHE_DIR=/app/.cache/uv
USER appuser

# Which torch build to install: `cpu` (default) or `gpu`
ARG TORCH_EXTRA=cpu

# Install the dependencies first: this layer only depends on pyproject.toml/uv.lock, so it stays
# cached when the source code changes. Without `--extra`, no torch is installed at all and the
# training/eval/serve scripts cannot even start.
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
# NB: no `--mount=type=cache` here: BuildKit creates cache mounts owned by root, and uv runs as
# `appuser` (add `,uid=10001,gid=10001` to the mount options if you want one back). Layer caching
# already avoids re-installing the dependencies unless pyproject.toml/uv.lock change.
RUN uv sync --locked --no-install-project --extra ${TORCH_EXTRA}

# Copy the source code into the container and install the project itself (so that `import src...`
# resolves inside the image).
COPY --chown=appuser:appuser . .
RUN uv sync --locked --extra ${TORCH_EXTRA}

# Serve the model on port 8000 (requires a trained checkpoint, see `just train`)
CMD ["uv", "run", "--no-sync", "src/serve.py"]
