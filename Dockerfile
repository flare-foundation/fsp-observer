FROM python:3.14-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./

ARG DEV=false
# avoid symlinks that break when copying the venv between stages
ENV UV_LINK_MODE=copy
# install outside /app so a local volume mount can't shadow the venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN if [ "$DEV" = "true" ]; then \
      uv sync --locked; \
    else \
      uv sync --locked --no-dev; \
    fi

FROM python:3.14-slim-trixie AS final

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . /app

# venv binaries first so `python` resolves to the synced environment
ENV PATH="/opt/venv/bin:${PATH}"

CMD ["python", "main.py"]
