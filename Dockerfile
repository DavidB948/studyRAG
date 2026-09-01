FROM python:3.12-slim

# uv is the package manager; --frozen installs exactly what uv.lock pins.
COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY studyrag/ studyrag/
COPY scripts/ scripts/
# --no-dev: the image serves the API. ragas and pytest are dev dependencies,
# so evals and tests run on the host, not in here.
RUN uv sync --frozen --no-dev

# Models (~900 MB) download on first use and are cached here. Mounted as a volume in
# compose so a container restart does not re-download them.
ENV HF_HOME=/models
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "studyrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
