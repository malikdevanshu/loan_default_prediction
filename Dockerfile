FROM python:3.10-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir uv


ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app


COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project


COPY config ./config
COPY loan_check ./loan_check

ENTRYPOINT ["python", "-m"]
CMD ["loan_check.train.training"]