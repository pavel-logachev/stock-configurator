# Local development

## Requirements

- Python 3.12 or newer;
- Docker with Compose v2 for PostgreSQL and container acceptance;
- Git.

Distributor, Telegram, and LLM credentials are not required for tests.

## Python environment

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If your shell exports a global `PYTHONPATH`, unset it before running project commands. A global package path can bypass the virtual environment and load incompatible binary wheels.

```bash
unset PYTHONPATH
```

## Tests and lint

```bash
ruff check .
pytest -q
```

Tests use synthetic fixtures and patched network clients. A change that needs a live account is not complete until the same behavior is also covered by a deterministic test.

## Local containers

```bash
cp .env.example .env
docker compose up --build -d stock-api
docker compose exec stock-api alembic upgrade head
curl http://127.0.0.1:8010/health
```

Stop the stack without deleting the database volume:

```bash
docker compose down
```

Deleting the named volume is destructive and is not part of normal cleanup.

## Optional integrations

Put credentials only in `.env`. Keep each integration disabled until its endpoint, account permissions, timeout, and rate limits have been reviewed.

The Telegram service starts only with the `telegram` profile:

```bash
docker compose --profile telegram up --build -d
```

## Database migrations

Create and review migrations with Alembic. Never rely on application startup to mutate the schema implicitly.

```bash
alembic upgrade head
alembic current
```

## Evaluation data

Keep real bundles under `evaluation/simple_stock/local/`. Keep generated reports under `evaluation/simple_stock/reports/`. Both paths are ignored. Do not use production requests, prices, or identifiers in tests or pull requests.
