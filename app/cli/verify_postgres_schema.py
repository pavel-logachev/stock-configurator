from __future__ import annotations

import asyncio

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from app.core.database import Base, get_engine
from app.db import models  # noqa: F401


async def verify_postgres_schema() -> None:
    engine = get_engine()
    try:
        if engine.dialect.name != "postgresql":
            raise RuntimeError("PostgreSQL is required for schema verification.")

        expected_head = ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
        if not expected_head:
            raise RuntimeError("Alembic does not have a single current head.")

        async with engine.connect() as connection:
            database_head = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            table_names = await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )

        if database_head != expected_head:
            raise RuntimeError(
                "Database revision "
                f"{database_head!r} does not match Alembic head {expected_head!r}."
            )

        expected_tables = set(Base.metadata.tables)
        missing_tables = sorted(expected_tables - table_names)
        if missing_tables:
            raise RuntimeError(f"Database is missing mapped tables: {', '.join(missing_tables)}")
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(verify_postgres_schema())


if __name__ == "__main__":
    main()
