from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

from app.config import settings
from app.database import Base

# Import all ORM models so SQLAlchemy knows about them.
from app.models import chunk  # noqa: F401
from app.models import document  # noqa: F401
from app.models import ingestion_job  # noqa: F401
from app.models import knowledge_base  # noqa: F401


config = context.config

if config.config_file_name is not None:
    fileConfig(
        config.config_file_name
    )

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in offline mode.
    """

    url = settings.database_url

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named"
        },
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in online mode.
    """

    connectable = create_engine(
        settings.database_url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()