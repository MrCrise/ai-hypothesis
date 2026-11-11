from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from common.config import settings
from common.db.models import Base
from common.schemas.health import ComponentHealth, ComponentStatus

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    await engine.dispose()


async def check_db_health() -> tuple[ComponentHealth, ComponentHealth]:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            ext_res = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            has_vector = ext_res.scalar_one_or_none() == "vector"

        pg_health = ComponentHealth(status=ComponentStatus.UP)
        vec_health = (
            ComponentHealth(status=ComponentStatus.UP)
            if has_vector
            else ComponentHealth(status=ComponentStatus.DOWN, detail="pgvector extension missing")
        )
        return pg_health, vec_health
    except Exception as exc:
        err = ComponentHealth(status=ComponentStatus.DOWN, detail=str(exc))
        return err, err


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
