from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from common.db.session import close_db, init_db
from common.storage.s3 import s3_storage
from fastapi import FastAPI

from api.routers import chat, corpus, health, hypotheses, reports, topics


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    await s3_storage.ensure_bucket()
    yield
    await close_db()


app = FastAPI(
    title="Metallurgy R&D Hypothesis Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(topics.router, prefix="/api/v1")
app.include_router(corpus.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(hypotheses.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
