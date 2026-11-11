from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


async def index_paper_chunks(session: AsyncSession, paper_id: UUID, parsed_doc: dict) -> int:
    return 0
