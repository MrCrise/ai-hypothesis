from uuid import UUID, uuid4

from common.db.models import PaperModel, TopicPaperModel
from common.db.session import get_db_session
from common.schemas.paper import AccessStatus, FunnelStats, PaperMetadata
from common.storage.s3 import s3_storage
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/topics/{topic_id}/corpus", tags=["corpus"])


@router.get("/funnel", response_model=FunnelStats)
async def get_funnel(
    topic_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> FunnelStats:
    total_res = await session.execute(
        select(func.count(TopicPaperModel.id)).where(TopicPaperModel.topic_id == topic_id)
    )
    after_dedup = total_res.scalar_one()

    relevant_res = await session.execute(
        select(func.count(TopicPaperModel.id)).where(
            TopicPaperModel.topic_id == topic_id,
            TopicPaperModel.is_relevant.is_(True),
        )
    )
    relevant = relevant_res.scalar_one()

    full_text_res = await session.execute(
        select(func.count(TopicPaperModel.id))
        .join(PaperModel, PaperModel.id == TopicPaperModel.paper_id)
        .where(
            TopicPaperModel.topic_id == topic_id,
            TopicPaperModel.is_relevant.is_(True),
            PaperModel.access_status.in_(
                [AccessStatus.FULL_TEXT.value, AccessStatus.INTERNAL.value]
            ),
        )
    )
    with_full_text = full_text_res.scalar_one()

    return FunnelStats(
        found_total=after_dedup,
        after_dedup=after_dedup,
        relevant=relevant,
        with_full_text=with_full_text,
    )


@router.post("/upload", response_model=PaperMetadata, status_code=status.HTTP_201_CREATED)
async def upload_internal_document(
    topic_id: UUID,
    title: str = Form(...),
    is_internal: bool = Form(True),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> PaperMetadata:
    paper_id = uuid4()
    content = await file.read()
    filename = file.filename or f"{paper_id}.pdf"
    s3_key = f"{topic_id}/{paper_id}/{filename}"

    await s3_storage.upload_bytes(
        key=s3_key,
        data=content,
        content_type=file.content_type or "application/pdf",
    )

    access_status = AccessStatus.INTERNAL if is_internal else AccessStatus.FULL_TEXT
    paper = PaperModel(
        id=paper_id,
        title=title,
        source="manual_upload",
        access_status=access_status.value,
        s3_key=s3_key,
    )
    topic_paper = TopicPaperModel(
        topic_id=topic_id,
        paper_id=paper_id,
        is_relevant=True,
        relevance_reason="Added manually by user",
        manual_override=True,
    )
    session.add_all([paper, topic_paper])
    await session.commit()

    return PaperMetadata(
        paper_id=paper.id,
        title=paper.title,
        source=paper.source,
        access_status=access_status,
        s3_key=paper.s3_key,
        is_relevant=True,
        relevance_reason="Added manually by user",
        manual_override=True,
    )
