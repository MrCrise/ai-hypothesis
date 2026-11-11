from uuid import UUID

from common.db.models import TopicModel
from common.db.session import get_db_session
from common.schemas.topic import (
    SearchPlan,
    TaskStage,
    TopicCreateRequest,
    TopicStatusResponse,
)
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/topics", tags=["topics"])


@router.post("", response_model=TopicStatusResponse, status_code=status.HTTP_201_CREATED)
async def create_topic(
    req: TopicCreateRequest,
    session: AsyncSession = Depends(get_db_session),
) -> TopicStatusResponse:
    topic = TopicModel(query=req.query, stage=TaskStage.PLANNING.value, search_plan={})
    session.add(topic)
    await session.commit()
    await session.refresh(topic)
    return TopicStatusResponse(
        topic_id=topic.id,
        query=topic.query,
        stage=TaskStage(topic.stage),
        created_at=topic.created_at,
    )


@router.get("", response_model=list[TopicStatusResponse])
async def list_topics(
    session: AsyncSession = Depends(get_db_session),
) -> list[TopicStatusResponse]:
    result = await session.execute(select(TopicModel).order_by(TopicModel.created_at.desc()))
    topics = result.scalars().all()
    return [
        TopicStatusResponse(
            topic_id=t.id,
            query=t.query,
            stage=TaskStage(t.stage),
            created_at=t.created_at,
        )
        for t in topics
    ]


@router.get("/{topic_id}", response_model=TopicStatusResponse)
async def get_topic(
    topic_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> TopicStatusResponse:
    topic = await session.get(TopicModel, topic_id)
    if not topic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found")
    return TopicStatusResponse(
        topic_id=topic.id,
        query=topic.query,
        stage=TaskStage(topic.stage),
        created_at=topic.created_at,
    )


@router.put("/{topic_id}/plan", response_model=SearchPlan)
async def update_search_plan(
    topic_id: UUID,
    plan: SearchPlan,
    session: AsyncSession = Depends(get_db_session),
) -> SearchPlan:
    topic = await session.get(TopicModel, topic_id)
    if not topic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found")
    topic.search_plan = plan.model_dump(mode="json")
    if plan.approved:
        topic.stage = TaskStage.SEARCHING.value
    await session.commit()
    return plan
