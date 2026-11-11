from uuid import UUID

from common.schemas.hypothesis import HypothesisSchema
from fastapi import APIRouter

router = APIRouter(prefix="/topics/{topic_id}/hypotheses", tags=["hypotheses"])


@router.get("", response_model=list[HypothesisSchema])
async def list_hypotheses(topic_id: UUID) -> list[HypothesisSchema]:
    return []
