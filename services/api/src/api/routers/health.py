from common.db.session import check_db_health
from common.schemas.health import ComponentStatus, ReadinessResponse
from common.storage.s3 import s3_storage
from fastapi import APIRouter, Response, status

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(response: Response) -> ReadinessResponse:
    pg_health, vec_health = await check_db_health()
    s3_health = await s3_storage.check_health()

    all_up = all(
        c.status == ComponentStatus.UP for c in (pg_health, vec_health, s3_health)
    )
    overall = ComponentStatus.UP if all_up else ComponentStatus.DOWN

    if overall == ComponentStatus.DOWN:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall,
        postgres=pg_health,
        pgvector=vec_health,
        s3=s3_health,
    )
