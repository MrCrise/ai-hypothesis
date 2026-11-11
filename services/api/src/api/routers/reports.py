from uuid import UUID

from fastapi import APIRouter

router = APIRouter(prefix="/topics/{topic_id}/reports", tags=["reports"])


@router.get("/export")
async def export_report(topic_id: UUID, format: str = "docx") -> dict[str, str]:
    return {"topic_id": str(topic_id), "format": format, "status": "ready"}
