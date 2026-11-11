from enum import StrEnum

from pydantic import BaseModel


class ComponentStatus(StrEnum):
    UP = "up"
    DOWN = "down"


class ComponentHealth(BaseModel):
    status: ComponentStatus
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: ComponentStatus
    postgres: ComponentHealth
    pgvector: ComponentHealth
    s3: ComponentHealth
