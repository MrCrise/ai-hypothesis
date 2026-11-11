from typing import Any

import aioboto3
from botocore.exceptions import ClientError

from common.config import settings
from common.schemas.health import ComponentHealth, ComponentStatus


class S3Storage:
    def __init__(self) -> None:
        self.session = aioboto3.Session()
        self.bucket = settings.s3_bucket_name

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "service_name": "s3",
            "endpoint_url": settings.s3_endpoint_url,
            "aws_access_key_id": settings.s3_access_key,
            "aws_secret_access_key": settings.s3_secret_key,
        }

    async def ensure_bucket(self) -> None:
        async with self.session.client(**self._client_kwargs()) as s3:
            try:
                await s3.head_bucket(Bucket=self.bucket)
            except ClientError:
                await s3.create_bucket(Bucket=self.bucket)

    async def check_health(self) -> ComponentHealth:
        try:
            async with self.session.client(**self._client_kwargs()) as s3:
                await s3.head_bucket(Bucket=self.bucket)
            return ComponentHealth(status=ComponentStatus.UP)
        except Exception as exc:
            return ComponentHealth(status=ComponentStatus.DOWN, detail=str(exc))

    async def upload_bytes(
        self, key: str, data: bytes, content_type: str = "application/pdf"
    ) -> str:
        async with self.session.client(**self._client_kwargs()) as s3:
            await s3.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
            )
        return key

    async def download_bytes(self, key: str) -> bytes:
        async with self.session.client(**self._client_kwargs()) as s3:
            response = await s3.get_object(Bucket=self.bucket, Key=key)
            return await response["Body"].read()


s3_storage = S3Storage()
