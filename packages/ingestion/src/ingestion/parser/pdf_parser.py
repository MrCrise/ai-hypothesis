import httpx
from common.config import settings


async def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.pdf_parser_url}/v1alpha/convert/file",
            files={"files": (filename, pdf_bytes, "application/pdf")},
        )
        response.raise_for_status()
        return response.json()
