import httpx
from common.config import settings


class BaseSearchConnector:
    def __init__(self) -> None:
        self.proxy = settings.http_proxy

    def get_http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(proxy=self.proxy, timeout=30.0)
