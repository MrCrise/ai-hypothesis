from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://rd_user:rd_secret@postgres:5432/metallurgy_rag"
    s3_endpoint_url: str = "http://seaweedfs:8333"
    s3_access_key: str = "any"
    s3_secret_key: str = "any"
    s3_bucket_name: str = "papers"

    llm_base_url: str = "http://llm-server:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model_name: str = "default-llm"
    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    pdf_parser_url: str = "http://pdf-parser:5001"
    http_proxy: str | None = None


settings = Settings()
