from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Storage & Database
    database_url: str = "postgresql+asyncpg://rd_user:rd_secret@postgres:5432/metallurgy_rag"
    s3_endpoint_url: str = "http://seaweedfs:8333"
    s3_access_key: str = "any"
    s3_secret_key: str = "any"
    s3_bucket_name: str = "papers"

    # Local LLM Server (llama.cpp)
    llm_base_url: str = "http://llm-server:8080/v1"
    llm_api_key: str = "EMPTY"
    llm_model_name: str = "Qwen3.8-27B"
    llm_model_file: str = "Qwen3.8-27B-UD-Q4_K_M.gguf"
    llama_cpp_image_tag: str = "server-cuda"
    llm_ctx_size: int = 32768
    llm_gpu_layers: int = 28
    llm_parallel_slots: int = 4
    llm_bg_max_concurrency: int = 3
    llm_request_timeout_seconds: float = 300.0

    # ML Inference Service (Qwen3-Embedding & Qwen3-Reranker)
    models_dir: str = "./models"
    ml_inference_base_url: str = "http://ml-inference:8081"
    ml_inference_device: Literal["cpu", "cuda"] = "cpu"
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    ml_batch_max_size: int = 32
    ml_batch_max_wait_ms: int = 10
    ml_inference_timeout_seconds: float = 60.0
    torch_num_threads: int = 4

    # External & Ingestion
    pdf_parser_url: str = "http://pdf-parser:5001"
    http_proxy: str | None = None


settings = Settings()
