from common.config import Settings


def test_settings_inference_defaults() -> None:
    settings = Settings()
    assert settings.llm_base_url == "http://llm-server:8080/v1"
    assert settings.ml_inference_base_url == "http://ml-inference:8081"
    assert settings.ml_inference_device in ("cpu", "cuda")
    assert settings.embedding_dim == 1024
    assert settings.llm_gpu_layers >= 0
    assert settings.llm_parallel_slots >= 1
    assert settings.llm_bg_max_concurrency >= 1
    assert settings.ml_batch_max_size >= 1
    assert settings.ml_batch_max_wait_ms >= 1
