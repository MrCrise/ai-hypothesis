from common.config import Settings


def test_parser_defaults() -> None:
    s = Settings()
    assert s.pdf_parser_url == "http://pdf-parser:5001"
    assert s.parser_task_timeout > s.parser_http_timeout > 0
    assert s.parser_min_chars_per_page == 50
    assert s.chunk_max_chars == 1500
