from __future__ import annotations

import app.cli.extract_stock_spec as extract_stock_spec_cli
from app.core.config import get_llm_settings

SERVER_REQUEST = "Нужно 2 сервера 2U, 2 процессора, 512 ГБ RAM, SSD, 2 БП, склад Москва"


def test_extract_stock_spec_cli_smoke_with_disabled_llm(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    get_llm_settings.cache_clear()

    exit_code = extract_stock_spec_cli.run(["--text", SERVER_REQUEST])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "confirmation_text:" in captured.out
    assert "spec_json:" in captured.out
    assert '"quantity": 2' in captured.out
    assert '"form_factor": "2U"' in captured.out
    assert '"shipment_city": "Москва"' in captured.out
    assert captured.err == ""

    get_llm_settings.cache_clear()


def test_extract_stock_spec_cli_reads_request_from_file(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    get_llm_settings.cache_clear()

    exit_code = extract_stock_spec_cli.run(["--file", "data/examples/request_server_basic.txt"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"quantity": 2' in captured.out
    assert '"storage": {' in captured.out
    assert captured.err == ""

    get_llm_settings.cache_clear()
