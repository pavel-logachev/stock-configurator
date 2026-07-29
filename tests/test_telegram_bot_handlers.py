from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.telegram_bot import handlers as telegram_handlers
from app.telegram_bot.handlers import (
    DISTRIBUTOR_CALLBACK_PREFIX,
    AccessController,
    distributor_callback,
    handle_text,
    result_command,
    start,
    v3_storage_command,
)


@pytest.fixture(autouse=True)
def fast_telegram_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telegram_handlers, "TELEGRAM_SEND_BASE_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(telegram_handlers, "TELEGRAM_SEND_MAX_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(telegram_handlers, "TELEGRAM_RETRY_AFTER_MAX_SECONDS", 0)


def test_handle_text_uses_v3_auto_without_excel() -> None:
    message = _FakeMessage("Need 2 servers")
    stock_api_client = _FakeStockApiClient()

    asyncio.run(handle_text(_update(message), _context(stock_api_client)))

    assert stock_api_client.v3_auto_texts == ["Need 2 servers"]
    assert stock_api_client.v3_auto_distributor_codes == ["ocs"]
    assert stock_api_client.created_texts == []
    assert stock_api_client.xlsx_report_ids == []
    assert stock_api_client.markdown_report_ids == []
    assert message.documents == []
    assert any("КП draft" in text for text in message.texts)


def test_start_shows_distributor_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_inline_keyboard(monkeypatch)
    message = _FakeMessage("/start")
    stock_api_client = _FakeStockApiClient()

    asyncio.run(start(_update(message), _context(stock_api_client)))

    assert message.texts
    assert "OCS" in message.texts[0]
    assert message.reply_markups[-1] is not None
    assert "Treolan" in repr(message.reply_markups[-1])


def test_handle_text_retries_reply_text_failure_then_succeeds() -> None:
    message = _FakeMessage("Need 2 servers", reply_text_failures=2)
    stock_api_client = _FakeStockApiClient()

    asyncio.run(handle_text(_update(message), _context(stock_api_client)))

    assert stock_api_client.v3_auto_texts == ["Need 2 servers"]
    assert stock_api_client.v3_auto_distributor_codes == ["ocs"]
    assert message.reply_text_attempts == len(message.texts) + 2
    assert message.texts[0].startswith(telegram_handlers.V3_ACCEPTED_TEXT)
    assert "OCS" in message.texts[0]
    assert any("КП draft" in text for text in message.texts)


def test_handle_text_does_not_send_excel_document() -> None:
    message = _FakeMessage("Need 2 servers", reply_document_failures=2)
    stock_api_client = _FakeStockApiClient()

    asyncio.run(handle_text(_update(message), _context(stock_api_client)))

    assert stock_api_client.v3_auto_texts == ["Need 2 servers"]
    assert stock_api_client.v3_auto_distributor_codes == ["ocs"]
    assert stock_api_client.xlsx_report_ids == []
    assert message.reply_document_attempts == 0
    assert message.documents == []


def test_handle_text_sends_v3_excel_when_result_is_persisted() -> None:
    summary = _v3_quote_summary("server")
    summary["match_run_id"] = 88
    summary["report_url"] = "/api/v1/match/88/report.md"
    summary["report_xlsx_url"] = "/api/v1/match/88/report.xlsx"
    message = _FakeMessage("Need 2 servers")
    stock_api_client = _FakeStockApiClient(v3_summary=summary)

    asyncio.run(handle_text(_update(message), _context(stock_api_client)))

    assert stock_api_client.v3_auto_texts == ["Need 2 servers"]
    assert stock_api_client.v3_auto_distributor_codes == ["ocs"]
    assert stock_api_client.xlsx_report_ids == [88]
    assert len(message.documents) == 1
    assert message.documents[0]["filename"] == "stock_match_88.xlsx"


def test_handle_text_uses_excel_caption_when_v3_card_delivery_fails() -> None:
    summary = {
        "profile": "custom",
        "category_ids": ["V1100"],
        "distributor_code": "ocs",
        "result_state": "no_recommendation",
        "pipeline_version": "v3_full_category_matrix",
        "match_run_id": 173,
        "engineering_review_required": False,
        "diagnostics": {"matrix_row_count": 21},
        "validated_quote": {},
        "no_recommendation_reason": {
            "summary": "Selected matrix does not contain required server components.",
            "failed_requirements": [
                "Нужны HDD, SSD и 2 x 10GbE SFP+, но в выбранной матрице только готовые серверы."
            ],
            "recommended_next_actions": [
                "Расширить матрицу до серверных компонентов или запросить BTO у дистрибьютора."
            ],
        },
        "report_url": "/api/v1/match/173/report.md",
        "report_xlsx_url": "/api/v1/match/173/report.xlsx",
    }
    message = _FakeMessage("Need 2 servers", reply_text_failures=6)
    stock_api_client = _FakeStockApiClient(v3_summary=summary)

    asyncio.run(handle_text(_update(message), _context(stock_api_client)))

    assert stock_api_client.xlsx_report_ids == [173]
    assert len(message.documents) == 1
    assert message.documents[0]["filename"] == "stock_match_173.xlsx"
    assert "КП не сформировано" in message.documents[0]["caption"]
    assert "Не закрыто:" in message.documents[0]["caption"]


def test_handle_text_sends_long_v3_result_as_compact_message() -> None:
    message = _FakeMessage("Need 2 servers")
    stock_api_client = _FakeStockApiClient(v3_summary=_long_v3_quote_summary())

    asyncio.run(handle_text(_update(message), _context(stock_api_client)))

    assert stock_api_client.v3_auto_texts == ["Need 2 servers"]
    assert stock_api_client.v3_auto_distributor_codes == ["ocs"]
    assert message.documents == []
    assert any("КП draft" in text for text in message.texts)
    assert any("Полная детализация отправлена в Excel-файле." in text for text in message.texts)


def test_handle_text_result_delivery_failure_is_controlled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    message = _FakeMessage("Need 2 servers", reply_text_failures=6)
    stock_api_client = _FakeStockApiClient()

    caplog.set_level(logging.INFO, logger="app.telegram_bot.handlers")
    asyncio.run(handle_text(_update(message), _context(stock_api_client)))

    assert stock_api_client.v3_auto_texts == ["Need 2 servers"]
    assert stock_api_client.v3_auto_distributor_codes == ["ocs"]
    assert message.documents == []
    assert "Telegram v3 auto result delivery failed after retries" in caplog.text
    assert "secret" not in caplog.text


def test_result_command_resends_summary_and_excel_without_creating_match() -> None:
    message = _FakeMessage("/result 70")
    stock_api_client = _FakeStockApiClient(result_match_run_id=70)

    asyncio.run(result_command(_update(message), _context(stock_api_client, args=["70"])))

    assert stock_api_client.created_texts == []
    assert stock_api_client.v3_auto_texts == []
    assert stock_api_client.match_summary_ids == [70]
    assert stock_api_client.xlsx_report_ids == [70]
    assert len(message.documents) == 1
    assert message.documents[0]["filename"] == "stock_match_70.xlsx"


def test_result_command_resends_v3_summary_and_excel() -> None:
    summary = _v3_quote_summary("server")
    summary["match_run_id"] = 88
    summary["pipeline_version"] = "v3_full_category_matrix"
    summary["v3_result_state"] = summary["result_state"]
    message = _FakeMessage("/result 88")
    stock_api_client = _FakeStockApiClient(result_match_run_id=88, v3_summary=summary)

    asyncio.run(result_command(_update(message), _context(stock_api_client, args=["88"])))

    assert stock_api_client.match_summary_ids == [88]
    assert stock_api_client.xlsx_report_ids == [88]
    assert any("КП draft" in text for text in message.texts)
    assert len(message.documents) == 1
    assert message.documents[0]["filename"] == "stock_match_88.xlsx"


def test_result_command_rejects_invalid_argument() -> None:
    message = _FakeMessage("/result abc")
    stock_api_client = _FakeStockApiClient()

    asyncio.run(result_command(_update(message), _context(stock_api_client, args=["abc"])))

    assert stock_api_client.created_texts == []
    assert stock_api_client.v3_auto_texts == []
    assert stock_api_client.match_summary_ids == []
    assert stock_api_client.xlsx_report_ids == []
    assert message.documents == []
    assert any("/result 70" in text for text in message.texts)


def test_v3_storage_command_calls_full_category_endpoint_without_excel() -> None:
    message = _FakeMessage("/v3_storage NAS 40TB")
    stock_api_client = _FakeStockApiClient()

    asyncio.run(
        v3_storage_command(
            _update(message),
            _context(stock_api_client, args=["NAS", "40TB"]),
        )
    )

    assert stock_api_client.simple_requests == [("storage", "NAS 40TB")]
    assert stock_api_client.simple_request_distributor_codes == ["ocs"]
    assert stock_api_client.created_texts == []
    assert stock_api_client.v3_requests == []
    assert stock_api_client.v3_auto_texts == []
    assert stock_api_client.xlsx_report_ids == []
    assert message.documents == []
    assert any("КП draft" in text for text in message.texts)


def test_distributor_callback_selects_treolan_for_next_text_request() -> None:
    message = _FakeMessage("select")
    query = _FakeCallbackQuery(
        data=f"{DISTRIBUTOR_CALLBACK_PREFIX}treolan",
        message=message,
    )
    stock_api_client = _FakeStockApiClient()
    context = _context(stock_api_client)

    asyncio.run(distributor_callback(_callback_update(query, message), context))

    assert context.user_data["stock_distributor_code"] == "treolan"
    assert query.answers == ["Выбран склад: Treolan. Следующий запрос пойдет в этот склад."]
    assert query.edits
    assert "Treolan" in query.edits[-1]["text"]

    message = _FakeMessage("Need 2 servers")
    asyncio.run(handle_text(_update(message), context))

    assert stock_api_client.v3_auto_texts == ["Need 2 servers"]
    assert stock_api_client.v3_auto_distributor_codes == ["treolan"]
    assert "Treolan" in message.texts[0]


class _FakeMessage:
    def __init__(
        self,
        text: str,
        *,
        reply_text_failures: int = 0,
        reply_document_failures: int = 0,
    ) -> None:
        self.text = text
        self.texts: list[str] = []
        self.reply_markups: list[Any | None] = []
        self.documents: list[dict[str, Any]] = []
        self.reply_text_failures = reply_text_failures
        self.reply_document_failures = reply_document_failures
        self.reply_text_attempts = 0
        self.reply_document_attempts = 0

    async def reply_text(self, text: str, **kwargs: Any) -> None:
        self.reply_text_attempts += 1
        if self.reply_text_failures > 0:
            self.reply_text_failures -= 1
            raise _transient_telegram_error()
        self.texts.append(text)
        self.reply_markups.append(kwargs.get("reply_markup"))

    async def reply_document(self, *, document: Any, filename: str, caption: str) -> None:
        self.reply_document_attempts += 1
        if self.reply_document_failures > 0:
            self.reply_document_failures -= 1
            raise _transient_telegram_error()
        self.documents.append(
            {
                "content": document.getvalue(),
                "filename": filename,
                "caption": caption,
            }
        )


class _FakeStockApiClient:
    def __init__(
        self,
        *,
        result_match_run_id: int = 42,
        v3_summary: dict[str, Any] | None = None,
    ) -> None:
        self.created_texts: list[str] = []
        self.simple_requests: list[tuple[str | None, str]] = []
        self.simple_request_distributor_codes: list[str | None] = []
        self.v3_auto_texts: list[str] = []
        self.v3_auto_distributor_codes: list[str | None] = []
        self.v3_requests: list[tuple[str | None, str]] = []
        self.v3_request_distributor_codes: list[str | None] = []
        self.match_summary_ids: list[int] = []
        self.xlsx_report_ids: list[int] = []
        self.markdown_report_ids: list[int] = []
        self.result_match_run_id = result_match_run_id
        self.v3_summary = v3_summary

    async def create_match(self, user_text: str) -> dict[str, Any]:
        self.created_texts.append(user_text)
        return _match_summary(42)

    async def get_match_summary(self, match_run_id: int) -> dict[str, Any]:
        self.match_summary_ids.append(match_run_id)
        if self.v3_summary is not None:
            summary = dict(self.v3_summary)
            summary.setdefault("match_run_id", match_run_id)
            return summary
        return _match_summary(self.result_match_run_id)

    async def create_v3_full_category_quote(
        self,
        *,
        profile: str | None = None,
        distributor_code: str | None = None,
        user_text: str,
    ) -> dict[str, Any]:
        self.v3_requests.append((profile, user_text))
        self.v3_request_distributor_codes.append(distributor_code)
        return self.v3_summary or _v3_quote_summary(profile or "server")

    async def create_simple_stock_quote(
        self,
        *,
        profile: str | None = None,
        distributor_code: str | None = None,
        user_text: str,
    ) -> dict[str, Any]:
        self.simple_requests.append((profile, user_text))
        self.simple_request_distributor_codes.append(distributor_code)
        return self.v3_summary or _v3_quote_summary(profile or "server")

    async def create_v3_full_category_quote_auto(
        self,
        user_text: str,
        *,
        distributor_code: str | None = None,
    ) -> dict[str, Any]:
        self.v3_auto_texts.append(user_text)
        self.v3_auto_distributor_codes.append(distributor_code)
        return self.v3_summary or _v3_quote_summary("server")

    async def get_match_report_xlsx(self, match_run_id: int) -> bytes:
        self.xlsx_report_ids.append(match_run_id)
        return b"xlsx-bytes"

    async def get_match_report_markdown(self, match_run_id: int) -> str:
        self.markdown_report_ids.append(match_run_id)
        return "# Report"


def _match_summary(match_run_id: int) -> dict[str, Any]:
    return {
        "match_run_id": match_run_id,
        "status": "partial_stock_matched",
        "engineer_review_required": True,
        "total_candidates": 2,
        "matched_items": 0,
        "risk_flags": [],
        "missing_requirements": [],
        "candidates": [
            {
                "producer": "NERPA",
                "part_number": "D5720-181125SA04",
                "item_id": "1000841882",
                "available_quantity": 3,
                "price_value": "6900",
                "price_currency": "USD",
            }
        ],
        "report_url": f"/api/v1/match/{match_run_id}/report.md",
    }


def _v3_quote_summary(profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "category_ids": ["V2101"],
        "distributor_code": "ocs",
        "result_state": "quote_draft_review_required",
        "engineering_review_required": True,
        "diagnostics": {
            "matrix_row_count": 1,
            "model": "qwen/qwen3.7-plus",
        },
        "validated_quote": {
            "total_price_value": "100.0000",
            "total_price_currency": "USD",
            "lines": [
                {
                    "role": "primary",
                    "producer": "ASUS",
                    "part_number": "PN-1",
                    "item_name": "NAS",
                    "quantity": 1,
                    "unit_price_value": "100.0000",
                    "unit_price_currency": "USD",
                    "line_total_value": "100.0000",
                    "line_total_currency": "USD",
                }
            ],
            "why_selected": "Lowest technically acceptable option.",
            "engineering_review_required": True,
        },
    }


def _long_v3_quote_summary() -> dict[str, Any]:
    summary = _v3_quote_summary("server")
    quote = summary["validated_quote"]
    quote["why_selected"] = "Long commercial explanation. " * 150
    quote["compatibility_check"] = {
        "status": "compatible",
        "checked_facts": ["Selected matrix rows are accounted for. " * 120],
        "blocking_mismatches": [],
        "unresolved_risks": ["Needs engineering review. " * 40],
    }
    base_line = dict(quote["lines"][0])
    quote["lines"] = [
        {
            **base_line,
            "part_number": f"LONG-{index:03d}",
            "stock_row_id": f"ocs:p1:{index}",
            "reason": "Long line reason. " * 120,
        }
        for index in range(80)
    ]
    return summary


def _context(stock_api_client: _FakeStockApiClient, *, args: list[str] | None = None) -> Any:
    return SimpleNamespace(
        args=args or [],
        user_data={},
        application=SimpleNamespace(
            bot_data={
                "access_controller": AccessController(
                    allowed_user_ids=frozenset(),
                    allow_all=True,
                ),
                "stock_api_client": stock_api_client,
            }
        ),
    )


def _update(message: _FakeMessage) -> Any:
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=100),
        effective_message=message,
    )


class _FakeCallbackQuery:
    def __init__(self, *, data: str, message: _FakeMessage) -> None:
        self.data = data
        self.message = message
        self.answers: list[str] = []
        self.edits: list[dict[str, Any]] = []

    async def answer(self, *, text: str) -> None:
        self.answers.append(text)

    async def edit_message_text(self, *, text: str, reply_markup: Any | None = None) -> None:
        self.edits.append({"text": text, "reply_markup": reply_markup})


def _callback_update(query: _FakeCallbackQuery, message: _FakeMessage) -> Any:
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=100),
        effective_message=message,
        callback_query=query,
    )


def _install_fake_inline_keyboard(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeInlineKeyboardButton:
        def __init__(self, text: str, callback_data: str) -> None:
            self.text = text
            self.callback_data = callback_data

        def __repr__(self) -> str:
            return f"FakeInlineKeyboardButton({self.text!r}, {self.callback_data!r})"

    class FakeInlineKeyboardMarkup:
        def __init__(self, inline_keyboard: list[list[FakeInlineKeyboardButton]]) -> None:
            self.inline_keyboard = inline_keyboard

        def __repr__(self) -> str:
            return f"FakeInlineKeyboardMarkup({self.inline_keyboard!r})"

    monkeypatch.setattr(telegram_handlers, "InlineKeyboardButton", FakeInlineKeyboardButton)
    monkeypatch.setattr(telegram_handlers, "InlineKeyboardMarkup", FakeInlineKeyboardMarkup)


def _transient_telegram_error() -> Exception:
    return httpx.ConnectError(
        "temporary proxy failure via socks5://user:secret@proxy "
        "for chat 123456789"
    )
