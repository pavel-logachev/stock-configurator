from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
from typing import Any

import httpx

from app.telegram_bot.formatting import (
    choose_excel_report_delivery,
    choose_v3_full_category_quote_delivery,
    format_match_summary,
    format_v3_excel_caption,
)
from app.telegram_bot.stock_api_client import StockApiClient, StockApiClientError

try:  # pragma: no cover - exercised with real python-telegram-bot in production.
    from telegram.error import NetworkError, RetryAfter, TimedOut
except ImportError:  # pragma: no cover - keeps unit-test fakes lightweight.
    NetworkError = None  # type: ignore[assignment]
    RetryAfter = None  # type: ignore[assignment]
    TimedOut = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised with real python-telegram-bot in production.
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
except ImportError:  # pragma: no cover - keeps unit-test fakes lightweight.
    InlineKeyboardButton = None  # type: ignore[assignment]
    InlineKeyboardMarkup = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

TELEGRAM_SEND_MAX_ATTEMPTS = 3
TELEGRAM_SEND_BASE_BACKOFF_SECONDS = 0.5
TELEGRAM_SEND_MAX_BACKOFF_SECONDS = 3.0
TELEGRAM_RETRY_AFTER_MAX_SECONDS = 10.0
DISTRIBUTOR_USER_DATA_KEY = "stock_distributor_code"
DISTRIBUTOR_CALLBACK_PREFIX = "stock_distributor:"
DEFAULT_DISTRIBUTOR_CODE = "ocs"
SUPPORTED_DISTRIBUTORS = {
    "ocs": "OCS",
    "treolan": "Treolan",
}

START_TEXT = (
    "Stock Configurator готов. Отправьте текст заявки, и я запущу подбор "
    "по актуальной складской матрице."
)
HELP_TEXT = (
    "Отправьте обычным сообщением требования к серверу, СХД/NAS или сетевому "
    "оборудованию. Я определю товарную группу, проверю склад и верну черновик КП "
    "или понятный ответ, почему предложение не сформировано."
)
STATUS_TEXT = "Бот работает. Обычные сообщения отправляются в универсальный подбор по складу."
ACCESS_DENIED_TEXT = "Нет доступа."
ACCEPTED_TEXT = "Принял запрос, подбираю по складу..."
MATCH_ERROR_TEXT = "Не удалось выполнить подбор. Попробуйте позже."
REPORT_ERROR_TEXT = (
    "Краткий результат готов, но подробный Excel-отчет пока не удалось получить."
)
REPORT_SENT_TEXT = "Подробный отчет отправлен Excel-файлом."
RESULT_USAGE_TEXT = "Укажите номер подбора. Пример: /result 70."
RESULT_LOAD_ERROR_TEXT = (
    "Не удалось найти или загрузить результат. Проверьте match_id и попробуйте позже."
)
RESULT_CREATED_HINT_TEMPLATE = (
    "Результат уже создан. Его можно переотправить командой /result {match_run_id}."
)
HANDLER_ERROR_TEXT = "Произошла ошибка обработки запроса. Попробуйте позже."
V3_ACCEPTED_TEXT = (
    "Принял запрос. Определяю товарную группу, обновляю склад и готовлю подбор..."
)
V3_ERROR_TEXT = "Не удалось выполнить подбор. Попробуйте позже."
V3_USAGE_TEXT = (
    "Напишите запрос после команды. Обычный текст без команды идет "
    "в универсальный подбор по складу."
)
DISTRIBUTOR_SELECT_TEXT = (
    "Выберите склад для следующих запросов. Сейчас выбран: {distributor_label}."
)
DISTRIBUTOR_SELECTED_TEXT = (
    "Выбран склад: {distributor_label}. Следующий запрос пойдет в этот склад."
)
DISTRIBUTOR_UNKNOWN_TEXT = "Неизвестный склад. Выберите один из доступных вариантов."

_HTTPX_TRANSIENT_SEND_ERRORS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.TimeoutException,
    httpx.TransportError,
)
_TELEGRAM_TRANSIENT_SEND_ERRORS = tuple(
    error_type
    for error_type in (NetworkError, TimedOut, RetryAfter)
    if isinstance(error_type, type)
)
_TRANSIENT_SEND_ERRORS = (*_HTTPX_TRANSIENT_SEND_ERRORS, *_TELEGRAM_TRANSIENT_SEND_ERRORS)


class TelegramDeliveryError(RuntimeError):
    """Raised when a user-facing Telegram send operation failed after retries."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"Telegram delivery failed during {operation}.")
        self.operation = operation


@dataclass(frozen=True)
class AccessController:
    allowed_user_ids: frozenset[int]
    allow_all: bool = False

    @classmethod
    def from_config(cls, raw_allowed_user_ids: str, *, environment: str) -> AccessController:
        allowed_user_ids = parse_allowed_user_ids(raw_allowed_user_ids)
        if allowed_user_ids:
            return cls(allowed_user_ids=allowed_user_ids)

        if environment.lower() == "dev":
            logger.warning("TELEGRAM_ALLOWED_USER_IDS is empty; allowing all users in dev mode.")
            return cls(allowed_user_ids=frozenset(), allow_all=True)

        logger.warning("TELEGRAM_ALLOWED_USER_IDS is empty; denying all users outside dev mode.")
        return cls(allowed_user_ids=frozenset(), allow_all=False)

    def is_allowed(self, user_id: int | None) -> bool:
        if self.allow_all:
            return True
        return user_id is not None and user_id in self.allowed_user_ids


def parse_allowed_user_ids(raw_allowed_user_ids: str) -> frozenset[int]:
    raw_value = raw_allowed_user_ids.strip()
    if not raw_value:
        return frozenset()

    user_ids: set[int] = set()
    for part in re.split(r"[\s,;]+", raw_value):
        if not part:
            continue
        try:
            user_ids.add(int(part))
        except ValueError as exc:
            raise ValueError("TELEGRAM_ALLOWED_USER_IDS must contain integer user ids.") from exc
    return frozenset(user_ids)


async def start(update: Any, context: Any) -> None:
    if not await _ensure_authorized(update, context):
        return
    await _reply_text(
        update,
        START_TEXT + "\n\n" + _distributor_selection_text(context),
        reply_markup=_distributor_keyboard(_selected_distributor_code(context)),
    )


async def help_command(update: Any, context: Any) -> None:
    if not await _ensure_authorized(update, context):
        return
    await _reply_text(
        update,
        HELP_TEXT + "\n\n" + _distributor_selection_text(context),
        reply_markup=_distributor_keyboard(_selected_distributor_code(context)),
    )


async def status(update: Any, context: Any) -> None:
    if not await _ensure_authorized(update, context):
        return
    await _reply_text(
        update,
        STATUS_TEXT + "\n\n" + _distributor_selection_text(context),
        reply_markup=_distributor_keyboard(_selected_distributor_code(context)),
    )


async def distributor_callback(update: Any, context: Any) -> None:
    if not await _ensure_authorized(update, context):
        return

    query = getattr(update, "callback_query", None)
    if query is None:
        return

    data = str(getattr(query, "data", "") or "")
    distributor_code = _parse_distributor_callback_data(data)
    if distributor_code is None:
        await _answer_callback_query(query, DISTRIBUTOR_UNKNOWN_TEXT)
        return

    _set_selected_distributor_code(context, distributor_code)
    selected_text = DISTRIBUTOR_SELECTED_TEXT.format(
        distributor_label=_distributor_label(distributor_code)
    )
    await _answer_callback_query(query, selected_text)
    await _edit_callback_message(
        query,
        selected_text + "\n\n" + _distributor_selection_text(context),
        reply_markup=_distributor_keyboard(distributor_code),
    )


async def result_command(update: Any, context: Any) -> None:
    if not await _ensure_authorized(update, context):
        return

    message = _message(update)
    if message is None:
        return

    match_run_id = _parse_result_match_id(update, context)
    if match_run_id is None:
        await _send_reply_text(message, RESULT_USAGE_TEXT)
        return

    client = _stock_api_client(context)
    try:
        summary = await client.get_match_summary(match_run_id)
    except StockApiClientError as exc:
        logger.info(
            "Telegram result resend failed in stock-api: error_type=%s message=%s",
            type(exc).__name__,
            _safe_exception_message(exc),
        )
        await _send_reply_text(message, RESULT_LOAD_ERROR_TEXT)
        return

    try:
        card_sent = True
        if _is_v3_summary(summary):
            card_sent = await _send_v3_full_category_quote(
                message,
                summary,
                log_context="v3 stored result",
            )
        else:
            await _send_reply_text(message, format_match_summary(summary))
        await _send_report(message, client, summary, v3_card_sent=card_sent)
    except TelegramDeliveryError:
        await _send_result_created_hint(message, match_run_id)


async def v3_storage_command(update: Any, context: Any) -> None:
    await _handle_v3_full_category_command(update, context, profile="storage")


async def v3_server_command(update: Any, context: Any) -> None:
    await _handle_v3_full_category_command(update, context, profile="server")


async def v3_network_command(update: Any, context: Any) -> None:
    await _handle_v3_full_category_command(update, context, profile="network")


async def _handle_v3_full_category_command(
    update: Any,
    context: Any,
    *,
    profile: str,
) -> None:
    if not await _ensure_authorized(update, context):
        return

    message = _message(update)
    if message is None:
        return

    user_text = _parse_command_text(update, context)
    if not user_text:
        await _send_reply_text(message, V3_USAGE_TEXT)
        return

    try:
        distributor_code = _selected_distributor_code(context)
        await _send_reply_text(message, _v3_accepted_text(distributor_code))
    except TelegramDeliveryError:
        logger.info("Telegram v3 accepted message delivery failed after retries")

    client = _stock_api_client(context)
    try:
        summary = await client.create_simple_stock_quote(
            profile=profile,
            distributor_code=distributor_code,
            user_text=user_text,
        )
    except StockApiClientError as exc:
        logger.info(
            "Telegram v3 match request failed in stock-api: error_type=%s message=%s",
            type(exc).__name__,
            _safe_exception_message(exc),
        )
        await _send_reply_text(message, V3_ERROR_TEXT)
        return

    card_sent = await _send_v3_full_category_quote(
        message,
        summary,
        log_context="v3 result",
    )
    if _match_run_id(summary) is not None:
        await _send_report(message, client, summary, v3_card_sent=card_sent)


async def handle_text(update: Any, context: Any) -> None:
    if not await _ensure_authorized(update, context):
        return

    message = _message(update)
    if message is None:
        return

    user_text = (getattr(message, "text", None) or "").strip()
    if not user_text:
        return

    try:
        distributor_code = _selected_distributor_code(context)
        await _send_reply_text(message, _v3_accepted_text(distributor_code))
    except TelegramDeliveryError:
        logger.info("Telegram accepted message delivery failed after retries")

    client = _stock_api_client(context)

    try:
        summary = await client.create_v3_full_category_quote_auto(
            user_text,
            distributor_code=distributor_code,
        )
    except StockApiClientError as exc:
        logger.info(
            "Telegram v3 auto match request failed in stock-api: error_type=%s message=%s",
            type(exc).__name__,
            _safe_exception_message(exc),
        )
        await _send_reply_text(message, V3_ERROR_TEXT)
        return

    card_sent = await _send_v3_full_category_quote(
        message,
        summary,
        log_context="v3 auto result",
    )
    if _match_run_id(summary) is not None:
        await _send_report(message, client, summary, v3_card_sent=card_sent)


async def _send_report(
    message: Any,
    client: StockApiClient,
    summary: dict[str, Any],
    *,
    v3_card_sent: bool = True,
) -> None:
    match_run_id = _match_run_id(summary)
    if match_run_id is None:
        logger.warning("Telegram report delivery skipped: missing match_run_id")
        await _send_reply_text(message, REPORT_ERROR_TEXT)
        return

    try:
        report_xlsx = await client.get_match_report_xlsx(match_run_id)
    except StockApiClientError as exc:
        logger.info(
            "Telegram match Excel report request failed in stock-api: error_type=%s message=%s",
            type(exc).__name__,
            _safe_exception_message(exc),
        )
        await _send_reply_text(message, REPORT_ERROR_TEXT)
        return

    delivery = choose_excel_report_delivery(report_xlsx, match_run_id=match_run_id)
    if delivery.mode == "message":
        await _send_reply_text(message, delivery.text or "")
        return

    document = BytesIO(delivery.content or b"")
    document.name = delivery.filename or f"stock_match_{match_run_id}.xlsx"
    await _send_reply_document(
        message,
        document=document,
        filename=document.name,
        caption=_report_document_caption(
            summary,
            match_run_id=match_run_id,
            v3_card_sent=v3_card_sent,
        ),
    )
    await _send_reply_text(message, REPORT_SENT_TEXT)


async def _send_v3_full_category_quote(
    message: Any,
    summary: dict[str, Any],
    *,
    log_context: str,
) -> bool:
    delivery = choose_v3_full_category_quote_delivery(summary)
    try:
        if delivery.mode == "message":
            await _send_reply_text(message, delivery.text or "")
            return True

        document = BytesIO(delivery.content or b"")
        document.name = delivery.filename or "v3_quote.txt"
        await _send_reply_document(
            message,
            document=document,
            filename=document.name,
            caption="Полный КП draft",
        )
        return True
    except TelegramDeliveryError:
        logger.info("Telegram %s delivery failed after retries", log_context)
        return False


def _report_document_caption(
    summary: dict[str, Any],
    *,
    match_run_id: int,
    v3_card_sent: bool,
) -> str:
    if not v3_card_sent and _is_v3_summary(summary):
        return format_v3_excel_caption(summary, match_run_id=match_run_id)
    return f"Подбор по складу №{match_run_id}"


async def _ensure_authorized(update: Any, context: Any) -> bool:
    user_id = _user_id(update)
    access_controller = _access_controller(context)
    if access_controller.is_allowed(user_id):
        return True

    await _reply_text(update, ACCESS_DENIED_TEXT)
    return False


async def handle_error(update: object, context: Any) -> None:
    error = getattr(context, "error", None)
    logger.warning(
        "Telegram update handler failed: error_type=%s message=%s",
        type(error).__name__ if error is not None else "unknown",
        _safe_exception_message(error),
    )

    try:
        message = _message(update)
        if message is not None:
            await _send_reply_text(message, HANDLER_ERROR_TEXT)
            return

        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
        bot = getattr(context, "bot", None)
        if bot is not None and chat_id is not None:
            await _send_bot_message(bot, chat_id=chat_id, text=HANDLER_ERROR_TEXT)
    except TelegramDeliveryError:
        logger.info("Telegram error fallback delivery failed after retries")


def _user_id(update: Any) -> int | None:
    user = getattr(update, "effective_user", None)
    return getattr(user, "id", None)


def _message(update: Any) -> Any | None:
    return getattr(update, "effective_message", None)


async def _reply_text(update: Any, text: str, *, reply_markup: Any | None = None) -> None:
    message = _message(update)
    if message is not None:
        await _send_reply_text(message, text, reply_markup=reply_markup)


def _access_controller(context: Any) -> AccessController:
    return context.application.bot_data["access_controller"]


def _stock_api_client(context: Any) -> StockApiClient:
    return context.application.bot_data["stock_api_client"]


def _selected_distributor_code(context: Any) -> str:
    user_data = getattr(context, "user_data", None)
    if not isinstance(user_data, dict):
        return DEFAULT_DISTRIBUTOR_CODE
    return _normalize_distributor_code(user_data.get(DISTRIBUTOR_USER_DATA_KEY))


def _set_selected_distributor_code(context: Any, distributor_code: str) -> None:
    user_data = getattr(context, "user_data", None)
    if isinstance(user_data, dict):
        user_data[DISTRIBUTOR_USER_DATA_KEY] = _normalize_distributor_code(distributor_code)


def _normalize_distributor_code(value: object) -> str:
    code = str(value or "").strip().casefold()
    return code if code in SUPPORTED_DISTRIBUTORS else DEFAULT_DISTRIBUTOR_CODE


def _distributor_label(distributor_code: str) -> str:
    return SUPPORTED_DISTRIBUTORS.get(
        _normalize_distributor_code(distributor_code),
        SUPPORTED_DISTRIBUTORS[DEFAULT_DISTRIBUTOR_CODE],
    )


def _distributor_selection_text(context: Any) -> str:
    return DISTRIBUTOR_SELECT_TEXT.format(
        distributor_label=_distributor_label(_selected_distributor_code(context))
    )


def _v3_accepted_text(distributor_code: str) -> str:
    return (
        V3_ACCEPTED_TEXT
        + "\n"
        + f"Склад: {_distributor_label(distributor_code)}."
    )


def _distributor_keyboard(selected_distributor_code: str) -> Any | None:
    if InlineKeyboardButton is None or InlineKeyboardMarkup is None:
        return None

    selected_code = _normalize_distributor_code(selected_distributor_code)
    buttons = []
    for code, label in SUPPORTED_DISTRIBUTORS.items():
        prefix = "✓ " if code == selected_code else ""
        buttons.append(
            InlineKeyboardButton(
                f"{prefix}{label}",
                callback_data=f"{DISTRIBUTOR_CALLBACK_PREFIX}{code}",
            )
        )
    return InlineKeyboardMarkup([buttons])


def _parse_distributor_callback_data(data: str) -> str | None:
    if not data.startswith(DISTRIBUTOR_CALLBACK_PREFIX):
        return None
    code = data.removeprefix(DISTRIBUTOR_CALLBACK_PREFIX).strip().casefold()
    return code if code in SUPPORTED_DISTRIBUTORS else None


async def _answer_callback_query(query: Any, text: str) -> None:
    answer = getattr(query, "answer", None)
    if callable(answer):
        await answer(text=text)


async def _edit_callback_message(
    query: Any,
    text: str,
    *,
    reply_markup: Any | None = None,
) -> None:
    edit_message_text = getattr(query, "edit_message_text", None)
    if callable(edit_message_text):
        await _send_with_retry(
            "edit_message_text",
            lambda: edit_message_text(text=text, reply_markup=reply_markup),
        )
        return

    message = getattr(query, "message", None)
    if message is not None:
        await _send_reply_text(message, text, reply_markup=reply_markup)


def _parse_command_text(update: Any, context: Any) -> str:
    args = getattr(context, "args", None)
    if args:
        return " ".join(str(part).strip() for part in args if str(part).strip()).strip()

    message = _message(update)
    raw_text = getattr(message, "text", "") or ""
    parts = raw_text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _parse_result_match_id(update: Any, context: Any) -> int | None:
    args = getattr(context, "args", None)
    raw_value = str(args[0]).strip() if args else ""
    if not raw_value:
        message = _message(update)
        parts = (getattr(message, "text", "") or "").split(maxsplit=1)
        raw_value = parts[1].strip() if len(parts) > 1 else ""
    try:
        match_run_id = int(raw_value)
    except (TypeError, ValueError):
        return None
    if match_run_id <= 0:
        return None
    return match_run_id


def _match_run_id(summary: dict[str, Any]) -> int | None:
    value = summary.get("match_run_id")
    try:
        match_run_id = int(value)
    except (TypeError, ValueError):
        return None
    return match_run_id if match_run_id > 0 else None


def _is_v3_summary(summary: dict[str, Any]) -> bool:
    return (
        summary.get("pipeline_version") == "v3_full_category_matrix"
        or "v3_result_state" in summary
        or "result_state" in summary
    )


async def _send_result_created_hint(message: Any, match_run_id: int) -> None:
    try:
        await _send_reply_text(
            message,
            RESULT_CREATED_HINT_TEMPLATE.format(match_run_id=match_run_id),
        )
    except TelegramDeliveryError:
        logger.info("Telegram result-created hint delivery failed after retries")


async def _send_reply_text(
    message: Any,
    text: str,
    *,
    reply_markup: Any | None = None,
) -> None:
    kwargs = {"reply_markup": reply_markup} if reply_markup is not None else {}
    await _send_with_retry(
        "reply_text",
        lambda: message.reply_text(text, **kwargs),
    )


async def _send_bot_message(bot: Any, *, chat_id: int, text: str) -> None:
    await _send_with_retry(
        "send_message",
        lambda: bot.send_message(chat_id=chat_id, text=text),
    )


async def _send_reply_document(
    message: Any,
    *,
    document: BytesIO,
    filename: str,
    caption: str,
) -> None:
    async def send_document() -> Any:
        document.seek(0)
        return await message.reply_document(
            document=document,
            filename=filename,
            caption=caption,
        )

    await _send_with_retry(
        "send_document",
        send_document,
    )


async def _send_with_retry(
    operation: str,
    send: Callable[[], Awaitable[Any]],
) -> Any:
    for attempt in range(1, TELEGRAM_SEND_MAX_ATTEMPTS + 1):
        try:
            return await send()
        except _TRANSIENT_SEND_ERRORS as exc:
            if attempt >= TELEGRAM_SEND_MAX_ATTEMPTS:
                logger.warning(
                    "Telegram delivery failed after retries: operation=%s error_type=%s "
                    "message=%s",
                    operation,
                    type(exc).__name__,
                    _safe_exception_message(exc),
                )
                raise TelegramDeliveryError(operation) from exc

            delay_seconds = _retry_delay_seconds(exc, attempt)
            logger.info(
                "Telegram delivery retry: operation=%s attempt=%s/%s error_type=%s "
                "delay_seconds=%.2f message=%s",
                operation,
                attempt,
                TELEGRAM_SEND_MAX_ATTEMPTS,
                type(exc).__name__,
                delay_seconds,
                _safe_exception_message(exc),
            )
            await asyncio.sleep(delay_seconds)

    raise TelegramDeliveryError(operation)


def _retry_delay_seconds(exc: BaseException, attempt: int) -> float:
    if RetryAfter is not None and isinstance(exc, RetryAfter):
        retry_after = getattr(exc, "retry_after", None)
        retry_after_seconds = _seconds_value(retry_after)
        if retry_after_seconds is not None:
            return min(max(retry_after_seconds, 0.0), TELEGRAM_RETRY_AFTER_MAX_SECONDS)

    delay = TELEGRAM_SEND_BASE_BACKOFF_SECONDS * (2 ** max(attempt - 1, 0))
    return min(delay, TELEGRAM_SEND_MAX_BACKOFF_SECONDS)


def _seconds_value(value: Any) -> float | None:
    if isinstance(value, timedelta):
        return value.total_seconds()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_exception_message(exc: object, *, limit: int = 160) -> str:
    text = str(exc or "").strip()
    if not text:
        return ""
    text = re.sub(r"\b\d{5,}:[A-Za-z0-9_-]+\b", "[redacted-token]", text)
    text = re.sub(r"\b\d{5,}\b", "[redacted-number]", text)
    text = re.sub(
        r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@",
        r"\1[redacted]@",
        text,
    )
    text = re.sub(
        r"(authorization|proxy-authorization|token)=\S+",
        r"\1=[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(['\"]?(authorization|proxy-authorization|token)['\"]?\s*:\s*)['\"]?[^,}\s]+",
        r"\1[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    return text[:limit]
