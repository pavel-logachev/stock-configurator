from __future__ import annotations

import logging
from inspect import signature
from typing import Any
from urllib.parse import urlsplit

from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters
from telegram.request import HTTPXRequest

from app.core.config import TelegramSettings, get_telegram_settings
from app.core.logging import setup_logging
from app.telegram_bot.handlers import (
    DISTRIBUTOR_CALLBACK_PREFIX,
    AccessController,
    distributor_callback,
    handle_error,
    handle_text,
    help_command,
    result_command,
    start,
    status,
    v3_network_command,
    v3_server_command,
    v3_storage_command,
)
from app.telegram_bot.stock_api_client import StockApiClient

logger = logging.getLogger(__name__)

GET_UPDATES_REQUEST_METHODS = ("get_updates_request", "get_update_request")
TELEGRAM_TIMEOUT_PARAMETERS = (
    "read_timeout",
    "write_timeout",
    "connect_timeout",
    "pool_timeout",
    "media_write_timeout",
)
SENSITIVE_REQUEST_LOGGERS = ("httpx", "httpcore", "telegram.request")


def create_application(settings: TelegramSettings) -> Application:
    suppress_sensitive_request_logs()
    if not settings.telegram_bot_token or settings.telegram_bot_token == "change-me":
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be set for stock-bot.")

    builder = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_shutdown(_close_stock_api_client)
    )
    builder = _configure_telegram_proxy(
        builder,
        proxy_url=settings.telegram_proxy_url,
        timeout_seconds=settings.telegram_request_timeout_seconds,
    )

    application = builder.build()

    access_controller = AccessController.from_config(
        settings.telegram_allowed_user_ids,
        environment=settings.environment,
    )
    stock_api_client = StockApiClient(
        base_url=settings.stock_api_base_url,
        timeout_seconds=settings.telegram_request_timeout_seconds,
        v3_timeout_seconds=settings.telegram_v3_request_timeout_seconds,
    )

    application.bot_data["access_controller"] = access_controller
    application.bot_data["stock_api_client"] = stock_api_client

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("result", result_command))
    application.add_handler(CommandHandler("v3_storage", v3_storage_command))
    application.add_handler(CommandHandler("v3_server", v3_server_command))
    application.add_handler(CommandHandler("v3_network", v3_network_command))
    application.add_handler(
        CallbackQueryHandler(
            distributor_callback,
            pattern=f"^{DISTRIBUTOR_CALLBACK_PREFIX}",
        )
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(handle_error)
    return application


def _configure_telegram_proxy(builder: Any, *, proxy_url: str, timeout_seconds: float) -> Any:
    normalized_proxy_url = proxy_url.strip()
    if not normalized_proxy_url:
        return builder

    logger.info("Telegram proxy enabled: %s", _proxy_scheme(normalized_proxy_url))
    request_method = getattr(builder, "request", None)
    if not callable(request_method):
        raise RuntimeError(
            "Installed python-telegram-bot ApplicationBuilder does not support custom "
            "request configuration."
        )

    builder = (
        request_method(_create_telegram_request(normalized_proxy_url, timeout_seconds)) or builder
    )
    get_updates_request_method = _get_builder_method(builder, GET_UPDATES_REQUEST_METHODS)
    if get_updates_request_method is None:
        logger.warning(
            "Installed python-telegram-bot ApplicationBuilder does not expose "
            "get_updates_request; polling requests may not use Telegram proxy."
        )
        return builder

    return (
        get_updates_request_method(
            _create_telegram_request(normalized_proxy_url, timeout_seconds)
        )
        or builder
    )


def _create_telegram_request(proxy_url: str, timeout_seconds: float) -> HTTPXRequest:
    return HTTPXRequest(**_telegram_proxy_kwargs(proxy_url, timeout_seconds))


def _telegram_proxy_kwargs(proxy_url: str, timeout_seconds: float) -> dict[str, str | float]:
    request_parameters = signature(HTTPXRequest).parameters
    kwargs: dict[str, str | float] = {}
    if "proxy" in request_parameters:
        kwargs["proxy"] = proxy_url
    elif "proxy_url" in request_parameters:
        kwargs["proxy_url"] = proxy_url
    else:
        raise RuntimeError(
            "Installed python-telegram-bot HTTPXRequest does not support Telegram proxy "
            "configuration. Install python-telegram-bot[socks]>=22,<23."
        )

    for timeout_parameter in TELEGRAM_TIMEOUT_PARAMETERS:
        if timeout_parameter in request_parameters:
            kwargs[timeout_parameter] = timeout_seconds
    return kwargs


def _get_builder_method(builder: Any, method_names: tuple[str, ...]) -> Any | None:
    for method_name in method_names:
        method = getattr(builder, method_name, None)
        if callable(method):
            return method
    return None


def _proxy_scheme(proxy_url: str) -> str:
    return urlsplit(proxy_url).scheme or "unknown"


def suppress_sensitive_request_logs() -> None:
    for logger_name in SENSITIVE_REQUEST_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


async def _close_stock_api_client(application: Application) -> None:
    stock_api_client = application.bot_data.get("stock_api_client")
    if isinstance(stock_api_client, StockApiClient):
        await stock_api_client.aclose()


def main() -> None:
    settings = get_telegram_settings()
    setup_logging(settings.log_level)
    suppress_sensitive_request_logs()
    logger.info("Starting stock-bot")
    application = create_application(settings)
    application.run_polling(bootstrap_retries=-1)


if __name__ == "__main__":
    main()
