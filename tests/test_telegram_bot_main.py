from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from app.core.config import TelegramSettings

EXPECTED_TIMEOUT_KWARGS = {
    "read_timeout": 60.0,
    "write_timeout": 60.0,
    "connect_timeout": 60.0,
    "pool_timeout": 60.0,
    "media_write_timeout": 60.0,
}


@pytest.fixture(autouse=True)
def clean_telegram_main_module() -> Iterator[None]:
    sys.modules.pop("app.telegram_bot.main", None)
    yield
    sys.modules.pop("app.telegram_bot.main", None)


def test_create_application_applies_proxy_to_bot_api_and_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_class = _request_class_with_proxy()
    application_class = _install_fake_telegram_modules(monkeypatch, request_class)
    telegram_main = importlib.import_module("app.telegram_bot.main")

    application = telegram_main.create_application(
        TelegramSettings(
            telegram_bot_token="test-token",
            telegram_proxy_url="socks5://sing-box:1080",
        )
    )

    assert len(request_class.instances) == 2
    assert request_class.instances[0].kwargs == _expected_request_kwargs("proxy")
    assert request_class.instances[1].kwargs == _expected_request_kwargs("proxy")
    assert application_class.last_builder.bot_api_request is request_class.instances[0]
    assert application_class.last_builder.polling_request is request_class.instances[1]
    assert application.bot_data["stock_api_client"] is not None
    assert application.bot_data["access_controller"] is not None
    assert [handler[1][0] for handler in application.handlers if handler[0] == "command"] == [
        "start",
        "help",
        "status",
        "result",
        "v3_storage",
        "v3_server",
        "v3_network",
    ]
    assert [handler for handler in application.handlers if handler[0] == "callback"]
    assert len(application.error_handlers) == 1


def test_telegram_proxy_kwargs_includes_proxy_and_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_class = _request_class_with_proxy()
    _install_fake_telegram_modules(monkeypatch, request_class)
    telegram_main = importlib.import_module("app.telegram_bot.main")

    kwargs = telegram_main._telegram_proxy_kwargs("socks5://sing-box:1080", 60)

    assert kwargs == _expected_request_kwargs("proxy")


def test_create_application_supports_proxy_url_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_class = _request_class_with_proxy_url()
    application_class = _install_fake_telegram_modules(monkeypatch, request_class)
    telegram_main = importlib.import_module("app.telegram_bot.main")

    telegram_main.create_application(
        TelegramSettings(
            telegram_bot_token="test-token",
            telegram_proxy_url="socks5://sing-box:1080",
        )
    )

    assert request_class.instances[0].kwargs == _expected_request_kwargs("proxy_url")
    assert request_class.instances[1].kwargs == _expected_request_kwargs("proxy_url")
    assert application_class.last_builder.bot_api_request is request_class.instances[0]
    assert application_class.last_builder.polling_request is request_class.instances[1]


def test_create_application_logs_proxy_scheme_without_full_url(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request_class = _request_class_with_proxy()
    _install_fake_telegram_modules(monkeypatch, request_class)
    telegram_main = importlib.import_module("app.telegram_bot.main")

    caplog.set_level(logging.INFO, logger="app.telegram_bot.main")
    telegram_main.create_application(
        TelegramSettings(
            telegram_bot_token="test-token",
            telegram_proxy_url="socks5://sing-box:1080",
        )
    )

    assert "Telegram proxy enabled: socks5" in caplog.text
    assert "socks5://sing-box:1080" not in caplog.text
    assert "sing-box" not in caplog.text


def test_create_application_suppresses_sensitive_request_loggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_class = _request_class_with_proxy()
    _install_fake_telegram_modules(monkeypatch, request_class)
    telegram_main = importlib.import_module("app.telegram_bot.main")
    previous_levels = {
        name: logging.getLogger(name).level
        for name in ("httpx", "httpcore", "telegram.request")
    }

    try:
        for logger_name in previous_levels:
            logging.getLogger(logger_name).setLevel(logging.INFO)

        telegram_main.create_application(
            TelegramSettings(
                telegram_bot_token="test-token",
                telegram_proxy_url="",
            )
        )

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("telegram.request").level == logging.WARNING
    finally:
        for logger_name, level in previous_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def test_main_retries_telegram_bootstrap_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_class = _request_class_with_proxy()
    application_class = _install_fake_telegram_modules(monkeypatch, request_class)
    telegram_main = importlib.import_module("app.telegram_bot.main")
    monkeypatch.setattr(
        telegram_main,
        "get_telegram_settings",
        lambda: TelegramSettings(
            telegram_bot_token="test-token",
            telegram_proxy_url="socks5://sing-box:1080",
        ),
    )

    telegram_main.main()

    assert application_class.last_application.run_polling_kwargs == {
        "bootstrap_retries": -1
    }


def test_create_application_rejects_httpxrequest_without_proxy_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_class = _request_class_without_proxy()
    _install_fake_telegram_modules(monkeypatch, request_class)
    telegram_main = importlib.import_module("app.telegram_bot.main")

    with pytest.raises(RuntimeError, match="does not support Telegram proxy"):
        telegram_main.create_application(
            TelegramSettings(
                telegram_bot_token="test-token",
                telegram_proxy_url="socks5://sing-box:1080",
            )
        )


class _RecordingBuilder:
    def __init__(self, application_class: type[_RecordingApplication]) -> None:
        self.application_class = application_class
        self.token_value: str | None = None
        self.shutdown_callback: Any | None = None
        self.bot_api_request: Any | None = None
        self.polling_request: Any | None = None

    def token(self, token: str) -> _RecordingBuilder:
        self.token_value = token
        return self

    def post_shutdown(self, callback: Any) -> _RecordingBuilder:
        self.shutdown_callback = callback
        return self

    def request(self, request: Any) -> _RecordingBuilder:
        self.bot_api_request = request
        return self

    def get_updates_request(self, request: Any) -> _RecordingBuilder:
        self.polling_request = request
        return self

    def build(self) -> _RecordingApplication:
        return self.application_class()


class _RecordingApplication:
    last_builder: _RecordingBuilder
    last_application: _RecordingApplication

    def __init__(self) -> None:
        self.__class__.last_application = self
        self.bot_data: dict[str, Any] = {}
        self.handlers: list[Any] = []
        self.error_handlers: list[Any] = []
        self.run_polling_kwargs: dict[str, Any] | None = None

    @classmethod
    def builder(cls) -> _RecordingBuilder:
        cls.last_builder = _RecordingBuilder(cls)
        return cls.last_builder

    def add_handler(self, handler: Any) -> None:
        self.handlers.append(handler)

    def add_error_handler(self, callback: Any) -> None:
        self.error_handlers.append(callback)

    def run_polling(self, **kwargs: Any) -> None:
        self.run_polling_kwargs = kwargs


class _FakeFilter:
    def __and__(self, other: Any) -> _FakeFilter:
        return self

    def __invert__(self) -> _FakeFilter:
        return self


def _install_fake_telegram_modules(
    monkeypatch: pytest.MonkeyPatch,
    request_class: type[Any],
) -> type[_RecordingApplication]:
    telegram_package = types.ModuleType("telegram")
    telegram_package.__path__ = []

    telegram_request = types.ModuleType("telegram.request")
    telegram_request.HTTPXRequest = request_class

    telegram_ext = types.ModuleType("telegram.ext")
    telegram_ext.Application = _RecordingApplication
    telegram_ext.CallbackQueryHandler = _fake_callback_query_handler
    telegram_ext.CommandHandler = _fake_command_handler
    telegram_ext.MessageHandler = _fake_message_handler
    telegram_ext.filters = types.SimpleNamespace(TEXT=_FakeFilter(), COMMAND=_FakeFilter())

    monkeypatch.setitem(sys.modules, "telegram", telegram_package)
    monkeypatch.setitem(sys.modules, "telegram.request", telegram_request)
    monkeypatch.setitem(sys.modules, "telegram.ext", telegram_ext)
    return _RecordingApplication


def _fake_command_handler(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return ("command", args, kwargs)


def _fake_callback_query_handler(
    *args: Any,
    **kwargs: Any,
) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return ("callback", args, kwargs)


def _fake_message_handler(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return ("message", args, kwargs)


def _expected_request_kwargs(proxy_parameter: str) -> dict[str, str | float]:
    return {proxy_parameter: "socks5://sing-box:1080", **EXPECTED_TIMEOUT_KWARGS}


def _request_class_with_proxy() -> type[Any]:
    class RequestWithProxy:
        instances: list[Any] = []

        def __init__(
            self,
            *,
            proxy: str | None = None,
            read_timeout: float | None = None,
            write_timeout: float | None = None,
            connect_timeout: float | None = None,
            pool_timeout: float | None = None,
            media_write_timeout: float | None = None,
        ) -> None:
            self.kwargs = {
                "proxy": proxy,
                "read_timeout": read_timeout,
                "write_timeout": write_timeout,
                "connect_timeout": connect_timeout,
                "pool_timeout": pool_timeout,
                "media_write_timeout": media_write_timeout,
            }
            self.__class__.instances.append(self)

    return RequestWithProxy


def _request_class_with_proxy_url() -> type[Any]:
    class RequestWithProxyUrl:
        instances: list[Any] = []

        def __init__(
            self,
            *,
            proxy_url: str | None = None,
            read_timeout: float | None = None,
            write_timeout: float | None = None,
            connect_timeout: float | None = None,
            pool_timeout: float | None = None,
            media_write_timeout: float | None = None,
        ) -> None:
            self.kwargs = {
                "proxy_url": proxy_url,
                "read_timeout": read_timeout,
                "write_timeout": write_timeout,
                "connect_timeout": connect_timeout,
                "pool_timeout": pool_timeout,
                "media_write_timeout": media_write_timeout,
            }
            self.__class__.instances.append(self)

    return RequestWithProxyUrl


def _request_class_without_proxy() -> type[Any]:
    class RequestWithoutProxy:
        def __init__(self) -> None:
            pass

    return RequestWithoutProxy
