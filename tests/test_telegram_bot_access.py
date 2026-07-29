from __future__ import annotations

import pytest

from app.telegram_bot.handlers import AccessController, parse_allowed_user_ids


def test_parse_allowed_user_ids_accepts_commas_spaces_and_semicolons() -> None:
    assert parse_allowed_user_ids("100, 200;300\n400") == frozenset({100, 200, 300, 400})


def test_access_control_allows_configured_user() -> None:
    access_controller = AccessController.from_config("100,200", environment="prod")

    assert access_controller.is_allowed(100) is True
    assert access_controller.is_allowed(300) is False


def test_access_control_allows_all_in_dev_when_empty() -> None:
    access_controller = AccessController.from_config("", environment="dev")

    assert access_controller.is_allowed(100) is True


def test_access_control_denies_all_outside_dev_when_empty() -> None:
    access_controller = AccessController.from_config("", environment="prod")

    assert access_controller.is_allowed(100) is False
    assert access_controller.is_allowed(None) is False


def test_parse_allowed_user_ids_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="integer user ids"):
        parse_allowed_user_ids("100,abc")
