import re
from pathlib import Path

from app.core.config import (
    LlmSettings,
    OcsSettings,
    Settings,
    TelegramSettings,
    TreolanSettings,
    WebEvidenceSettings,
)


def _service_section(compose: str, service: str, next_service: str | None) -> str:
    section = compose.split(f"  {service}:", maxsplit=1)[1]
    if next_service:
        section = section.split(f"  {next_service}:", maxsplit=1)[0]
    return section


def _environment_keys(service_section: str) -> set[str]:
    environment = service_section.split("    environment:", maxsplit=1)[1]
    environment = environment.split("    depends_on:", maxsplit=1)[0]
    return set(re.findall(r"^      ([A-Z][A-Z0-9_]+):", environment, flags=re.MULTILINE))


def test_llm_configurator_max_package_chars_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CONFIGURATOR_MAX_PACKAGE_CHARS", "200000")

    assert LlmSettings().llm_configurator_max_package_chars == 200000


def test_llm_configurator_high_quality_defaults() -> None:
    settings = LlmSettings(_env_file=None)

    assert settings.llm_model == "qwen/qwen3.7-plus"
    assert settings.llm_configurator_max_package_chars == 1500000
    assert settings.llm_configurator_read_timeout_seconds == 1800
    assert settings.llm_configurator_max_output_tokens == 65536
    assert settings.high_quality_full_matrix_by_default is True
    assert settings.v3_refresh_categories_before_llm is True


def test_high_quality_full_matrix_by_default_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("HIGH_QUALITY_FULL_MATRIX_BY_DEFAULT", "false")

    assert LlmSettings().high_quality_full_matrix_by_default is False


def test_llm_full_matrix_timeouts_read_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_FULL_MATRIX_MAX_SECONDS", "1200")
    monkeypatch.setenv("LLM_FULL_MATRIX_CHUNK_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("LLM_FULL_MATRIX_FORCE", "true")

    settings = LlmSettings()

    assert settings.llm_full_matrix_max_seconds == 1200
    assert settings.llm_full_matrix_chunk_timeout_seconds == 240
    assert settings.llm_full_matrix_force is True


def test_treolan_settings_read_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("TREOLAN_API_BASE_URL", "https://api.treolan.example")
    monkeypatch.setenv("TREOLAN_LOGIN", "treolan-login")
    monkeypatch.setenv("TREOLAN_PASSWORD", "treolan-password")
    monkeypatch.setenv("TREOLAN_REQUEST_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("TREOLAN_REQUESTS_PER_MINUTE_LIMIT", "12")
    monkeypatch.setenv("TREOLAN_CATALOG_FREE_NOM_ONLY", "false")

    settings = TreolanSettings()

    assert settings.treolan_base_url == "https://api.treolan.example"
    assert settings.treolan_login == "treolan-login"
    assert settings.treolan_password == "treolan-password"
    assert settings.treolan_timeout_seconds == 45
    assert settings.treolan_requests_per_minute_limit == 12
    assert settings.treolan_catalog_free_nom_only is False


def test_llm_semantic_planner_timeouts_read_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_SEMANTIC_PLANNER_MAX_SECONDS", "240")
    monkeypatch.setenv("LLM_SEMANTIC_PLANNER_STAGE_TIMEOUT_SECONDS", "90")

    settings = LlmSettings()

    assert settings.llm_semantic_planner_max_seconds == 240
    assert settings.llm_semantic_planner_stage_timeout_seconds == 90


def test_docker_compose_passes_all_api_settings_to_runtime() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    api_keys = _environment_keys(_service_section(compose, "stock-api", "stock-bot"))
    settings_classes = (OcsSettings, TreolanSettings, LlmSettings, WebEvidenceSettings)
    expected_keys = {
        field_name.upper()
        for settings_class in settings_classes
        for field_name in settings_class.model_fields
    }
    expected_keys.update({"SERVICE_NAME", "ENVIRONMENT", "LOG_LEVEL", "DATABASE_URL"})

    assert expected_keys <= api_keys
    assert {
        "OCS_API_BASE_URL",
        "OCS_REQUEST_TIMEOUT_SECONDS",
        "TREOLAN_API_BASE_URL",
        "TREOLAN_REQUEST_TIMEOUT_SECONDS",
        "LLM_COMPOSER_FIRST_PIPELINE",
    } <= api_keys


def test_docker_compose_passes_only_bot_settings_to_bot_runtime() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    bot_keys = _environment_keys(_service_section(compose, "stock-bot", "stock-postgres"))
    expected_keys = {field_name.upper() for field_name in TelegramSettings.model_fields}

    assert expected_keys <= bot_keys
    assert "LLM_API_KEY" not in bot_keys
    assert "OCS_API_KEY" not in bot_keys


def test_docker_compose_services_survive_host_reboot() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    service_sections = {
        "stock-api": _service_section(compose, "stock-api", "stock-bot"),
        "stock-bot": _service_section(compose, "stock-bot", "stock-postgres"),
        "stock-postgres": compose.split("  stock-postgres:", maxsplit=1)[1].split(
            "volumes:", maxsplit=1
        )[0],
    }

    for service_name, section in service_sections.items():
        assert "restart: unless-stopped" in section, service_name


def test_env_example_lists_all_application_settings() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")
    env_keys = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", env_example, flags=re.MULTILINE))
    expected_keys = {field_name.upper() for field_name in Settings.model_fields}

    assert expected_keys <= env_keys
    assert {
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "OCS_API_BASE_URL",
        "OCS_REQUEST_TIMEOUT_SECONDS",
        "TREOLAN_API_BASE_URL",
        "TREOLAN_REQUEST_TIMEOUT_SECONDS",
        "LLM_COMPOSER_FIRST_PIPELINE",
    } <= env_keys


def test_dockerfile_installs_the_frozen_runtime_lock() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY pyproject.toml uv.lock" in dockerfile
    assert "uv sync --frozen --no-dev --no-cache" in dockerfile
    assert "pip install --no-cache-dir ." not in dockerfile
