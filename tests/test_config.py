from pathlib import Path

from app.core.config import LlmSettings, TreolanSettings


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


def test_docker_compose_passes_llm_package_limit_to_runtime() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    stock_api_section = compose.split("  stock-api:", maxsplit=1)[1].split(
        "  stock-bot:",
        maxsplit=1,
    )[0]
    stock_bot_section = compose.split("  stock-bot:", maxsplit=1)[1].split(
        "  stock-postgres:",
        maxsplit=1,
    )[0]

    expected = "LLM_CONFIGURATOR_MAX_PACKAGE_CHARS: ${LLM_CONFIGURATOR_MAX_PACKAGE_CHARS:-1500000}"
    assert expected in stock_api_section
    assert expected in stock_bot_section
    assert "LLM_MODEL: ${LLM_MODEL:-qwen/qwen3.7-plus}" in stock_api_section
    treolan_flags = [
        "TREOLAN_BASE_URL: ${TREOLAN_BASE_URL:-${TREOLAN_API_BASE_URL:-https://demo-api.treolan.ru}}",
        "TREOLAN_API_BASE_URL: ${TREOLAN_API_BASE_URL:-${TREOLAN_BASE_URL:-https://demo-api.treolan.ru}}",
        "TREOLAN_LOGIN: ${TREOLAN_LOGIN:-}",
        "TREOLAN_PASSWORD: ${TREOLAN_PASSWORD:-}",
        "TREOLAN_REQUESTS_PER_MINUTE_LIMIT: ${TREOLAN_REQUESTS_PER_MINUTE_LIMIT:-30}",
    ]
    for treolan_flag in treolan_flags:
        assert treolan_flag in stock_api_section
    assert (
        "LLM_CONFIGURATOR_READ_TIMEOUT_SECONDS: "
        "${LLM_CONFIGURATOR_READ_TIMEOUT_SECONDS:-1800}"
    ) in stock_api_section
    assert (
        "LLM_CONFIGURATOR_MAX_OUTPUT_TOKENS: "
        "${LLM_CONFIGURATOR_MAX_OUTPUT_TOKENS:-65536}"
    ) in stock_api_section
    high_quality = (
        "HIGH_QUALITY_FULL_MATRIX_BY_DEFAULT: "
        "${HIGH_QUALITY_FULL_MATRIX_BY_DEFAULT:-true}"
    )
    assert high_quality in stock_api_section
    assert high_quality in stock_bot_section
    assert (
        "TELEGRAM_V3_REQUEST_TIMEOUT_SECONDS: ${TELEGRAM_V3_REQUEST_TIMEOUT_SECONDS:-1800}"
        in stock_bot_section
    )
    runtime_v2_flags = [
        "STOCK_MATCH_PIPELINE_V2: ${STOCK_MATCH_PIPELINE_V2:-true}",
        "LLM_COMPOSER_FIRST_PIPELINE: ${LLM_COMPOSER_FIRST_PIPELINE:-true}",
        "STOCK_MATCH_PIPELINE_V2_MODE: ${STOCK_MATCH_PIPELINE_V2_MODE:-composer_cascade}",
        "LLM_ROLE_EVALUATION_ENABLED: ${LLM_ROLE_EVALUATION_ENABLED:-false}",
        "LLM_COMPOSER_CRITIC_ENABLED: ${LLM_COMPOSER_CRITIC_ENABLED:-true}",
        "LLM_COMPOSER_REPAIR_MAX_ATTEMPTS: ${LLM_COMPOSER_REPAIR_MAX_ATTEMPTS:-1}",
        "LLM_MAX_CALLS_PER_MATCH: ${LLM_MAX_CALLS_PER_MATCH:-6}",
        "LLM_COMPOSER_MULTI_PASS: ${LLM_COMPOSER_MULTI_PASS:-false}",
        "V3_REFRESH_CATEGORIES_BEFORE_LLM: ${V3_REFRESH_CATEGORIES_BEFORE_LLM:-true}",
    ]
    for runtime_flag in runtime_v2_flags:
        assert runtime_flag in stock_api_section
        assert runtime_flag in stock_bot_section
    assert "LLM_FULL_MATRIX_MAX_SECONDS: ${LLM_FULL_MATRIX_MAX_SECONDS:-900}" in stock_api_section
    assert (
        "LLM_FULL_MATRIX_CHUNK_TIMEOUT_SECONDS: ${LLM_FULL_MATRIX_CHUNK_TIMEOUT_SECONDS:-300}"
        in stock_api_section
    )
    assert (
        "LLM_SEMANTIC_PLANNER_MAX_SECONDS: ${LLM_SEMANTIC_PLANNER_MAX_SECONDS:-300}"
        in stock_api_section
    )
    assert (
        "LLM_SEMANTIC_PLANNER_STAGE_TIMEOUT_SECONDS: "
        "${LLM_SEMANTIC_PLANNER_STAGE_TIMEOUT_SECONDS:-120}"
    ) in stock_api_section
    assert "LLM_FULL_MATRIX_FORCE: ${LLM_FULL_MATRIX_FORCE:-false}" in stock_api_section
    assert (
        "LLM_CONFIGURATOR_NO_RECOMMENDATION_MIN_LARGE_ROLE_CANDIDATES: "
        "${LLM_CONFIGURATOR_NO_RECOMMENDATION_MIN_LARGE_ROLE_CANDIDATES:-12}"
    ) in stock_api_section


def test_docker_compose_services_survive_host_reboot() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    service_sections = {
        "stock-api": compose.split("  stock-api:", maxsplit=1)[1].split(
            "  stock-bot:", maxsplit=1
        )[0],
        "stock-bot": compose.split("  stock-bot:", maxsplit=1)[1].split(
            "  stock-postgres:", maxsplit=1
        )[0],
        "stock-postgres": compose.split("  stock-postgres:", maxsplit=1)[1].split(
            "volumes:", maxsplit=1
        )[0],
    }

    for service_name, section in service_sections.items():
        assert "restart: unless-stopped" in section, service_name


def test_env_example_lists_v2_runtime_flags() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "LLM_MODEL=qwen/qwen3.7-plus" in env_example
    assert "TREOLAN_BASE_URL=https://demo-api.treolan.ru" in env_example
    assert "TREOLAN_LOGIN=" in env_example
    assert "TREOLAN_PASSWORD=" in env_example
    assert "TREOLAN_REQUESTS_PER_MINUTE_LIMIT=30" in env_example
    assert "LLM_CONFIGURATOR_READ_TIMEOUT_SECONDS=1800" in env_example
    assert "LLM_CONFIGURATOR_MAX_OUTPUT_TOKENS=65536" in env_example
    assert "STOCK_MATCH_PIPELINE_V2=true" in env_example
    assert "LLM_COMPOSER_FIRST_PIPELINE=true" in env_example
    assert "TELEGRAM_V3_REQUEST_TIMEOUT_SECONDS=1800" in env_example
    assert "STOCK_MATCH_PIPELINE_V2_MODE=composer_cascade" in env_example
    assert "LLM_ROLE_EVALUATION_ENABLED=false" in env_example
    assert "LLM_COMPOSER_CRITIC_ENABLED=true" in env_example
    assert "LLM_COMPOSER_REPAIR_MAX_ATTEMPTS=1" in env_example
    assert "LLM_MAX_CALLS_PER_MATCH=6" in env_example
    assert "LLM_COMPOSER_MULTI_PASS=false" in env_example
    assert "V3_REFRESH_CATEGORIES_BEFORE_LLM=true" in env_example
