from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LLM_MODEL = "qwen/qwen3.7-plus"


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


class OcsSettings(EnvSettings):
    ocs_base_url: str = Field(
        default="https://testconnector.b2b.ocs.ru",
        validation_alias=AliasChoices("OCS_BASE_URL", "OCS_API_BASE_URL"),
    )
    ocs_api_key: str = Field(default="", repr=False, validation_alias="OCS_API_KEY")
    ocs_shipment_city: str = "Москва"
    ocs_timeout_seconds: float = Field(
        default=30,
        gt=0,
        validation_alias=AliasChoices("OCS_TIMEOUT_SECONDS", "OCS_REQUEST_TIMEOUT_SECONDS"),
    )
    ocs_request_delay_seconds: float = Field(default=0, ge=0)
    ocs_requests_per_hour_limit: int = Field(default=180, ge=0, le=200)
    ocs_content_enabled: bool = True
    ocs_content_batch_size: int = Field(default=50, ge=1, le=100)
    ocs_content_max_items_per_run: int = Field(default=120, ge=0, le=180)
    ocs_content_cache_ttl_hours: int = Field(default=168, ge=0)


class TreolanSettings(EnvSettings):
    treolan_base_url: str = Field(
        default="https://demo-api.treolan.ru",
        validation_alias=AliasChoices("TREOLAN_BASE_URL", "TREOLAN_API_BASE_URL"),
    )
    treolan_login: str = Field(default="", repr=False, validation_alias="TREOLAN_LOGIN")
    treolan_password: str = Field(
        default="",
        repr=False,
        validation_alias="TREOLAN_PASSWORD",
    )
    treolan_shipment_city: str = "Treolan"
    treolan_timeout_seconds: float = Field(
        default=30,
        gt=0,
        validation_alias=AliasChoices(
            "TREOLAN_TIMEOUT_SECONDS",
            "TREOLAN_REQUEST_TIMEOUT_SECONDS",
        ),
    )
    treolan_request_delay_seconds: float = Field(default=0, ge=0)
    treolan_requests_per_minute_limit: int = Field(default=30, ge=0)
    treolan_catalog_criterion: int = Field(default=1, ge=0)
    treolan_catalog_show_nc: int = Field(default=1, ge=0)
    treolan_catalog_free_nom_only: bool = True


class LlmSettings(EnvSettings):
    llm_provider: str = "disabled"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = DEFAULT_LLM_MODEL
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_configurator_enabled: bool = False
    llm_configurator_mode: str = "disabled"
    llm_configurator_output_mode: str = "single_best_cost_valid"
    llm_component_candidates_per_role: int = Field(default=100, ge=1)
    llm_configurator_max_package_chars: int = Field(default=1500000, ge=10000)
    high_quality_full_matrix_by_default: bool = True
    llm_build_recommendations_limit: int = Field(default=5, ge=1)
    llm_proposal_pool_limit: int = Field(default=10, ge=1)
    llm_configurator_timeout_seconds: float = Field(default=60, gt=0)
    llm_configurator_read_timeout_seconds: float = Field(default=1800, gt=0)
    llm_configurator_max_output_tokens: int = Field(default=65536, ge=1)
    llm_semantic_planner_max_seconds: float = Field(default=300, gt=0)
    llm_semantic_planner_stage_timeout_seconds: float = Field(default=120, gt=0)
    llm_full_matrix_max_seconds: float = Field(default=900, gt=0)
    llm_full_matrix_chunk_timeout_seconds: float = Field(default=300, gt=0)
    llm_full_matrix_force: bool = False
    llm_composer_multi_pass: bool = Field(
        default=False,
        validation_alias="LLM_COMPOSER_MULTI_PASS",
    )
    stock_match_pipeline_v2_mode: str = Field(
        default="composer_cascade",
        validation_alias="STOCK_MATCH_PIPELINE_V2_MODE",
    )
    llm_role_evaluation_enabled: bool = Field(
        default=False,
        validation_alias="LLM_ROLE_EVALUATION_ENABLED",
    )
    llm_composer_critic_enabled: bool = Field(
        default=True,
        validation_alias="LLM_COMPOSER_CRITIC_ENABLED",
    )
    llm_composer_repair_max_attempts: int = Field(
        default=1,
        ge=0,
        validation_alias="LLM_COMPOSER_REPAIR_MAX_ATTEMPTS",
    )
    llm_max_calls_per_match: int = Field(
        default=6,
        ge=1,
        validation_alias="LLM_MAX_CALLS_PER_MATCH",
    )
    llm_composer_multi_pass_candidate_threshold: int = Field(default=120, ge=1)
    llm_composer_multi_pass_chunk_size: int = Field(default=80, ge=1)
    stock_match_pipeline_v2: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "STOCK_MATCH_PIPELINE_V2",
            "LLM_COMPOSER_FIRST_PIPELINE",
        ),
    )
    llm_configurator_repair_enabled: bool = True
    v3_refresh_categories_before_llm: bool = Field(
        default=True,
        validation_alias="V3_REFRESH_CATEGORIES_BEFORE_LLM",
    )
    v3_full_category_contract_version: str = Field(
        default="v7_1",
        validation_alias="V3_FULL_CATEGORY_CONTRACT_VERSION",
    )
    llm_configurator_repair_max_alternatives_per_role: int = Field(default=5, ge=1)
    llm_configurator_repair_timeout_seconds: float = Field(default=300, gt=0)
    llm_configurator_no_recommendation_full_coverage_limit: int = Field(
        default=12,
        ge=1,
    )
    llm_configurator_no_recommendation_min_large_role_candidates: int = Field(
        default=12,
        ge=1,
    )
    llm_configurator_no_recommendation_min_large_role_fraction: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )
    llm_configurator_thinking_enabled: bool = False
    llm_configurator_thinking_budget_tokens: int | None = Field(default=None, ge=1)

    @field_validator("llm_configurator_thinking_budget_tokens", mode="before")
    @classmethod
    def _blank_thinking_budget_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


class WebEvidenceSettings(EnvSettings):
    web_evidence_enabled: bool = False
    web_evidence_provider: str = "disabled"
    web_evidence_mode: str = "separate"
    web_evidence_base_url: str = ""
    web_evidence_api_key: str = ""
    web_evidence_model: str = "deepseek/deepseek-v4-pro:online"
    web_evidence_max_output_tokens: int = Field(default=4096, ge=1)
    web_evidence_max_queries: int = Field(default=12, ge=0)
    web_evidence_timeout_seconds: float = Field(default=120, gt=0)
    web_evidence_cache_ttl_hours: int = Field(default=168, ge=0)
    web_evidence_max_results_per_query: int = Field(default=5, ge=1)
    web_evidence_max_snippet_chars: int = Field(default=1200, ge=0)
    web_evidence_trusted_domains: str = (
        "dell.com,i.dell.com,hpe.com,lenovo.com,asus.com,servers.asus.com,"
        "supermicro.com,intel.com,ark.intel.com,amd.com,kioxia.com,samsung.com,"
        "semiconductor.samsung.com,micron.com,gooxi.com"
    )
    tavily_api_key: str = ""


class TelegramSettings(EnvSettings):
    service_name: str = "stock-configurator"
    environment: str = "dev"
    log_level: str = "INFO"
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""
    telegram_proxy_url: str = ""
    stock_api_base_url: str = "http://stock-api:8000"
    telegram_request_timeout_seconds: float = Field(default=60, gt=0)
    telegram_v3_request_timeout_seconds: float = Field(default=1800, gt=0)


class Settings(
    OcsSettings,
    TreolanSettings,
    LlmSettings,
    WebEvidenceSettings,
    TelegramSettings,
):
    database_url: str = Field(..., description="PostgreSQL DSN, например postgresql+asyncpg://...")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_ocs_settings() -> OcsSettings:
    return OcsSettings()


@lru_cache
def get_treolan_settings() -> TreolanSettings:
    return TreolanSettings()


@lru_cache
def get_llm_settings() -> LlmSettings:
    return LlmSettings()


@lru_cache
def get_web_evidence_settings() -> WebEvidenceSettings:
    return WebEvidenceSettings()


@lru_cache
def get_telegram_settings() -> TelegramSettings:
    return TelegramSettings()
